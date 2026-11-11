/*=============================================================================
  Written by   chaudharyshivansh9@gmail.com
  Task         Whisper — transcribes a WAV file in 30 s windows, profiles
               encode and decode.
               Usage: demo-whisper [audio.wav] [tiny|base|small|medium|large-v3|large-v3-turbo]
  Assets       data/whisper/ckpts/<variant>/model.safetensors
               data/whisper/ckpts/<variant>/tokenizer.json
                 from  huggingface.co/openai/whisper-<variant>
               data/whisper/jfk.wav (included)
                 from  github.com/ggml-org/whisper.cpp (samples/jfk.wav)
=============================================================================*/

#include "models/whisper/model.h"
#include "models/whisper/openai-weights.h"
#include "models/whisper/processing.h"
#include "vai/utils/tokenizer.h"
#include "vai/utils/time-checker.h"
#include "eva-runtime.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vai;


// PCM16 / float32 WAV -> mono 16 kHz floats in [-1, 1]
std::vector<float> readWav(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot open " + path);
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
    if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) || std::memcmp(bytes.data() + 8, "WAVE", 4))
        throw std::runtime_error(path + " is not a RIFF/WAVE file");

    auto u16 = [&](size_t at) { uint16_t v; std::memcpy(&v, bytes.data() + at, 2); return v; };
    auto u32 = [&](size_t at) { uint32_t v; std::memcpy(&v, bytes.data() + at, 4); return v; };

    uint16_t format = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const char* data = nullptr;
    size_t dataBytes = 0;

    // Walk the chunks: fmt and data may sit behind LIST and others
    for (size_t at = 12; at + 8 <= bytes.size();)
    {
        const uint32_t size = u32(at + 4);
        const size_t body = at + 8;
        if (!std::memcmp(bytes.data() + at, "fmt ", 4))
        {
            format = u16(body);
            channels = u16(body + 2);
            rate = u32(body + 4);
            bits = u16(body + 14);
            if (format == 0xFFFE && size >= 26)   // WAVE_FORMAT_EXTENSIBLE: subformat's first word
                format = u16(body + 24);
        }
        else if (!std::memcmp(bytes.data() + at, "data", 4))
        {
            data = bytes.data() + body;
            dataBytes = std::min<size_t>(size, bytes.size() - body);
        }
        at = body + size + (size & 1);
    }

    const bool pcm16 = format == 1 && bits == 16;
    const bool f32   = format == 3 && bits == 32;
    if (!data || !channels || !(pcm16 || f32))
        throw std::runtime_error(path + ": only 16-bit PCM or 32-bit float WAV is supported");

    const size_t frames = dataBytes / (bits / 8) / channels;
    std::vector<float> mono(frames, 0.0f);
    for (size_t i = 0; i < frames; ++i)
    {
        for (uint16_t c = 0; c < channels; ++c)
        {
            const size_t k = i * channels + c;
            float v;
            if (pcm16) { int16_t s; std::memcpy(&s, data + k * 2, 2); v = s / 32768.0f; }
            else       { std::memcpy(&v, data + k * 4, 4); }
            mono[i] += v;
        }
        mono[i] /= channels;
    }

    if (rate == whisper::kSampleRate)
        return mono;

    // Linear resampling: fine for a demo, a proper filter does better
    printf("Resampling %u Hz -> %u Hz\n", rate, whisper::kSampleRate);
    const double step = double(rate) / whisper::kSampleRate;
    std::vector<float> out(size_t(frames / step));
    for (size_t i = 0; i < out.size(); ++i)
    {
        const double pos = i * step;
        const size_t j = size_t(pos);
        const float frac = float(pos - j);
        out[i] = mono[j] * (1.0f - frac) + mono[std::min(j + 1, frames - 1)] * frac;
    }
    return out;
}


void setProfileTags(whisper::Model& model)
{
    const uint32_t D = model.modelCfg.embedDim();
    const uint32_t V = model.modelCfg.vocabSize;

    auto lin = [](uint32_t in, uint32_t out) {
        return "Linear.[" + std::to_string(in) + "x" + std::to_string(out) + "]";
    };

    auto& enc = model.encoder;
    enc.conv1.setProfileTag("Conv");
    enc.conv2.setProfileTag("Conv");
    for (auto& b : enc.transformer.layers)
    {
        b.norm1.setProfileTag("LayerNorm");
        b.mhAttn.toQ.setProfileTag(lin(D, D));
        b.mhAttn.toK.setProfileTag(lin(D, D));
        b.mhAttn.toV.setProfileTag(lin(D, D));
        b.mhAttn.attn.setProfileTag("Attn");
        b.mhAttn.toOut.setProfileTag(lin(D, D));
        b.norm2.setProfileTag("LayerNorm");
        b.ffn.fc1.setProfileTag(lin(D, 4 * D));
        b.ffn.fc2.setProfileTag(lin(4 * D, D));
    }
    enc.transformer.finalNorm.setProfileTag("LayerNorm");
    for (size_t i = 0; i < enc.crossK.size(); ++i)
    {
        enc.crossK[i].setProfileTag("CrossKV " + lin(D, D));
        enc.crossV[i].setProfileTag("CrossKV " + lin(D, D));
    }

    auto& dec = model.decoder;
    for (auto& b : dec.layers)
    {
        b.selfNorm.setProfileTag("LayerNorm");
        b.selfAttn.toQ.setProfileTag(lin(D, D));
        b.selfAttn.toK.setProfileTag(lin(D, D));
        b.selfAttn.toV.setProfileTag(lin(D, D));
        b.selfAttn.attn.setProfileTag("SelfAttn");
        b.selfAttn.toOut.setProfileTag(lin(D, D));
        b.crossNorm.setProfileTag("LayerNorm");
        b.crossQ.setProfileTag(lin(D, D));
        b.crossAttn.setProfileTag("CrossAttn");
        b.crossOut.setProfileTag(lin(D, D));
        b.ffnNorm.setProfileTag("LayerNorm");
        b.ffn.fc1.setProfileTag(lin(D, 4 * D));
        b.ffn.fc2.setProfileTag(lin(4 * D, D));
    }
    dec.finalNorm.setProfileTag("LayerNorm");
    dec.lmHead.setProfileTag(lin(D, V));
    dec.argmax.setProfileTag("Argmax");
}


int main(int argc, char** argv)
{
    const std::string wavPath = argc > 1 ? argv[1] : VAI_DATA_DIR "/whisper/jfk.wav";
    const whisper::ModelConfig modelCfg = whisper::ModelConfig::byName(argc > 2 ? argv[2] : "tiny");
    const std::string ckptDir = std::string(VAI_DATA_DIR "/whisper/ckpts/") + modelCfg.variant;

    // Initialize device
    eva::Device device = getVaiDevice();

    // Load model
    printf("Loading Whisper-%s model...\n", modelCfg.variant);
    whisper::Model model({
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
    }, modelCfg);

    // Configure the per-node GPU execution-time profiler
    setProfileTags(model);

    // Load weights
    printf("Loading weights...\n");
    {
        TimeChecker loadTimer("Weights loaded successfully in {}ms");
        loadWeights<"openai">(model);
    }

    // Load tokenizer; special tokens are looked up since large-v3 shifts their ids
    Tokenizer tokenizer(ckptDir + "/tokenizer.json");
    auto special = [&](const char* piece) {
        const int id = tokenizer.pieceToId(piece);
        if (id < 0) throw std::runtime_error(std::string("tokenizer has no ") + piece);
        return id;
    };
    const std::vector<int> prompt = {
        special("<|startoftranscript|>"), special("<|en|>"),
        special("<|transcribe|>"), special("<|notimestamps|>")};
    const uint32_t eot = (uint32_t)special("<|endoftext|>");
    const uint32_t maxNewTokens = modelCfg.textContext / 2;   // Whisper's own sample_len

    // Load audio
    std::vector<float> audio = readWav(wavPath);
    const uint32_t numWindows = (uint32_t)CEIL_DIV(audio.size(), (size_t)whisper::kWindowSamples);
    printf("\nAudio: \"%s\" (%.2f s, %u window%s)\n", wavPath.c_str(),
        audio.size() / float(whisper::kSampleRate), numWindows, numWindows > 1 ? "s" : "");

    // Initialize the input tensors, reused by every window
    Tensor mel({modelCfg.numMels, 1, whisper::kWindowFrames});
    Tensor promptTokenIds = Tensor({(uint32_t)prompt.size()}).setData(prompt.data());
    Tensor nextTokenId;

    // Warm-up: compiles the pipelines and records both nets
    {
        mel.setFloats(whisper::logMel(audio.data(), audio.size(), modelCfg.numMels));
        model.encode(mel);
        model.prefill(promptTokenIds, &nextTokenId);
        model.decode(&nextTokenId);
        nextTokenId.getData<uint32_t>();
    }

    model.resetProfileRecords();

    std::string transcript;
    uint32_t totalTokens = 0;
    double melMs = 0.0, encodeMs = 0.0, decodeMs = 0.0;
    auto msSince = [](auto t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    };

    printf("\n=== Transcript ===\n");
    for (uint32_t w = 0; w < numWindows; ++w)
    {
        const size_t start = size_t(w) * whisper::kWindowSamples;

        auto t0 = std::chrono::high_resolution_clock::now();
        mel.setFloats(whisper::logMel(audio.data() + start, audio.size() - start, modelCfg.numMels));
        melMs += msSince(t0);

        t0 = std::chrono::high_resolution_clock::now();
        model.encode(mel);
        model.encoder.sync();
        encodeMs += msSince(t0);

        // Greedy decode until <|endoftext|>
        t0 = std::chrono::high_resolution_clock::now();
        std::vector<int> ids;
        model.prefill(promptTokenIds, &nextTokenId);
        for (uint32_t n = 0; n < maxNewTokens; ++n)
        {
            const uint32_t id = nextTokenId.getData<uint32_t>()[0];
            if (id == eot)
                break;
            ids.push_back((int)id);
            printf("%s", tokenizer.decode({(int)id}).c_str());
            fflush(stdout);
            model.decode(&nextTokenId);
        }
        decodeMs += msSince(t0);
        totalTokens += (uint32_t)ids.size();
        transcript += tokenizer.decode(ids);
    }
    printf("\n");

    printf("\n"
        "────────────────────────────────────────\n"
        "▶ Log-mel      %8.1f ms (CPU)\n"
        "▶ Encode       %8.1f ms (%u window%s)\n"
        "▶ Decode       %8.1f ms (%u tokens, %.1f tokens/s)\n"
        "────────────────────────────────────────\n",
        melMs, encodeMs, numWindows, numWindows > 1 ? "s" : "",
        decodeMs, totalTokens, totalTokens / (decodeMs / 1000.0));

    printf("\n[Encoder]");
    model.encoder.printProfile();
    printf("\n[Decoder]");
    model.decoder.printProfile();

    printf("\n--- Transcription complete ---\n");
    return 0;
}
