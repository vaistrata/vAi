/*=============================================================================
  Written by   ippx123@gmail.com
  Task         Z-Image on Android — the native side of the app
               (com.vaistrata.zimage.ZImage): a prompt through the caption
               encoder (Qwen3) and a seeded noise latent -> 8 Turbo steps on
               the DiT -> VAE decode at 512, 768 or 1024 square, all on the GPU, returned as ARGB pixels.
  Assets       under VAI_DATA_DIR (the app's external files, see
               CMakeLists.txt and README.md):
               zimage/Z-Image-Turbo-q4-split.safetensors     (prepared DiT)
               zimage/Z-Image-Turbo/vae/                     (VAE decoder)
               zimage/Z-Image-Turbo-text-encoder-q4.safetensors
                                                             (prepared encoder)
               zimage/Z-Image-Turbo/tokenizer/tokenizer.json
               Without the encoder the app falls back to the fixed caption
               features of zimage/demo-input.safetensors and zimage/prompt.txt.
=============================================================================*/

#include "models/zimage/model.h"
#include "models/zimage/sampler.h"
#include "models/zimage/tongyi-weights.h"
#include "models/zimage/tongyi-vae-weights.h"
#include "models/zimage/tongyi-text-encoder-weights.h"
#include "vai/engine/safetensors-parser.h"

#include <android/log.h>
#include <jni.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace vai;

namespace {

constexpr const char* kTag = "zimage";
constexpr uint32_t kVaeScale = 8;      // image pixels per latent position
// Past 512x512 the decode runs alone: the DiT goes before it, and the VAE,
// with the feature maps its buffer pool keeps, after it.
constexpr uint32_t kKeepNetsUpTo = 512;

// The DiT (3.7 GB) and the text encoder (2.0 GB) are never in memory at
// once: with both and the VAE, a phone running other apps has the low-memory
// killer take the app. A new prompt drops the DiT, encodes with an encoder
// built for it and dropped after, then reads the DiT again (a few seconds).
struct Engine
{
    eva::Device device;
    zimage::ModelConfig cfg = zimage::ModelConfig::Turbo();
    std::unique_ptr<zimage::Model> dit;             // null until needed
    std::unique_ptr<zimage::VaeDecoder> vae;
    std::unique_ptr<Tokenizer> tokenizer;           // null: the demo features only
    Tensor capFeats;                                // of `prompt`
    std::string prompt;
};

std::mutex gMutex;
std::unique_ptr<Engine> gEngine;

// Set by ZImage.cancel() (without the mutex, which generate holds); a
// generation stops at its next stage, at most one DiT step away.
std::atomic<bool> gCancel{false};
struct Cancelled : std::exception
{
    const char* what() const noexcept override { return "cancelled"; }
};

// ZImage.Listener.onProgress(stage, done, total) on the calling thread.
struct Progress
{
    JNIEnv* env;
    jobject listener;
    jmethodID method;

    Progress(JNIEnv* env, jobject listener)
    : env(env), listener(listener)
    , method(env->GetMethodID(env->GetObjectClass(listener), "onProgress", "(Ljava/lang/String;II)V"))
    {}

    // Throws Cancelled once ZImage.cancel() was called, unless the stage is
    // the last. Only between stages: no GPU work is in flight then.
    void operator()(const std::string& stage, int done, int total, bool cancellable = true) const
    {
        if (cancellable && gCancel.load())
            throw Cancelled();
        __android_log_print(ANDROID_LOG_INFO, kTag, "%s (%d/%d)", stage.c_str(), done, total);
        jstring s = env->NewStringUTF(stage.c_str());
        env->CallVoidMethod(listener, method, s, done, total);
        env->DeleteLocalRef(s);
    }
};

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string missing(const std::string& path)
{
    return "Missing " + path + "\nPush the weights as examples/zimage/android/README.md describes.";
}

NeuralNetConfig netConfig(eva::Device device)
{
    return {
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
        .nodesPerSubmit = 1,                    // Adreno's submission watchdog
    };
}

std::unique_ptr<zimage::Model> makeDit(const Engine& e)
{
    return std::make_unique<zimage::Model>(netConfig(e.device), e.cfg, zimage::RuntimeConfig{
        .blockWeightDtype = Dtype::Q4,
        .splitFfnUp = true,
    });
}

// Frees the DiT. The caption features go first: fed to the DiT, they hold
// the device buffer they were uploaded to, and that buffer keeps the DiT's
// whole buffer pool alive (weights included), so resetting the DiT alone
// freed nothing. A host-only copy takes their place.
void dropDit(Engine& e)
{
    if (e.capFeats.numElements() > 0)
        e.capFeats = Tensor(e.capFeats.shape(), Dtype::Fp32)
                         .setFloats(static_cast<const float*>(e.capFeats.hostData()));
    e.dit.reset();
}

void loadVae(Engine& e)
{
    auto t0 = std::chrono::steady_clock::now();
    e.vae = std::make_unique<zimage::VaeDecoder>(netConfig(e.device));
    loadWeights<"tongyi">(*e.vae);
    __android_log_print(ANDROID_LOG_INFO, kTag, "VAE weights in %.0f ms", msSince(t0));
}

void loadDit(Engine& e, const Progress& progress, int done, int total)
{
    auto t0 = std::chrono::steady_clock::now();
    progress("Loading the DiT", done, total);
    e.dit = makeDit(e);
    zimage::tongyi::loadPrepared(*e.dit, zimage::tongyi::preparedPath(*e.dit));
    __android_log_print(ANDROID_LOG_INFO, kTag, "DiT weights in %.0f ms", msSince(t0));
}

// Checks the files and reads the VAE (and the DiT, without the encoder);
// empty on success, else why not.
std::string load(const Progress& progress)
{
    auto engine = std::make_unique<Engine>();
    engine->device = getVaiDevice();
    const std::string data = VAI_DATA_DIR "/zimage";

    const std::string vaeDir = data + "/Z-Image-Turbo/vae";
    if (!std::filesystem::exists(vaeDir)) return missing(vaeDir);
    const std::string ditFile = zimage::tongyi::preparedPath(*makeDit(*engine));
    if (!std::filesystem::exists(ditFile)) return missing(ditFile);

    const std::string encoderFile = zimage::tongyi::preparedTextEncoderPath(
        zimage::TextEncoder(netConfig(engine->device)));
    const std::string tokenizerFile = zimage::tongyi::tokenizerPath();
    const bool withEncoder = std::filesystem::exists(encoderFile) && std::filesystem::exists(tokenizerFile);
    if (withEncoder)
    {
        // The app hands over NFC text (java.text.Normalizer).
        engine->tokenizer = std::make_unique<Tokenizer>(tokenizerFile, /*inputIsNfc*/ true);
    }
    else
    {
        const std::string input = data + "/demo-input.safetensors";
        if (!std::filesystem::exists(input)) return missing(encoderFile);
        SafetensorsParser parser(input.c_str());
        engine->capFeats = Tensor(parser["cap_feats"], Dtype::Fp32);
        std::ifstream promptFile(data + "/prompt.txt");
        std::getline(promptFile, engine->prompt);
    }

    const int parts = withEncoder ? 1 : 2;
    progress("Loading the VAE", 0, parts);
    loadVae(*engine);
    if (!withEncoder)
        loadDit(*engine, progress, 1, parts);
    progress("Ready", parts, parts);

    gEngine = std::move(engine);
    return {};
}

// The prompt's caption features, through an encoder that lives for this call.
void encode(Engine& e, const std::string& prompt, const Progress& progress)
{
    dropDit(e);
    auto t0 = std::chrono::steady_clock::now();
    progress("Loading the text encoder", 0, 2);
    zimage::TextEncoder encoder(netConfig(e.device));
    zimage::tongyi::loadPreparedTextEncoder(encoder, zimage::tongyi::preparedTextEncoderPath(encoder));
    __android_log_print(ANDROID_LOG_INFO, kTag, "Text encoder weights in %.0f ms", msSince(t0));

    t0 = std::chrono::steady_clock::now();
    progress("Encoding the prompt", 1, 2);
    const std::vector<int> ids = zimage::captionTokens(encoder.encCfg, *e.tokenizer, prompt);
    e.capFeats = zimage::encodeCaption(encoder, ids);
    e.prompt = prompt;
    std::ostringstream s;
    s.precision(2);
    s << std::fixed << "Encoded " << ids.size() << " tokens in " << msSince(t0) / 1e3 << " s";
    progress(s.str(), 2, 2);
}

// size x size ARGB pixels; size a multiple of 16 (the DiT's 2x2 patches of
// the latent).
std::vector<jint> generate(Engine& e, const std::string& prompt, uint64_t seed, uint32_t size,
                           const Progress& progress)
{
    const uint32_t H = size / kVaeScale, W = size / kVaeScale;
    if (e.tokenizer && (prompt != e.prompt || e.capFeats.numElements() == 0))
        encode(e, prompt, progress);
    if (!e.dit)
        loadDit(e, progress, 0, 1);

    std::vector<float> latent = zimage::gaussianNoise(e.cfg, H, W, seed);
    const std::vector<float> sigmas = zimage::turboSigmas();
    const int steps = int(sigmas.size() - 1);

    progress("Step 1 (uploads the weights first)", 0, steps + 1);
    auto t0 = std::chrono::steady_clock::now();
    zimage::sampleTurbo(*e.dit, e.capFeats, latent, H, W, sigmas, [&](uint32_t i, float) {
        std::ostringstream s;
        s.precision(1);
        s << std::fixed << "Step " << i + 1 << " took " << msSince(t0) / 1e3 << " s";
        progress(s.str(), int(i) + 1, steps + 1);
        t0 = std::chrono::steady_clock::now();
    });

    // The decode's feature maps grow with the image (at 768, 128 x 768 x 768
    // fp32 is 0.3 GB, 1.8 GB of device memory in all) and do not fit beside
    // the DiT: the low-memory killer took the app mid-decode. The next image
    // reads both again (~4.5 s).
    const bool alone = size > kKeepNetsUpTo;
    if (alone)
        dropDit(e);
    if (!e.vae)
        loadVae(e);

    progress("Decoding with the VAE", steps, steps + 1);
    t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> rgb;
    uint32_t imgH, imgW;
    {
        Tensor image;
        (*e.vae)({{"inLatent", zimage::decoderInput(e.vae->vaeCfg, latent.data(), H, W)}}, {{"outImage", &image}});
        imgH = image.shape()[1], imgW = image.shape()[2];
        rgb = zimage::toRgb8(image.getData<float>(), imgH, imgW);
    }
    if (alone)
        e.vae.reset();
    std::ostringstream s;
    s.precision(1);
    s << std::fixed << "Decoded in " << msSince(t0) / 1e3 << " s";
    progress(s.str(), steps + 1, steps + 1, /*cancellable*/ false);

    std::vector<jint> argb(size_t(imgH) * imgW);
    for (size_t i = 0; i < argb.size(); ++i)
        argb[i] = jint(0xff000000u | uint32_t(rgb[3 * i]) << 16 | uint32_t(rgb[3 * i + 1]) << 8 | rgb[3 * i + 2]);
    return argb;
}

// UTF-16 -> UTF-8, surrogate pairs joined (emoji); GetStringUTFChars would
// give modified UTF-8, which spells those as two 3-byte halves.
std::string utf8(JNIEnv* env, jstring js)
{
    std::string out;
    if (!js) return out;
    const jsize n = env->GetStringLength(js);
    std::vector<jchar> u(n);
    env->GetStringRegion(js, 0, n, u.data());
    for (jsize i = 0; i < n; ++i)
    {
        uint32_t c = u[i];
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < n && u[i + 1] >= 0xDC00 && u[i + 1] < 0xE000)
            c = 0x10000 + ((c - 0xD800) << 10) + (u[++i] - 0xDC00);
        if (c < 0x80) out += char(c);
        else if (c < 0x800) { out += char(0xC0 | c >> 6); out += char(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { out += char(0xE0 | c >> 12); out += char(0x80 | (c >> 6 & 0x3F)); out += char(0x80 | (c & 0x3F)); }
        else { out += char(0xF0 | c >> 18); out += char(0x80 | (c >> 12 & 0x3F)); out += char(0x80 | (c >> 6 & 0x3F)); out += char(0x80 | (c & 0x3F)); }
    }
    return out;
}

jstring toJava(JNIEnv* env, const std::string& s)
{
    return s.empty() ? nullptr : env->NewStringUTF(s.c_str());
}

} // namespace


extern "C" {

// {vai shader dir, zimage shader dir}: where ZImage unpacks the APK's shaders.
JNIEXPORT jobjectArray JNICALL
Java_com_vaistrata_zimage_ZImage_shaderDirs(JNIEnv* env, jclass)
{
    jobjectArray dirs = env->NewObjectArray(2, env->FindClass("java/lang/String"), nullptr);
    env->SetObjectArrayElement(dirs, 0, env->NewStringUTF(SHADER_OUTPUT_DIR));
    env->SetObjectArrayElement(dirs, 1, env->NewStringUTF(ZIMAGE_SHADER_OUTPUT_DIR));
    return dirs;
}

JNIEXPORT jstring JNICALL
Java_com_vaistrata_zimage_ZImage_dataDir(JNIEnv* env, jclass)
{
    return env->NewStringUTF(VAI_DATA_DIR);
}

// null once loaded, else the reason it could not be.
JNIEXPORT jstring JNICALL
Java_com_vaistrata_zimage_ZImage_load(JNIEnv* env, jclass, jobject listener)
{
    std::lock_guard lock(gMutex);
    if (gEngine) return nullptr;
    gCancel = false;
    try {
        return toJava(env, load(Progress(env, listener)));
    } catch (const std::exception& e) {
        return toJava(env, std::string("Load failed: ") + e.what());
    }
}

// Whether a prompt can be typed; without the encoder, the fixed demo prompt.
JNIEXPORT jboolean JNICALL
Java_com_vaistrata_zimage_ZImage_hasEncoder(JNIEnv*, jclass)
{
    std::lock_guard lock(gMutex);
    return gEngine && gEngine->tokenizer;
}

JNIEXPORT jstring JNICALL
Java_com_vaistrata_zimage_ZImage_prompt(JNIEnv* env, jclass)
{
    std::lock_guard lock(gMutex);
    return env->NewStringUTF(gEngine ? gEngine->prompt.c_str() : "");
}

// ARGB pixels of a size x size image of this prompt (NFC; ignored without
// the encoder), or null on failure.
JNIEXPORT jintArray JNICALL
Java_com_vaistrata_zimage_ZImage_generate(JNIEnv* env, jclass, jstring jprompt, jlong seed, jint size,
                                          jobject listener)
{
    std::lock_guard lock(gMutex);
    if (!gEngine || size < 256 || size > 1024 || size % 16 != 0) return nullptr;
    gCancel = false;
    try {
        const std::string prompt = utf8(env, jprompt);
        const std::vector<jint> argb = generate(*gEngine, prompt, uint64_t(seed), uint32_t(size),
                                                Progress(env, listener));
        jintArray out = env->NewIntArray(jsize(argb.size()));
        env->SetIntArrayRegion(out, 0, jsize(argb.size()), argb.data());
        return out;
    } catch (const Cancelled&) {
        __android_log_print(ANDROID_LOG_INFO, kTag, "generate cancelled");
        return nullptr;
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "generate failed: %s", e.what());
        return nullptr;
    }
}

// Stops the running generate at its next stage; it then returns null.
JNIEXPORT void JNICALL
Java_com_vaistrata_zimage_ZImage_cancel(JNIEnv*, jclass)
{
    gCancel = true;
}

} // extern "C"
