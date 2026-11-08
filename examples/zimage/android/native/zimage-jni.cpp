/*=============================================================================
  Written by   ippx123@gmail.com
  Task         Z-Image on Android — the native side of the app
               (com.vaistrata.zimage.ZImage): caption features and a seeded
               noise latent -> 9 Turbo steps on the DiT -> VAE decode, all on
               the GPU, returned as ARGB pixels.
  Assets       under VAI_DATA_DIR (the app's external files, see
               CMakeLists.txt and README.md):
               zimage/Z-Image-Turbo-q4-split.safetensors  (prepared DiT)
               zimage/Z-Image-Turbo/vae/                  (VAE decoder)
               zimage/demo-input.safetensors              (caption features)
               zimage/prompt.txt                          (optional, shown)
=============================================================================*/

#include "models/zimage/model.h"
#include "models/zimage/sampler.h"
#include "models/zimage/tongyi-weights.h"
#include "models/zimage/tongyi-vae-weights.h"
#include "vai/engine/safetensors-parser.h"

#include <android/log.h>
#include <jni.h>

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
constexpr uint32_t kLatentSize = 64;   // 512x512 images

struct Engine
{
    zimage::ModelConfig cfg = zimage::ModelConfig::Turbo();
    std::unique_ptr<zimage::Model> dit;
    std::unique_ptr<zimage::VaeDecoder> vae;
    Tensor capFeats;
    std::string prompt;
};

std::mutex gMutex;
std::unique_ptr<Engine> gEngine;

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

    void operator()(const std::string& stage, int done, int total) const
    {
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

// Builds both networks and reads their weights; empty on success, else why not.
std::string load(const Progress& progress)
{
    auto engine = std::make_unique<Engine>();
    const std::string data = VAI_DATA_DIR "/zimage";

    const std::string input = data + "/demo-input.safetensors";
    if (!std::filesystem::exists(input)) return missing(input);
    const std::string vaeDir = data + "/Z-Image-Turbo/vae";
    if (!std::filesystem::exists(vaeDir)) return missing(vaeDir);

    {
        SafetensorsParser parser(input.c_str());
        engine->capFeats = Tensor(parser["cap_feats"], Dtype::Fp32);
    }
    std::ifstream promptFile(data + "/prompt.txt");
    std::getline(promptFile, engine->prompt);

    eva::Device device = getVaiDevice();
    engine->dit = std::make_unique<zimage::Model>(NeuralNetConfig{
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
        .nodesPerSubmit = 1,                    // Adreno's submission watchdog
    }, engine->cfg, zimage::RuntimeConfig{
        .blockWeightDtype = Dtype::Q4,
        .splitFfnUp = true,
    });
    const std::string prepared = zimage::tongyi::preparedPath(*engine->dit);
    if (!std::filesystem::exists(prepared)) return missing(prepared);

    auto t0 = std::chrono::steady_clock::now();
    progress("Loading the DiT", 0, 2);
    zimage::tongyi::loadPrepared(*engine->dit, prepared);
    __android_log_print(ANDROID_LOG_INFO, kTag, "DiT weights in %.0f ms", msSince(t0));

    t0 = std::chrono::steady_clock::now();
    progress("Loading the VAE", 1, 2);
    engine->vae = std::make_unique<zimage::VaeDecoder>(NeuralNetConfig{
        .device = device,
        .wDefaultDtype = Dtype::Fp16,
        .oDefaultDtype = Dtype::Fp32,
        .nodesPerSubmit = 1,
    });
    loadWeights<"tongyi">(*engine->vae);
    __android_log_print(ANDROID_LOG_INFO, kTag, "VAE weights in %.0f ms", msSince(t0));
    progress("Ready", 2, 2);

    gEngine = std::move(engine);
    return {};
}

std::vector<jint> generate(Engine& e, uint64_t seed, const Progress& progress)
{
    const uint32_t H = kLatentSize, W = kLatentSize;
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

    progress("Decoding with the VAE", steps, steps + 1);
    t0 = std::chrono::steady_clock::now();
    Tensor image;
    (*e.vae)({{"inLatent", zimage::decoderInput(e.vae->vaeCfg, latent.data(), H, W)}}, {{"outImage", &image}});
    const uint32_t imgH = image.shape()[1], imgW = image.shape()[2];
    const std::vector<uint8_t> rgb = zimage::toRgb8(image.getData<float>(), imgH, imgW);
    std::ostringstream s;
    s.precision(1);
    s << std::fixed << "Decoded in " << msSince(t0) / 1e3 << " s";
    progress(s.str(), steps + 1, steps + 1);

    std::vector<jint> argb(size_t(imgH) * imgW);
    for (size_t i = 0; i < argb.size(); ++i)
        argb[i] = jint(0xff000000u | uint32_t(rgb[3 * i]) << 16 | uint32_t(rgb[3 * i + 1]) << 8 | rgb[3 * i + 2]);
    return argb;
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
    try {
        return toJava(env, load(Progress(env, listener)));
    } catch (const std::exception& e) {
        return toJava(env, std::string("Load failed: ") + e.what());
    }
}

JNIEXPORT jstring JNICALL
Java_com_vaistrata_zimage_ZImage_prompt(JNIEnv* env, jclass)
{
    std::lock_guard lock(gMutex);
    return env->NewStringUTF(gEngine ? gEngine->prompt.c_str() : "");
}

// ARGB pixels of a 512x512 image, or null on failure.
JNIEXPORT jintArray JNICALL
Java_com_vaistrata_zimage_ZImage_generate(JNIEnv* env, jclass, jlong seed, jobject listener)
{
    std::lock_guard lock(gMutex);
    if (!gEngine) return nullptr;
    try {
        const std::vector<jint> argb = generate(*gEngine, uint64_t(seed), Progress(env, listener));
        jintArray out = env->NewIntArray(jsize(argb.size()));
        env->SetIntArrayRegion(out, 0, jsize(argb.size()), argb.data());
        return out;
    } catch (const std::exception& e) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "generate failed: %s", e.what());
        return nullptr;
    }
}

} // extern "C"
