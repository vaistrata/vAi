plugins {
    id("com.android.application")
}

android {
    namespace = "com.vaistrata.zimage"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.vaistrata.zimage"
        minSdk = 33
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // A local build: signed with the debug key so it installs as is.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    // build-native.sh already strips it.
    packaging { jniLibs { keepDebugSymbols += "**/libzimage-jni.so" } }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// libzimage-jni.so and the shaders come from build-native.sh (the vAi CMake
// build, as for the command-line tools), not from externalNativeBuild.
val buildNative = tasks.register<Exec>("buildNative") {
    workingDir = rootDir
    commandLine("sh", "build-native.sh")
}
tasks.named("preBuild") { dependsOn(buildNative) }
