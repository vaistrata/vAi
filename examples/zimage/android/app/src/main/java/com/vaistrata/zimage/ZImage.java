package com.vaistrata.zimage;

import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/** The native Z-Image pipeline (libzimage-jni.so); every call blocks, so run it off the UI thread. */
final class ZImage {
    static { System.loadLibrary("zimage-jni"); }

    interface Listener {
        /** A stage of a load or a generation; done out of total. */
        void onProgress(String stage, int done, int total);
    }

    /** {vai, zimage}: the shader directories the native side reads. */
    static native String[] shaderDirs();

    /** Where the native side looks for the weights (the app's external files). */
    static native String dataDir();

    /** Reads both networks' weights once; null when done, else the reason it could not. */
    static native String load(Listener listener);

    /** Whether the text encoder loaded; without it only the demo prompt's features exist. */
    static native boolean hasEncoder();

    /** The last prompt encoded, or the demo features' prompt (prompt.txt) without the encoder. */
    static native String prompt();

    /**
     * size x size ARGB pixels of this prompt (NFC, see java.text.Normalizer)
     * from this seed, or null on failure; size is a multiple of 16 from 256
     * to 1024. The prompt is encoded only when it changed.
     */
    static native int[] generate(String prompt, long seed, int size, Listener listener);

    /**
     * Stops the running generate() at its next stage (at most one DiT step
     * away); it then returns null. Callable from any thread.
     */
    static native void cancel();

    /**
     * Unpacks the APK's shaders (assets/shaders/{vai,zimage}) to where the
     * native side reads them, once per installed version.
     */
    static void unpackShaders(Context context) throws IOException {
        String version;
        try {
            version = Long.toString(context.getPackageManager()
                    .getPackageInfo(context.getPackageName(), 0).lastUpdateTime);
        } catch (PackageManager.NameNotFoundException e) {
            throw new IOException(e);
        }
        String[] dirs = shaderDirs();
        File stamp = new File(dirs[0], ".version");
        if (stamp.exists() && version.equals(
                new String(Files.readAllBytes(stamp.toPath()), StandardCharsets.UTF_8))) {
            return;
        }

        AssetManager assets = context.getAssets();
        String[] names = {"vai", "zimage"};
        byte[] buf = new byte[1 << 16];
        for (int d = 0; d < names.length; ++d) {
            File dir = new File(dirs[d]);
            if (!dir.isDirectory() && !dir.mkdirs()) {
                throw new IOException("Cannot create " + dir);
            }
            for (String name : assets.list("shaders/" + names[d])) {
                try (InputStream in = assets.open("shaders/" + names[d] + "/" + name);
                     OutputStream out = new FileOutputStream(new File(dir, name))) {
                    for (int n; (n = in.read(buf)) > 0; ) {
                        out.write(buf, 0, n);
                    }
                }
            }
        }
        Files.write(stamp.toPath(), version.getBytes(StandardCharsets.UTF_8));
    }

    private ZImage() {}
}
