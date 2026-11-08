package com.vaistrata.zimage;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.text.InputType;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.OutputStream;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * One screen: a seed, a Generate button, the image and how the run went.
 * {@code adb shell am start -S -n com.vaistrata.zimage/.MainActivity --el seed 42}
 * generates as soon as the weights are loaded. Launched that way it also turns
 * the screen on and shows over the lock screen, since a stopped activity's
 * process is frozen mid-run (Samsung's FreecessController does it within a minute).
 */
public final class MainActivity extends Activity {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());

    private TextView prompt;
    private EditText seed;
    private Button generate;
    private ProgressBar progress;
    private TextView status;
    private ImageView image;
    private TextView log;
    private PowerManager power;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        power = getSystemService(PowerManager.class);
        buildViews();

        // The native side reads its weights from here; creating it lets adb push to it.
        getExternalFilesDir(null);
        if (getIntent().hasExtra("seed")) {
            setShowWhenLocked(true);
            setTurnScreenOn(true);
            seed.setText(Long.toString(getIntent().getLongExtra("seed", 0)));
        }
        status.setText("Loading...");
        worker.execute(this::load);
    }

    @Override
    protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }

    private void load() {
        String error;
        try {
            ZImage.unpackShaders(this);
            error = ZImage.load(this::onProgress);
        } catch (Exception e) {
            error = "Load failed: " + e;
        }
        final String result = error;
        ui.post(() -> {
            if (result != null) {
                status.setText(result + "\n\nData directory: " + ZImage.dataDir());
                return;
            }
            String p = ZImage.prompt();
            prompt.setText(p.isEmpty() ? "(prompt.txt not found: the prepared caption features)" : p);
            status.setText("Ready. " + thermal());
            generate.setEnabled(true);
            if (getIntent().hasExtra("seed")) {
                startGenerate();
            }
        });
    }

    private void startGenerate() {
        final long s;
        try {
            s = Long.parseLong(seed.getText().toString().trim());
        } catch (NumberFormatException e) {
            status.setText("The seed must be a number");
            return;
        }
        generate.setEnabled(false);
        log.setText("");
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        worker.execute(() -> {
            long t0 = SystemClock.elapsedRealtime();
            int[] argb = ZImage.generate(s, this::onProgress);
            double seconds = (SystemClock.elapsedRealtime() - t0) / 1e3;
            Bitmap bitmap = null;
            String saved = null;
            if (argb != null) {
                bitmap = Bitmap.createBitmap(argb, ZImage.IMAGE_SIZE, ZImage.IMAGE_SIZE, Bitmap.Config.ARGB_8888);
                saved = save(bitmap, s);
            }
            final Bitmap result = bitmap;
            final String where = saved;
            ui.post(() -> {
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                generate.setEnabled(true);
                if (result == null) {
                    status.setText("Generation failed (see logcat -s zimage)");
                    return;
                }
                image.setImageBitmap(result);
                status.setText(String.format(Locale.US, "Image in %.1f s. %s%s", seconds, thermal(),
                        where != null ? "\nSaved to " + where : ""));
            });
        });
    }

    /** Called on the worker thread by the native side. */
    private void onProgress(String stage, int done, int total) {
        final String line = stage + "  [" + thermal() + "]";
        ui.post(() -> {
            progress.setMax(total);
            progress.setProgress(done);
            status.setText(stage);
            if (stage.startsWith("Step ") && stage.contains(" took ") || stage.startsWith("Decoded")) {
                log.append(line + "\n");
            }
        });
    }

    /** The system's thermal status and headroom (1.0 = throttling starts). */
    private String thermal() {
        String[] names = {"none", "light", "moderate", "severe", "critical", "emergency", "shutdown"};
        int s = power.getCurrentThermalStatus();
        float headroom = power.getThermalHeadroom(0);
        String name = s >= 0 && s < names.length ? names[s] : Integer.toString(s);
        return Float.isNaN(headroom)
                ? "thermal " + name
                : String.format(Locale.US, "thermal %s, headroom %.2f", name, headroom);
    }

    /** Into Pictures/ZImage; the gallery's path for it, or null. */
    private String save(Bitmap bitmap, long s) {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, "zimage-" + s + ".png");
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/ZImage");
        ContentResolver resolver = getContentResolver();
        Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        if (uri == null) {
            return null;
        }
        try (OutputStream out = resolver.openOutputStream(uri)) {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
        } catch (Exception e) {
            return null;
        }
        return Environment.DIRECTORY_PICTURES + "/ZImage/zimage-" + s + ".png";
    }

    private void buildViews() {
        int pad = dp(16);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("Z-Image Turbo, on this phone's GPU");
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        root.addView(title);

        prompt = new TextView(this);
        prompt.setPadding(0, dp(8), 0, dp(8));
        root.addView(prompt);

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setGravity(Gravity.CENTER_VERTICAL);
        TextView seedLabel = new TextView(this);
        seedLabel.setText("Seed ");
        row.addView(seedLabel);
        seed = new EditText(this);
        seed.setInputType(InputType.TYPE_CLASS_NUMBER);
        seed.setText(Long.toString(new Random().nextInt(1_000_000)));
        row.addView(seed, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        Button random = new Button(this);
        random.setText("Random");
        random.setOnClickListener(v -> seed.setText(Long.toString(new Random().nextInt(1_000_000))));
        row.addView(random);
        root.addView(row);

        generate = new Button(this);
        generate.setText("Generate");
        generate.setEnabled(false);
        generate.setOnClickListener(v -> startGenerate());
        root.addView(generate);

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        root.addView(progress);

        status = new TextView(this);
        status.setPadding(0, dp(8), 0, dp(8));
        root.addView(status);

        image = new ImageView(this);
        image.setAdjustViewBounds(true);
        image.setScaleType(ImageView.ScaleType.FIT_CENTER);
        root.addView(image, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        log = new TextView(this);
        log.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        log.setPadding(0, dp(8), 0, 0);
        root.addView(log);

        ScrollView scroll = new ScrollView(this);
        scroll.setFitsSystemWindows(true);
        scroll.addView(root);
        setContentView(scroll);
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }
}
