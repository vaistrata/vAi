package com.vaistrata.zimage;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.text.InputType;
import android.util.Size;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * One screen: a prompt, a seed, the image size, a Generate button (Cancel while
 * it runs), the image, how the run went, and the images made so far.
 * {@code adb shell am start -S -n com.vaistrata.zimage/.MainActivity --el seed 42 --es prompt "..."}
 * (and optionally {@code --ei size 768}) generates as soon as the weights are
 * loaded. Launched that way it also turns the screen on and shows over the lock
 * screen, since a stopped activity's process is frozen mid-run (Samsung's
 * FreecessController does it within a minute).
 */
public final class MainActivity extends Activity {
    private static final int[] SIZES = {512, 768, 1024};
    private static final int GALLERY_MAX = 40;
    private static final String GALLERY_DIR = Environment.DIRECTORY_PICTURES + "/ZImage/";

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final ExecutorService thumbs = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());

    private EditText prompt;
    private EditText seed;
    private RadioGroup sizes;
    private Button generate;
    private ProgressBar progress;
    private TextView status;
    private TextView times;
    private ImageView image;
    private TextView log;
    private LinearLayout gallery;
    private PowerManager power;
    private int adbSize;   // a size from adb that has no button, else 0

    // The running generation, on the UI thread.
    private boolean running;
    private volatile boolean cancelRequested;
    private long runStart, firstStepDone;
    private int stepsDone, stepsTotal;
    private final Runnable ticker = new Runnable() {
        @Override
        public void run() {
            if (!running) {
                return;
            }
            times.setText(timing());
            ui.postDelayed(this, 1000);
        }
    };

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
            String p = getIntent().getStringExtra("prompt");
            if (p != null) {
                prompt.setText(p);
            }
            int s = getIntent().getIntExtra("size", 512);
            if (!checkSize(s)) {
                sizes.clearCheck();
                adbSize = s;
            }
        }
        status.setText("Loading...");
        final boolean autoRun = state == null && getIntent().hasExtra("seed");
        worker.execute(() -> load(autoRun));
        refreshGallery();
    }

    @Override
    protected void onDestroy() {
        if (running) {
            ZImage.cancel();
        }
        worker.shutdownNow();
        thumbs.shutdownNow();
        super.onDestroy();
    }

    private void load(boolean autoRun) {
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
            if (!ZImage.hasEncoder()) {
                // The fixed demo features: their prompt is shown, not edited.
                String p = ZImage.prompt();
                prompt.setText(p.isEmpty() ? "(no text encoder: the demo caption features)" : p);
                prompt.setEnabled(false);
            }
            status.setText("Ready. " + thermal());
            generate.setEnabled(true);
            if (autoRun) {
                startGenerate();
            }
        });
    }

    /** Checks the button for this size; false if there is none. */
    private boolean checkSize(int s) {
        for (int i = 0; i < SIZES.length; ++i) {
            if (SIZES[i] == s) {
                sizes.check(i + 1);
                return true;
            }
        }
        return false;
    }

    private int selectedSize() {
        int id = sizes.getCheckedRadioButtonId();
        return id >= 1 && id <= SIZES.length ? SIZES[id - 1] : (adbSize > 0 ? adbSize : 512);
    }

    private void onGenerateClicked() {
        if (!running) {
            startGenerate();
            return;
        }
        cancelRequested = true;
        ZImage.cancel();
        generate.setEnabled(false);
        generate.setText("Cancelling...");
        status.setText("Cancelling after this stage...");
    }

    private void startGenerate() {
        final long s;
        try {
            s = Long.parseLong(seed.getText().toString().trim());
        } catch (NumberFormatException e) {
            status.setText("The seed must be a number");
            return;
        }
        final String text = Normalizer.normalize(prompt.getText().toString().trim(), Normalizer.Form.NFC);
        if (ZImage.hasEncoder() && text.isEmpty()) {
            status.setText("Type a prompt");
            return;
        }
        final int n = selectedSize();
        setRunning(true);
        log.setText("");
        worker.execute(() -> {
            long t0 = SystemClock.elapsedRealtime();
            int[] argb = ZImage.generate(text, s, n, this::onProgress);
            double seconds = (SystemClock.elapsedRealtime() - t0) / 1e3;
            Bitmap bitmap = null;
            String saved = null;
            if (argb != null) {
                bitmap = Bitmap.createBitmap(argb, n, n, Bitmap.Config.ARGB_8888);
                saved = save(bitmap, text, s, n);
            }
            final Bitmap result = bitmap;
            final String where = saved;
            ui.post(() -> {
                final boolean cancelled = cancelRequested;
                setRunning(false);
                if (result == null) {
                    status.setText(cancelled ? "Cancelled. " + thermal()
                            : "Generation failed (see logcat -s zimage)");
                    return;
                }
                image.setImageBitmap(result);
                status.setText(String.format(Locale.US, "%dx%d in %s. %s%s", n, n, clock(seconds), thermal(),
                        where != null ? "\nSaved to " + where : ""));
                refreshGallery();
            });
        });
    }

    private void setRunning(boolean on) {
        running = on;
        cancelRequested = false;
        prompt.setEnabled(!on && ZImage.hasEncoder());
        seed.setEnabled(!on);
        for (int i = 0; i < sizes.getChildCount(); ++i) {
            sizes.getChildAt(i).setEnabled(!on);
        }
        generate.setEnabled(true);
        generate.setText(on ? "Cancel" : "Generate");
        if (on) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            runStart = SystemClock.elapsedRealtime();
            firstStepDone = 0;
            stepsDone = 0;
            stepsTotal = 0;
            progress.setProgress(0);
            ui.post(ticker);
        } else {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            ui.removeCallbacks(ticker);
            times.setText("");
        }
    }

    /** Called on the worker thread by the native side. */
    private void onProgress(String stage, int done, int total) {
        final long now = SystemClock.elapsedRealtime();
        final String line = stage + "  [" + thermal() + "]";
        ui.post(() -> {
            progress.setMax(total);
            progress.setProgress(done);
            if (!cancelRequested) {
                status.setText(stage);
            }
            if (stage.startsWith("Step ") && stage.contains(" took ")) {
                // total = steps + 1 (the decode); step 1 also uploads, so the
                // pace is taken from the steps after it.
                stepsDone = done;
                stepsTotal = total - 1;
                if (done == 1) {
                    firstStepDone = now;
                }
            }
            if (stage.startsWith("Step ") && stage.contains(" took ")
                    || stage.startsWith("Decoded") || stage.startsWith("Encoded")) {
                log.append(line + "\n");
            }
            if (running) {
                times.setText(timing());
            }
        });
    }

    /** Elapsed time, and once two steps are done, about how long is left. */
    private String timing() {
        long now = SystemClock.elapsedRealtime();
        String s = clock((now - runStart) / 1e3) + " elapsed";
        if (stepsDone >= 2 && stepsDone < stepsTotal) {
            double perStep = (now - firstStepDone) / 1e3 / (stepsDone - 1);
            // The decode takes about 0.6 of a step at every size.
            double left = perStep * (stepsTotal - stepsDone + 0.6);
            s += ", about " + clock(left) + " left";
        }
        return s;
    }

    private static String clock(double seconds) {
        long t = Math.round(seconds);
        return t < 60 ? t + " s" : String.format(Locale.US, "%d:%02d", t / 60, t % 60);
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

    // ---- Saved images ------------------------------------------------------

    /** What made a saved image; the gallery puts it back in the fields. */
    private static final class Made {
        final long seed;
        final int size;
        final String prompt;

        Made(long seed, int size, String prompt) {
            this.seed = seed;
            this.size = size;
            this.prompt = prompt;
        }
    }

    /** One line per saved image: uri, seed, size, prompt (tab-separated). */
    private File historyFile() {
        return new File(getFilesDir(), "history.tsv");
    }

    private Map<String, Made> readHistory() {
        Map<String, Made> made = new HashMap<>();
        try {
            for (String l : Files.readAllLines(historyFile().toPath(), StandardCharsets.UTF_8)) {
                String[] f = l.split("\t", 4);
                if (f.length == 4) {
                    made.put(f[0], new Made(Long.parseLong(f[1]), Integer.parseInt(f[2]), f[3]));
                }
            }
        } catch (IOException | NumberFormatException e) {
            // No history yet, or a damaged line: those images show without it.
        }
        return made;
    }

    /** Into Pictures/ZImage; the gallery's path for it, or null. */
    private String save(Bitmap bitmap, String text, long s, int n) {
        String name = "zimage-" + s + "-" + n + ".png";
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, GALLERY_DIR);
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
        String line = uri + "\t" + s + "\t" + n + "\t" + text.replaceAll("[\\t\\r\\n]+", " ") + "\n";
        try {
            Files.write(historyFile().toPath(), line.getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            // The image is saved; only the gallery's "use these settings" misses it.
        }
        return GALLERY_DIR + name;
    }

    /** The newest images in Pictures/ZImage, as thumbnails; tap to view, long-press to open elsewhere. */
    private void refreshGallery() {
        final int px = dp(88);
        thumbs.execute(() -> {
            ContentResolver resolver = getContentResolver();
            List<Uri> uris = new ArrayList<>();
            try (Cursor c = resolver.query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    new String[]{MediaStore.Images.Media._ID},
                    MediaStore.Images.Media.RELATIVE_PATH + "=?", new String[]{GALLERY_DIR},
                    MediaStore.Images.Media.DATE_ADDED + " DESC")) {
                while (c != null && c.moveToNext() && uris.size() < GALLERY_MAX) {
                    uris.add(ContentUris.withAppendedId(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, c.getLong(0)));
                }
            } catch (Exception e) {
                return;
            }
            final Map<String, Made> made = readHistory();
            List<Bitmap> bitmaps = new ArrayList<>();
            List<Uri> shown = new ArrayList<>();
            for (Uri u : uris) {
                try {
                    bitmaps.add(resolver.loadThumbnail(u, new Size(px, px), null));
                    shown.add(u);
                } catch (IOException e) {
                    // Deleted or unreadable: skipped.
                }
            }
            ui.post(() -> {
                gallery.removeAllViews();
                for (int i = 0; i < shown.size(); ++i) {
                    final Uri u = shown.get(i);
                    final Made m = made.get(u.toString());
                    ImageView v = new ImageView(this);
                    v.setImageBitmap(bitmaps.get(i));
                    v.setScaleType(ImageView.ScaleType.CENTER_CROP);
                    v.setOnClickListener(x -> showSaved(u, m));
                    v.setOnLongClickListener(x -> {
                        startActivity(new Intent(Intent.ACTION_VIEW).setDataAndType(u, "image/png")
                                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION));
                        return true;
                    });
                    LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(px, px);
                    lp.setMarginEnd(dp(6));
                    gallery.addView(v, lp);
                }
            });
        });
    }

    /** Shows a saved image and, while idle, puts its prompt, seed and size back. */
    private void showSaved(Uri u, Made m) {
        thumbs.execute(() -> {
            Bitmap b;
            try {
                b = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), u));
            } catch (IOException e) {
                return;
            }
            ui.post(() -> {
                image.setImageBitmap(b);
                if (m == null || running) {
                    return;
                }
                if (ZImage.hasEncoder()) {
                    prompt.setText(m.prompt);
                }
                seed.setText(Long.toString(m.seed));
                if (checkSize(m.size)) {
                    adbSize = 0;
                } else {
                    sizes.clearCheck();
                    adbSize = m.size;
                }
                status.setText(String.format(Locale.US, "Seed %d, %dx%d. Generate makes it again.",
                        m.seed, m.size, m.size));
            });
        });
    }

    // ---- Views -------------------------------------------------------------

    private void buildViews() {
        int pad = dp(16);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this);
        title.setText("Z-Image Turbo, on this phone's GPU");
        title.setTextSize(TypedValue.COMPLEX_UNIT_SP, 20);
        root.addView(title);

        prompt = new EditText(this);
        prompt.setHint("Describe the image");
        prompt.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        prompt.setMinLines(2);
        prompt.setText("A red fox sitting in a snowy birch forest at dawn");
        root.addView(prompt, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

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

        sizes = new RadioGroup(this);
        sizes.setOrientation(RadioGroup.HORIZONTAL);
        for (int i = 0; i < SIZES.length; ++i) {
            RadioButton b = new RadioButton(this);
            b.setId(i + 1);
            b.setText(SIZES[i] + "\u00d7" + SIZES[i]);
            sizes.addView(b, new RadioGroup.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        }
        sizes.check(1);
        sizes.setOnCheckedChangeListener((g, id) -> {
            if (id != -1) {
                adbSize = 0;
            }
        });
        root.addView(sizes, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        // Totals for a new prompt on a warm S25 Ultra (README.md).
        TextView sizeTimes = new TextView(this);
        sizeTimes.setText("About 1, 2.5 and 4.5 minutes on a warm Galaxy S25 Ultra");
        sizeTimes.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        root.addView(sizeTimes);

        generate = new Button(this);
        generate.setText("Generate");
        generate.setEnabled(false);
        generate.setOnClickListener(v -> onGenerateClicked());
        root.addView(generate);

        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        root.addView(progress);

        status = new TextView(this);
        status.setPadding(0, dp(8), 0, 0);
        root.addView(status);

        times = new TextView(this);
        times.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        times.setPadding(0, 0, 0, dp(8));
        root.addView(times);

        image = new ImageView(this);
        image.setAdjustViewBounds(true);
        image.setScaleType(ImageView.ScaleType.FIT_CENTER);
        root.addView(image, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView galleryLabel = new TextView(this);
        galleryLabel.setText("Made so far: tap to view and reuse, long-press to open");
        galleryLabel.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        galleryLabel.setPadding(0, dp(12), 0, dp(4));
        root.addView(galleryLabel);
        gallery = new LinearLayout(this);
        gallery.setOrientation(LinearLayout.HORIZONTAL);
        HorizontalScrollView strip = new HorizontalScrollView(this);
        strip.addView(gallery);
        root.addView(strip);

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
