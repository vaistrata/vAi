package com.vaistrata.zimage;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.ContentValues;
import android.content.Intent;
import android.content.res.ColorStateList;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
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
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
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
 * One screen, top to bottom: the image (with the run's progress over it), the
 * prompt, the size and seed, Generate (Cancel while it runs), the images made
 * so far, and the run's details.
 * {@code adb shell am start -S -n com.vaistrata.zimage/.MainActivity --el seed 42 --es prompt "..."}
 * (and optionally {@code --ei size 768}) generates as soon as the weights are
 * loaded. Launched that way it also turns the screen on and shows over the lock
 * screen, since a stopped activity's process is frozen mid-run (Samsung's
 * FreecessController does it within a minute).
 */
public final class MainActivity extends Activity {
    private static final int[] SIZES = {512, 768, 1024};
    /** Rough totals for a new prompt on a warm S25 Ultra (README.md). */
    private static final String[] SIZE_TIMES = {"~1 min", "~2.5 min", "~4.5 min"};
    private static final int GALLERY_MAX = 40;
    private static final String GALLERY_DIR = Environment.DIRECTORY_PICTURES + "/ZImage/";

    // Palette: a dark UI, one accent.
    private static final int BG = 0xFF0F1014;
    private static final int SURFACE = 0xFF1A1C22;
    private static final int SURFACE_HI = 0xFF262932;
    private static final int OUTLINE = 0xFF33363F;
    private static final int TEXT = 0xFFECEDF0;
    private static final int TEXT_DIM = 0xFF9CA1AB;
    private static final int ACCENT = 0xFF8E9BFF;
    private static final int ON_ACCENT = 0xFF10131F;
    private static final int DANGER = 0xFFFF7A7A;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final ExecutorService thumbs = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());

    private EditText prompt;
    private EditText seed;
    private final TextView[] sizeButtons = new TextView[SIZES.length];
    private int sizeIndex = 0;        // into SIZES, or -1 for adbSize
    private int adbSize;              // a size from adb or the gallery that has no button
    private TextView generate;
    private ImageView image;
    private TextView placeholder;
    private LinearLayout overlay;
    private TextView overlayStage;
    private TextView overlayTime;
    private ProgressBar progress;
    private TextView status;
    private LinearLayout gallery;
    private TextView galleryEmpty;
    private TextView detailsToggle;
    private TextView log;
    private PowerManager power;
    private ScrollView scroll;

    // The running generation, on the UI thread.
    private boolean ready;
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
            overlayTime.setText(timing());
            ui.postDelayed(this, 1000);
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        power = getSystemService(PowerManager.class);
        getWindow().getDecorView().setBackgroundColor(BG);
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
            selectSize(getIntent().getIntExtra("size", 512));
        }
        setStatus("Loading the model...", false);
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
                setStatus(result + "\n\nData directory: " + ZImage.dataDir(), true);
                return;
            }
            if (!ZImage.hasEncoder()) {
                // The fixed demo features: their prompt is shown, not edited.
                String p = ZImage.prompt();
                prompt.setText(p.isEmpty() ? "(no text encoder: the demo caption features)" : p);
                prompt.setEnabled(false);
            }
            ready = true;
            setStatus("", false);
            updateGenerate();
            if (autoRun) {
                startGenerate();
            }
        });
    }

    // ---- Size ----------------------------------------------------------------

    /** Selects the button for this size, or none (the size is kept aside). */
    private void selectSize(int s) {
        sizeIndex = -1;
        adbSize = s;
        for (int i = 0; i < SIZES.length; ++i) {
            if (SIZES[i] == s) {
                sizeIndex = i;
                adbSize = 0;
            }
        }
        for (int i = 0; i < SIZES.length; ++i) {
            styleSegment(sizeButtons[i], i == sizeIndex);
        }
    }

    private int selectedSize() {
        return sizeIndex >= 0 ? SIZES[sizeIndex] : (adbSize > 0 ? adbSize : 512);
    }

    // ---- Generation ----------------------------------------------------------

    private void onGenerateClicked() {
        if (!running) {
            startGenerate();
            return;
        }
        cancelRequested = true;
        ZImage.cancel();
        generate.setEnabled(false);
        generate.setText("Stopping after this step...");
        overlayStage.setText("Stopping...");
    }

    private void startGenerate() {
        final long s;
        try {
            s = Long.parseLong(seed.getText().toString().trim());
        } catch (NumberFormatException e) {
            setStatus("The seed must be a number.", true);
            return;
        }
        final String text = Normalizer.normalize(prompt.getText().toString().trim(), Normalizer.Form.NFC);
        if (ZImage.hasEncoder() && text.isEmpty()) {
            setStatus("Describe the image first.", true);
            return;
        }
        hideKeyboard();
        // Up to the image card, where the run's progress shows.
        scroll.post(() -> scroll.smoothScrollTo(0, 0));
        final int n = selectedSize();
        setRunning(true);
        setStatus("", false);
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
                    setStatus(cancelled ? "Cancelled." : "Generation failed (see logcat -s zimage).", !cancelled);
                    return;
                }
                showImage(result);
                setStatus(String.format(Locale.US, "%d×%d · seed %d · %s%s", n, n, s, clock(seconds),
                        where != null ? " · saved to Pictures/ZImage" : ""), false);
                refreshGallery();
            });
        });
    }

    private void setRunning(boolean on) {
        running = on;
        cancelRequested = false;
        prompt.setEnabled(!on && ZImage.hasEncoder());
        seed.setEnabled(!on);
        for (TextView b : sizeButtons) {
            b.setEnabled(!on);
            b.setAlpha(on ? 0.5f : 1f);
        }
        overlay.setVisibility(on ? View.VISIBLE : View.GONE);
        if (on) {
            placeholder.setVisibility(View.GONE);
        } else if (image.getDrawable() == null) {
            placeholder.setVisibility(View.VISIBLE);
        }
        updateGenerate();
        if (on) {
            getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            runStart = SystemClock.elapsedRealtime();
            firstStepDone = 0;
            stepsDone = 0;
            stepsTotal = 0;
            progress.setProgress(0);
            overlayStage.setText("Starting...");
            overlayTime.setText("");
            ui.post(ticker);
        } else {
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            ui.removeCallbacks(ticker);
        }
    }

    private void updateGenerate() {
        generate.setEnabled(ready);
        generate.setText(running ? "Cancel" : "Generate");
        GradientDrawable bg = rounded(running ? Color.TRANSPARENT : ACCENT, dp(28));
        if (running) {
            bg.setStroke(dp(1), DANGER);
        }
        generate.setBackground(bg);
        generate.setTextColor(running ? DANGER : ON_ACCENT);
        generate.setAlpha(ready ? 1f : 0.4f);
    }

    /** Called on the worker thread by the native side. */
    private void onProgress(String stage, int done, int total) {
        final long now = SystemClock.elapsedRealtime();
        final String line = stage + "  \u00b7 " + thermal();
        ui.post(() -> {
            progress.setMax(total);
            progress.setProgress(done);
            if (stage.startsWith("Step ") && stage.contains(" took ")) {
                // total = steps + 1 (the decode); step 1 also uploads, so the
                // pace is taken from the steps after it.
                stepsDone = done;
                stepsTotal = total - 1;
                if (done == 1) {
                    firstStepDone = now;
                }
            }
            if (running && !cancelRequested) {
                overlayStage.setText(friendlyStage(stage, done, total));
                overlayTime.setText(timing());
            } else if (!running) {
                setStatus(stage, false);   // loading
            }
            if (stage.startsWith("Step ") && stage.contains(" took ")
                    || stage.startsWith("Decoded") || stage.startsWith("Encoded")) {
                log.append(line + "\n");
            }
        });
    }

    /** The native stage, said for the overlay. */
    private String friendlyStage(String stage, int done, int total) {
        if (stage.startsWith("Step ") && stage.contains(" took ")) {
            return done < total - 1 ? "Step " + (done + 1) + " of " + (total - 1) : "Decoding";
        }
        if (stage.startsWith("Step 1")) {
            return "Step 1 of " + (total - 1);
        }
        if (stage.startsWith("Decoding")) {
            return "Decoding";
        }
        if (stage.startsWith("Decoded")) {
            return "Done";
        }
        if (stage.startsWith("Loading the text encoder") || stage.startsWith("Encoding")) {
            return "Reading the prompt";
        }
        if (stage.startsWith("Encoded")) {
            return "Prompt read";
        }
        if (stage.startsWith("Loading the DiT")) {
            return "Loading the model";
        }
        return stage;
    }

    /** Elapsed time, and once two steps are done, about how long is left. */
    private String timing() {
        long now = SystemClock.elapsedRealtime();
        String s = clock((now - runStart) / 1e3);
        if (stepsDone >= 2 && stepsDone < stepsTotal) {
            double perStep = (now - firstStepDone) / 1e3 / (stepsDone - 1);
            // The decode takes about 0.6 of a step at every size.
            double left = perStep * (stepsTotal - stepsDone + 0.6);
            s += "  ·  about " + clock(left) + " left";
        }
        return s;
    }

    private static String clock(double seconds) {
        long t = Math.round(seconds);
        return t < 60 ? t + " s" : String.format(Locale.US, "%d:%02d", t / 60, t % 60);
    }

    /** The system's thermal status and headroom (1.0 = throttling starts), for the details. */
    private String thermal() {
        String[] names = {"none", "light", "moderate", "severe", "critical", "emergency", "shutdown"};
        int s = power.getCurrentThermalStatus();
        float headroom = power.getThermalHeadroom(0);
        String name = s >= 0 && s < names.length ? names[s] : Integer.toString(s);
        return Float.isNaN(headroom) ? name : String.format(Locale.US, "%s %.2f", name, headroom);
    }

    private void setStatus(String text, boolean error) {
        status.setText(text);
        status.setTextColor(error ? DANGER : TEXT_DIM);
        status.setVisibility(text.isEmpty() ? View.GONE : View.VISIBLE);
    }

    private void showImage(Bitmap b) {
        image.setImageBitmap(b);
        placeholder.setVisibility(View.GONE);
    }

    private void hideKeyboard() {
        InputMethodManager imm = getSystemService(InputMethodManager.class);
        if (imm != null) {
            imm.hideSoftInputFromWindow(prompt.getWindowToken(), 0);
        }
        prompt.clearFocus();
        seed.clearFocus();
    }

    // ---- Saved images --------------------------------------------------------

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
        final int px = dp(84);
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
                galleryEmpty.setVisibility(shown.isEmpty() ? View.VISIBLE : View.GONE);
                for (int i = 0; i < shown.size(); ++i) {
                    final Uri u = shown.get(i);
                    final Made m = made.get(u.toString());
                    ImageView v = new ImageView(this);
                    v.setImageBitmap(bitmaps.get(i));
                    v.setScaleType(ImageView.ScaleType.CENTER_CROP);
                    v.setBackground(rounded(SURFACE, dp(14)));
                    v.setClipToOutline(true);
                    v.setOnClickListener(x -> showSaved(u, m));
                    v.setOnLongClickListener(x -> {
                        startActivity(new Intent(Intent.ACTION_VIEW).setDataAndType(u, "image/png")
                                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION));
                        return true;
                    });
                    LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(px, px);
                    lp.setMarginEnd(dp(8));
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
                if (running) {
                    return;   // the overlay is over the image; keep the run in view
                }
                showImage(b);
                if (m == null) {
                    setStatus("Long-press a thumbnail to open it in the gallery.", false);
                    return;
                }
                if (ZImage.hasEncoder()) {
                    prompt.setText(m.prompt);
                }
                seed.setText(Long.toString(m.seed));
                selectSize(m.size);
                setStatus(String.format(Locale.US, "%d×%d · seed %d · Generate makes it again",
                        m.size, m.size, m.seed), false);
            });
        });
    }

    // ---- Views ---------------------------------------------------------------

    private void buildViews() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(12), dp(20), dp(16));

        // Header
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        header.setPadding(dp(4), 0, dp(4), dp(16));
        TextView title = text("Z-Image", 26, TEXT, true);
        header.addView(title);
        header.addView(text("Text to image, made on this phone's GPU", 13, TEXT_DIM, false));
        root.addView(header);

        // Image card, square, with the run's progress over it
        FrameLayout card = new FrameLayout(this) {
            @Override
            protected void onMeasure(int w, int h) {
                super.onMeasure(w, w);   // square
            }
        };
        card.setBackground(rounded(SURFACE, dp(24)));
        card.setClipToOutline(true);
        image = new ImageView(this);
        image.setScaleType(ImageView.ScaleType.CENTER_CROP);
        card.addView(image, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        placeholder = text("Your image will appear here", 15, TEXT_DIM, false);
        placeholder.setGravity(Gravity.CENTER);
        card.addView(placeholder, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        overlay = new LinearLayout(this);
        overlay.setOrientation(LinearLayout.VERTICAL);
        overlay.setGravity(Gravity.CENTER);
        overlay.setBackgroundColor(0xB30F1014);
        overlay.setPadding(dp(28), 0, dp(28), 0);
        overlayStage = text("", 22, TEXT, true);
        overlayStage.setGravity(Gravity.CENTER);
        overlay.addView(overlayStage);
        overlayTime = text("", 14, TEXT_DIM, false);
        overlayTime.setGravity(Gravity.CENTER);
        overlayTime.setPadding(0, dp(6), 0, dp(18));
        overlay.addView(overlayTime);
        progress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        progress.setProgressTintList(ColorStateList.valueOf(ACCENT));
        progress.setProgressBackgroundTintList(ColorStateList.valueOf(SURFACE_HI));
        overlay.addView(progress, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(6)));
        overlay.setVisibility(View.GONE);
        card.addView(overlay, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        root.addView(card, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0));
        ((LinearLayout.LayoutParams) card.getLayoutParams()).height = ViewGroup.LayoutParams.WRAP_CONTENT;

        status = text("", 13, TEXT_DIM, false);
        status.setPadding(dp(4), dp(10), dp(4), 0);
        root.addView(status);

        // Prompt
        root.addView(label("Prompt"));
        prompt = new EditText(this);
        prompt.setHint("Describe the image");
        prompt.setHintTextColor(TEXT_DIM);
        prompt.setTextColor(TEXT);
        prompt.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        prompt.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        prompt.setMinLines(3);
        prompt.setGravity(Gravity.TOP | Gravity.START);
        prompt.setBackground(rounded(SURFACE, dp(16)));
        prompt.setPadding(dp(16), dp(14), dp(16), dp(14));
        prompt.setText("A red fox sitting in a snowy birch forest at dawn");
        root.addView(prompt, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        // Size
        root.addView(label("Size"));
        LinearLayout segments = new LinearLayout(this);
        segments.setOrientation(LinearLayout.HORIZONTAL);
        segments.setBackground(rounded(SURFACE, dp(16)));
        segments.setPadding(dp(4), dp(4), dp(4), dp(4));
        for (int i = 0; i < SIZES.length; ++i) {
            final int index = i;
            TextView b = text(SIZES[i] + "²\n" + SIZE_TIMES[i], 14, TEXT, true);
            b.setGravity(Gravity.CENTER);
            b.setPadding(0, dp(10), 0, dp(10));
            b.setOnClickListener(v -> selectSize(SIZES[index]));
            sizeButtons[i] = b;
            segments.addView(b, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        }
        root.addView(segments, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        selectSize(512);

        // Seed
        root.addView(label("Seed"));
        LinearLayout seedRow = new LinearLayout(this);
        seedRow.setOrientation(LinearLayout.HORIZONTAL);
        seedRow.setGravity(Gravity.CENTER_VERTICAL);
        seed = new EditText(this);
        seed.setInputType(InputType.TYPE_CLASS_NUMBER);
        seed.setTextColor(TEXT);
        seed.setTextSize(TypedValue.COMPLEX_UNIT_SP, 16);
        seed.setBackground(rounded(SURFACE, dp(16)));
        seed.setPadding(dp(16), dp(12), dp(16), dp(12));
        seed.setText(Long.toString(new Random().nextInt(1_000_000)));
        seedRow.addView(seed, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        TextView dice = text("🎲  New seed", 14, TEXT, true);
        dice.setGravity(Gravity.CENTER);
        dice.setBackground(rounded(SURFACE_HI, dp(16)));
        dice.setPadding(dp(16), dp(12), dp(16), dp(12));
        dice.setOnClickListener(v -> {
            if (!running) {
                seed.setText(Long.toString(new Random().nextInt(1_000_000)));
            }
        });
        LinearLayout.LayoutParams dlp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        dlp.setMarginStart(dp(10));
        seedRow.addView(dice, dlp);
        root.addView(seedRow);

        // Recent
        root.addView(label("Recent"));
        gallery = new LinearLayout(this);
        gallery.setOrientation(LinearLayout.HORIZONTAL);
        HorizontalScrollView strip = new HorizontalScrollView(this);
        strip.setHorizontalScrollBarEnabled(false);
        strip.addView(gallery);
        root.addView(strip);
        galleryEmpty = text("Images you make show up here.", 13, TEXT_DIM, false);
        galleryEmpty.setPadding(dp(4), 0, dp(4), 0);
        galleryEmpty.setVisibility(View.GONE);
        root.addView(galleryEmpty);
        TextView hint = text("Tap to view and reuse its settings · long-press to open", 12, TEXT_DIM, false);
        hint.setPadding(dp(4), dp(8), dp(4), 0);
        root.addView(hint);

        // Details (the per-step log), folded
        detailsToggle = text("Details  ▸", 13, TEXT_DIM, true);
        detailsToggle.setPadding(dp(4), dp(24), dp(4), dp(8));
        log = text("", 12, TEXT_DIM, false);
        log.setTypeface(Typeface.MONOSPACE);
        log.setPadding(dp(4), 0, dp(4), 0);
        log.setVisibility(View.GONE);
        detailsToggle.setOnClickListener(v -> {
            boolean show = log.getVisibility() != View.VISIBLE;
            log.setVisibility(show ? View.VISIBLE : View.GONE);
            detailsToggle.setText(show ? "Details  ▾" : "Details  ▸");
        });
        root.addView(detailsToggle);
        root.addView(log);

        scroll = new ScrollView(this);
        scroll.setBackgroundColor(BG);
        scroll.addView(root);

        // Generate stays in reach at the bottom, over the scrolling content.
        generate = text("Generate", 17, ON_ACCENT, true);
        generate.setGravity(Gravity.CENTER);
        generate.setOnClickListener(v -> onGenerateClicked());
        LinearLayout bar = new LinearLayout(this);
        bar.setBackgroundColor(BG);
        bar.setPadding(dp(20), dp(10), dp(20), dp(14));
        bar.addView(generate, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(56)));
        updateGenerate();

        LinearLayout screen = new LinearLayout(this);
        screen.setOrientation(LinearLayout.VERTICAL);
        screen.setFitsSystemWindows(true);
        screen.setBackgroundColor(BG);
        screen.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));
        screen.addView(bar, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        setContentView(screen);
    }

    private void styleSegment(TextView b, boolean selected) {
        b.setBackground(selected ? rounded(ACCENT, dp(12)) : null);
        b.setTextColor(selected ? ON_ACCENT : TEXT);
    }

    private TextView label(String s) {
        TextView t = text(s, 13, TEXT_DIM, true);
        t.setPadding(dp(4), dp(20), dp(4), dp(8));
        return t;
    }

    private TextView text(String s, int sp, int color, boolean medium) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(TypedValue.COMPLEX_UNIT_SP, sp);
        t.setTextColor(color);
        if (medium) {
            t.setTypeface(Typeface.create("sans-serif-medium", Typeface.NORMAL));
        }
        return t;
    }

    private GradientDrawable rounded(int color, int radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(radius);
        return d;
    }

    private int dp(int v) {
        return Math.round(v * getResources().getDisplayMetrics().density);
    }
}
