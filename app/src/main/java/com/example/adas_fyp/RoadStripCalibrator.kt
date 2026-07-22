package com.example.adas_fyp

import androidx.camera.core.ImageProxy

/**
 * Finds where the road actually is in the analysis frame, automatically.
 *
 * WHY THIS EXISTS -- THE 16 JULY BUG
 * The v3 lane detector analysed a fixed strip at frame y 0.78..0.94, chosen
 * by hand. On the 16 July test it reported "Lines:1  L:false R:false
 * Cross:false" for the entire 293 s drive and never once produced a warning,
 * including through real lane crossings.
 *
 * The reason: PreviewView uses FILL_CENTER, so a 4:3 analysis image shown in
 * a 2.2:1 viewport has its top and bottom 20% cropped OFF SCREEN. Everything
 * below frame y 0.80 is invisible in the preview -- and that bottom region is
 * where the bonnet/dashboard sits. The strip was parked on the car's own
 * bodywork: a smooth, static surface with no lane markings in it, which is
 * exactly what "Lines:1, nothing ever detected" looks like.
 *
 * Hand-picking this number failed three times, so it is no longer hand-picked.
 *
 * HOW IT CALIBRATES
 * While the car is moving, road texture streams through the image and every
 * road row changes frame to frame. The bonnet is bolted to the car, so its
 * rows barely change at all. Measuring per-row temporal variation therefore
 * separates them cleanly:
 *
 *   row variation HIGH  -> road (or any moving scene)
 *   row variation ~0    -> bonnet / dashboard / static bodywork
 *
 * We look for a block of near-static rows at the BOTTOM of the frame and put
 * the strip just above it. If the phone is mounted high enough that no bonnet
 * is visible, no static block is found and the strip simply sits at the
 * bottom of the frame. Either way the strip lands on road.
 *
 * Calibration only runs while moving (a static block is meaningless when the
 * whole scene is still) and settles within a few seconds of driving.
 */
class RoadStripCalibrator {

    private val rows = 48                  // vertical resolution of the scan
    private val cols = 32

    // Per-row EMA of |frame-to-frame change|, in luma units.
    private val rowActivity = FloatArray(rows)
    private var prevRowLuma: FloatArray? = null
    private val activityEma = 0.85f

    private var samples = 0
    private val samplesToConverge = 40     // ~2-4 s of driving

    // A row is "static" when its long-run activity is below this. Road rows
    // sit far above it, so the boundary is not sensitive to the exact value.
    private val staticActivityThreshold = 1.2f

    // The bonnet must occupy a plausible slice of the bottom of the frame;
    // beyond this we assume something else (e.g. a stopped scene) and ignore it.
    private val maxBonnetFraction = 0.35f

    private var bonnetTopFraction = 1.0f   // 1.0 = no bonnet found

    /** Strip bounds as fractions of frame height, {top, bottom}. */
    private var stripTopF = 0.62f
    private var stripBottomF = 0.74f

    // Strip geometry relative to whatever the bonnet edge turns out to be.
    private val stripHeightF = 0.12f
    private val stripGapAboveBonnetF = 0.02f

    // HARD CEILING on how low the strip may sit.
    //
    // The 16 Jul evening run found no bonnet (Bonnet:0.98 / none), so the
    // "sit just above the bonnet" rule pushed the strip to 0.84-0.96 -- the
    // very bottom of the frame. Two problems live down there:
    //   - it is outside the visible preview (FILL_CENTER crops below ~0.80),
    //     so nobody can see or check it;
    //   - the road 1-2 m ahead sweeps past so fast that, at evening exposure
    //     times, markings are motion-blurred into the asphalt.
    // Road ~4-8 m ahead is sharp, visible, and equally valid: while
    // straddling a line, that line projects near the image centre at every
    // height, not just at the bottom.
    //
    // So the bonnet rule may only ever move the strip UP, never below this.
    private val maxStripBottomF = 0.74f

    var converged = false
        private set

    fun stripTop(): Float = stripTopF
    fun stripBottom(): Float = stripBottomF
    fun bonnetTop(): Float = bonnetTopFraction

    fun statusText(): String {
        val b = if (bonnetTopFraction >= 0.999f) "none" else String.format("%.2f", bonnetTopFraction)
        val state = if (converged) "ok" else "calibrating $samples/$samplesToConverge"
        return "Strip:${String.format("%.2f", stripTopF)}-${String.format("%.2f", stripBottomF)} " +
                "Bonnet:$b ($state)"
    }

    /**
     * Feed every analysed back-camera frame. `moving` must be true only when
     * the vehicle is actually driving. Reads the luma plane directly and does
     * not consume the ImageProxy.
     */
    fun update(imageProxy: ImageProxy, moving: Boolean) {
        val width = imageProxy.width
        val height = imageProxy.height
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val limit = buffer.limit()

        val rowLuma = FloatArray(rows)

        for (r in 0 until rows) {
            val y = ((r + 0.5f) / rows * height).toInt().coerceIn(0, height - 1)
            var sum = 0
            var n = 0

            for (c in 0 until cols) {
                // Sample the middle 70% horizontally: the frame edges see
                // kerbs, pillars and passing traffic that are not the road
                // ahead and would blur the bonnet boundary.
                val xRel = 0.15f + 0.70f * (c + 0.5f) / cols
                val x = (xRel * width).toInt().coerceIn(0, width - 1)
                val p = y * rowStride + x * pixelStride

                if (p in 0 until limit) {
                    sum += buffer.get(p).toInt() and 0xFF
                    n++
                }
            }

            rowLuma[r] = if (n > 0) sum.toFloat() / n else 0f
        }

        val prev = prevRowLuma
        prevRowLuma = rowLuma

        if (prev == null || !moving) return

        for (r in 0 until rows) {
            val d = kotlin.math.abs(rowLuma[r] - prev[r])
            rowActivity[r] = activityEma * rowActivity[r] + (1f - activityEma) * d
        }

        samples++

        if (samples >= samplesToConverge) {
            recompute()
            converged = true
        }
    }

    private fun recompute() {
        // Walk up from the bottom while rows stay static; the first row that
        // moves is the top of the bonnet.
        var r = rows - 1
        var staticCount = 0

        while (r >= 0 && rowActivity[r] < staticActivityThreshold) {
            staticCount++
            r--
        }

        val bonnetFraction = staticCount.toFloat() / rows

        bonnetTopFraction = if (staticCount == 0 || bonnetFraction > maxBonnetFraction) {
            1.0f     // no believable bonnet: use the full frame
        } else {
            (r + 1).toFloat() / rows
        }

        // Take whichever is higher: just above the bonnet, or the ceiling.
        val bottom = minOf(
            bonnetTopFraction - stripGapAboveBonnetF,
            maxStripBottomF
        ).coerceIn(0.30f, maxStripBottomF)

        val top = (bottom - stripHeightF).coerceIn(0.20f, bottom - 0.04f)

        stripTopF = top
        stripBottomF = bottom
    }

    fun reset() {
        prevRowLuma = null
        samples = 0
        converged = false
        bonnetTopFraction = 1.0f
        stripTopF = 0.62f
        stripBottomF = 0.74f
        java.util.Arrays.fill(rowActivity, 0f)
    }
}