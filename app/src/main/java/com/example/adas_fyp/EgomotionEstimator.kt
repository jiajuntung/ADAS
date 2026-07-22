package com.example.adas_fyp

import androidx.camera.core.ImageProxy

/**
 * Pure-vision ego-motion estimator (no GPS / no IMU).
 *
 * Samples a sparse grid of luma points over the road area of the back-camera
 * frame and compares them with the same points from the previous frame.
 * When the vehicle moves, the road texture flows and the per-point difference
 * is large. When the vehicle is truly stationary the road is static and the
 * difference stays near sensor noise, even with engine vibration shaking the
 * phone.
 *
 * This is the signal the accelerometer physically cannot provide:
 * linear acceleration is ~0 m/s^2 both when parked AND when cruising at a
 * constant speed, so accelerometer-only logic confuses "smooth driving" with
 * "stopped". That single confusion caused missed FCW while approaching a lead
 * car, missed LDW while cruising over a lane line, and late/false LVSA.
 *
 * The MEDIAN of the per-point differences is used instead of the mean, so a
 * single moving object covering part of the ROI (e.g. the lead car pulling
 * away while we are stopped, or a pedestrian) cannot flip the result --
 * more than half of the sampled road must be changing.
 */
class EgoMotionEstimator {

    private val gridCols = 24
    private val gridRows = 10

    // ROI in frame-relative coordinates: road surface below the horizon,
    // above the very bottom of the frame (own bonnet / dashboard edge).
    private val roiLeft = 0.15f
    private val roiRight = 0.85f
    private val roiTop = 0.55f
    private val roiBottom = 0.92f

    // Below this mean luma the road is too dark for reliable differencing
    // (night without street lighting). Callers should fail safe: never
    // declare "stationary" from vision in low light.
    private val lowLightLumaThreshold = 22f

    // Differencing is done against the frame from TWO updates ago: the wider
    // time baseline roughly doubles the motion signal while sensor noise
    // stays flat, giving a clearer gap between "creeping" and "stopped".
    private var prevSamples: IntArray? = null
    private var prevPrevSamples: IntArray? = null
    private var lastMeanLuma = 0f

    fun isLowLight(): Boolean =
        lastMeanLuma > 0.1f && lastMeanLuma < lowLightLumaThreshold

    fun lastMeanLuma(): Float = lastMeanLuma

    /**
     * Returns the median absolute luma difference (0..255) between this frame
     * and the previous one over the road grid, or -1 when no previous frame
     * is available yet.
     *
     * Call this on every analyzed back frame BEFORE handing the frame to the
     * detectors. It only reads the Y plane with absolute indexing and does
     * NOT close or consume the ImageProxy.
     */
    fun update(imageProxy: ImageProxy): Float {
        val width = imageProxy.width
        val height = imageProxy.height
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val limit = buffer.limit()

        val count = gridCols * gridRows
        val samples = IntArray(count)
        var lumaSum = 0L
        var idx = 0

        for (r in 0 until gridRows) {
            val yRel = roiTop + (roiBottom - roiTop) * (r + 0.5f) / gridRows
            val y = (yRel * height).toInt().coerceIn(0, height - 1)

            for (c in 0 until gridCols) {
                val xRel = roiLeft + (roiRight - roiLeft) * (c + 0.5f) / gridCols
                val x = (xRel * width).toInt().coerceIn(1, width - 2)

                val base = y * rowStride + x * pixelStride

                // Average 3 horizontal neighbours to reduce per-pixel sensor noise.
                var v = 0
                var n = 0
                for (dx in -1..1) {
                    val p = base + dx * pixelStride
                    if (p in 0 until limit) {
                        v += buffer.get(p).toInt() and 0xFF
                        n++
                    }
                }

                val luma = if (n > 0) v / n else 0
                samples[idx++] = luma
                lumaSum += luma
            }
        }

        lastMeanLuma = lumaSum.toFloat() / count

        val base = prevPrevSamples
        prevPrevSamples = prevSamples
        prevSamples = samples

        if (base == null || base.size != count) {
            return -1f
        }

        val diffs = IntArray(count)
        for (i in 0 until count) {
            val d = samples[i] - base[i]
            diffs[i] = if (d >= 0) d else -d
        }
        diffs.sort()

        return diffs[count / 2].toFloat()
    }

    fun reset() {
        prevSamples = null
        prevPrevSamples = null
        lastMeanLuma = 0f
    }
}