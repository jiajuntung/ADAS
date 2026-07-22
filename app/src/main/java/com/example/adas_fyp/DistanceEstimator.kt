package com.example.adas_fyp

import android.content.Context
import android.graphics.Rect
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager

/**
 * Monocular distance to a detected object, from real camera intrinsics.
 *
 * WHY THIS REPLACES areaRatio
 * Until now FCW fired on how BIG the YOLO box looked (areaRatio >= 0.040 in
 * YoloDetector.collisionWarning). Apparent size is not distance and it is
 * definitely not danger:
 *   - it depends on the object class (a bus at 30 m looks like a car at 12 m),
 *   - it depends on the phone's field of view,
 *   - and, most importantly, the SAME gap is safe at 10 km/h and an emergency
 *     at 70 km/h. Size alone cannot know that.
 * The 16 July test fired BRAKE at Area:0.056 and Area:0.072 -- distant cars --
 * because a fixed size threshold has no notion of either metres or speed.
 *
 * THE PINHOLE RELATION
 *     distance = (real_width_m * focal_px) / box_width_px
 * focal_px comes from the actual camera hardware:
 *     focal_px = focal_mm / sensor_width_mm * image_width_px
 * so there is nothing to hand-tune per phone.
 *
 * ACCURACY / LIMITS (deliberately stated, since FCW acts on this)
 *   - Width is used, not height: a vehicle's width is far more consistent
 *     than its height, and the bottom edge of a box is often clipped.
 *   - It assumes a rear view. A vehicle seen at an angle looks wider and so
 *     reads slightly closer -- an error in the safe direction.
 *   - Expect roughly +/-15-20% at 5-30 m. That is ample for headway and TTC
 *     decisions, which is why production ADAS is happy with far coarser
 *     inputs than people assume.
 */
class DistanceEstimator(context: Context) {

    // Typical real-world widths in metres, by COCO class id.
    private val realWidths = mapOf(
        0 to 0.55f,   // person
        1 to 0.65f,   // bicycle
        2 to 1.80f,   // car
        3 to 0.80f,   // motorcycle
        5 to 2.55f,   // bus
        7 to 2.45f    // truck
    )

    private val defaultWidth = 1.80f

    // Fallback if the camera refuses to report intrinsics: a typical phone
    // main camera has ~65 deg horizontal FOV -> f_px = (W/2)/tan(32.5 deg).
    private val fallbackHFovDeg = 65.0

    private var focalPx = -1f
    private var focalRatio = -1f   // focal_px / image_width_px  (resolution independent)

    var intrinsicsSource = "none"
        private set

    init {
        readIntrinsics(context)
    }

    private fun readIntrinsics(context: Context) {
        try {
            val cm = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

            for (id in cm.cameraIdList) {
                val ch = cm.getCameraCharacteristics(id)

                if (ch.get(CameraCharacteristics.LENS_FACING) !=
                    CameraCharacteristics.LENS_FACING_BACK
                ) continue

                val focals = ch.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                val physical = ch.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)

                if (focals != null && focals.isNotEmpty() && physical != null &&
                    physical.width > 0f
                ) {
                    // focal_px / image_width_px is constant for a given lens,
                    // so it survives any analysis resolution we choose later.
                    focalRatio = focals[0] / physical.width
                    intrinsicsSource = "camera2"
                    return
                }
            }
        } catch (e: Exception) {
            // fall through to the FOV fallback
        }

        val halfFov = Math.toRadians(fallbackHFovDeg / 2.0)
        focalRatio = (0.5 / Math.tan(halfFov)).toFloat()
        intrinsicsSource = "fallback"
    }

    /** Must be called once the analysis image width is known. */
    fun setImageWidth(imageWidthPx: Int) {
        if (imageWidthPx > 0 && focalRatio > 0f) {
            focalPx = focalRatio * imageWidthPx
        }
    }

    fun focalPx(): Float = focalPx

    /**
     * Distance in metres to a detected object, or -1 if it cannot be
     * estimated (unknown intrinsics, degenerate box, or a box clipped by the
     * frame edge -- a clipped box has a meaningless width).
     */
    fun distanceMeters(
        box: Rect,
        classId: Int?,
        imageWidth: Int
    ): Float {
        if (focalPx <= 0f || imageWidth <= 0) return -1f

        val wPx = box.width().toFloat()
        if (wPx < 8f) return -1f

        // A box touching the left/right edge is cut off: its width understates
        // the object and would read as "far away" exactly when it is closest.
        val margin = 2
        if (box.left <= margin || box.right >= imageWidth - margin) return -1f

        val realW = realWidths[classId] ?: defaultWidth

        return realW * focalPx / wPx
    }

    /**
     * How many metres left (-) or right (+) of the camera axis the object is.
     *
     * THIS IS THE EGO-LANE TEST. Image x-position is an ANGLE, not a lane:
     * beyond ~25 m a vehicle in the next lane (3.4 m to the side) projects to
     * image x ~0.55-0.59 and falls inside any "centre of frame" window, which
     * is why buses and cars beside us were triggering BRAKE. Multiplying that
     * angle by the distance recovers the real offset, which is ~3.4 m at EVERY
     * range and is cleanly outside a 1.7 m half-lane.
     *
     *     X = (box_centre_px - principal_px) * distance / focal_px
     *
     * Returns Float.NaN when distance or intrinsics are unavailable.
     */
    fun lateralOffsetMeters(
        box: Rect,
        distanceM: Float,
        imageWidth: Int
    ): Float {
        if (focalPx <= 0f || distanceM <= 0f || imageWidth <= 0) return Float.NaN

        val centreX = (box.left + box.right) / 2f
        val principalX = imageWidth / 2f

        return (centreX - principalX) * distanceM / focalPx
    }

    /**
     * Headway time in seconds: how long until we reach the spot the lead
     * vehicle occupies right now. This is the quantity behind the familiar
     * "two-second rule", and it is what makes the same gap safe at low speed
     * and dangerous at high speed.
     * Returns -1 when speed is unknown or we are barely moving.
     */
    fun headwaySeconds(distanceM: Float, speedMps: Float): Float {
        if (distanceM <= 0f || speedMps <= 1.0f) return -1f
        return distanceM / speedMps
    }
}