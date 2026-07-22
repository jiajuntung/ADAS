package com.example.adas_fyp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

/**
 * Warning overlay + LDW debug view.
 *
 * TWO IMPORTANT FIXES HERE
 *
 * 1. COORDINATE MAPPING WAS WRONG.
 *    The old code mapped analysis-image coordinates to the view with
 *        scaleX = viewW / imageW ,  scaleY = viewH / imageH
 *    i.e. an independent stretch on each axis. But PreviewView renders with
 *    FILL_CENTER: it scales BOTH axes by the same factor and centre-crops the
 *    overflow. With a 4:3 analysis image in this 2.2:1 landscape viewport the
 *    vertical error is a factor of 0.6, so every green YOLO box has been drawn
 *    in the wrong place -- boxes appeared to sit on cars while the detector
 *    was actually looking somewhere else. Any visual judgement of the
 *    detectors made from earlier recordings was reading a lie.
 *
 * 2. THE LANE DETECTOR IS NOW VISIBLE.
 *    Three road tests were spent guessing why LDW never fired; the cause was
 *    that its analysis strip sat on the car's bonnet, which no status line
 *    could reveal. drawLaneDebug() paints the strip and the detected bands
 *    onto the preview, so the next test SHOWS the problem instead of hiding
 *    it. Note the strip may legitimately fall outside the visible preview
 *    (that is what FILL_CENTER cropping does) -- if the strip is clipped off
 *    the bottom of the screen, that itself is the diagnosis.
 */
class SafetyOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private val warningPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private val stripPaint = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 3f
    }

    private val bandPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    private val rawBandPaint = Paint().apply {
        color = Color.argb(140, 255, 255, 255)
        style = Paint.Style.STROKE
        strokeWidth = 2f
    }

    private val labelPaint = Paint().apply {
        color = Color.CYAN
        textSize = 26f
        isAntiAlias = true
    }

    private var fcwBox: Rect? = null
    private var fcwImageWidth: Int = 1
    private var fcwImageHeight: Int = 1

    private var roadOverlayVisible: Boolean = true
    private var warningMode: String = "NORMAL"

    // Lane debug state
    private var laneDebugEnabled = true
    private var stripTop = 0f
    private var stripBottom = 0f
    private var bandXs: FloatArray = FloatArray(0)
    private var rawBandXs: FloatArray = FloatArray(0)
    private var laneLabel: String = ""

    fun setFcwBox(box: Rect?, imageWidth: Int, imageHeight: Int) {
        fcwBox = box
        fcwImageWidth = imageWidth.coerceAtLeast(1)
        fcwImageHeight = imageHeight.coerceAtLeast(1)
        invalidate()
    }

    fun clearFcwBox() {
        fcwBox = null
        invalidate()
    }

    fun setRoadOverlayVisible(visible: Boolean) {
        roadOverlayVisible = visible
        invalidate()
    }

    fun setMode(mode: String) {
        warningMode = mode
        invalidate()
    }

    fun setLaneDebugEnabled(enabled: Boolean) {
        laneDebugEnabled = enabled
        invalidate()
    }

    fun setLaneDebug(
        stripTopF: Float,
        stripBottomF: Float,
        bands: FloatArray,
        rawBands: FloatArray,
        label: String
    ) {
        stripTop = stripTopF
        stripBottom = stripBottomF
        bandXs = bands
        rawBandXs = rawBands
        laneLabel = label
        invalidate()
    }

    // FIT_CENTER mapping: identical scale on both axes, whole image visible
    // with letterbox bars.
    //
    // MainActivity now sets PreviewView.scaleType = FIT_CENTER. Under the old
    // FILL_CENTER the preview cropped the top and bottom 20% of a 4:3 analysis
    // frame, so anything the detectors did down there was literally invisible
    // -- which is how the LDW strip sat on the bonnet for three road tests
    // without anyone being able to see it. FIT_CENTER makes the preview show
    // exactly what the detectors analyse: what you see is what YOLO/LDW see.
    private fun scaleFactor(): Float {
        val w = width.toFloat()
        val h = height.toFloat()
        return minOf(w / fcwImageWidth, h / fcwImageHeight)
    }

    private fun offsetX(s: Float): Float = (width - fcwImageWidth * s) / 2f

    private fun offsetY(s: Float): Float = (height - fcwImageHeight * s) / 2f

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        val s = scaleFactor()
        val ox = offsetX(s)
        val oy = offsetY(s)

        val currentBoxPaint = if (warningMode == "FCW") warningPaint else boxPaint

        fcwBox?.let { box ->
            val mappedBox = RectF(
                ox + box.left * s,
                oy + box.top * s,
                ox + box.right * s,
                oy + box.bottom * s
            )

            canvas.drawRect(mappedBox, currentBoxPaint)
        }

        if (laneDebugEnabled && stripBottom > stripTop) {
            val yTop = oy + stripTop * fcwImageHeight * s
            val yBot = oy + stripBottom * fcwImageHeight * s
            val xL = ox + 0.10f * fcwImageWidth * s
            val xR = ox + 0.90f * fcwImageWidth * s

            canvas.drawRect(xL, yTop, xR, yBot, stripPaint)

            for (bx in rawBandXs) {
                val x = ox + bx * fcwImageWidth * s
                canvas.drawLine(x, yTop, x, yBot, rawBandPaint)
            }

            for (bx in bandXs) {
                val x = ox + bx * fcwImageWidth * s
                canvas.drawLine(x, yTop, x, yBot, bandPaint)
            }

            // Always keep the label on screen, whatever the strip does.
            val labelY = (yTop - 8f).coerceIn(30f, height - 10f)
            canvas.drawText("LDW strip  $laneLabel", xL, labelY, labelPaint)
        }

        if (roadOverlayVisible && warningMode == "LDW") {
            canvas.drawLine(w * 0.25f, h, w * 0.45f, h * 0.55f, warningPaint)
            canvas.drawLine(w * 0.75f, h, w * 0.55f, h * 0.55f, warningPaint)
        }
    }
}