package com.example.adas_fyp

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import android.graphics.Rect
import android.graphics.RectF

class SafetyOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }

    private var fcwBox: Rect? = null
    private var fcwImageWidth: Int = 1
    private var fcwImageHeight: Int = 1

    fun setFcwBox(box: Rect?, imageWidth: Int, imageHeight: Int) {
        fcwBox = box
        fcwImageWidth = imageWidth
        fcwImageHeight = imageHeight
        invalidate()
    }

    fun clearFcwBox() {
        fcwBox = null
        invalidate()
    }

    private var roadOverlayVisible: Boolean = true
    fun setRoadOverlayVisible(visible: Boolean) {
        roadOverlayVisible = visible
        invalidate()
    }
    private val lanePaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private val warningPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 8f
    }

    private var warningMode: String = "NORMAL"

    fun setMode(mode: String) {
        warningMode = mode
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()

        val currentBoxPaint = if (warningMode == "FCW") warningPaint else boxPaint

        fcwBox?.let { box ->
            val scaleX = w / fcwImageWidth.toFloat()
            val scaleY = h / fcwImageHeight.toFloat()

            val mappedBox = RectF(
                box.left * scaleX,
                box.top * scaleY,
                box.right * scaleX,
                box.bottom * scaleY
            )

            canvas.drawRect(mappedBox, currentBoxPaint)
        }

        if (roadOverlayVisible && warningMode == "LDW") {
            canvas.drawLine(w * 0.25f, h, w * 0.45f, h * 0.55f, warningPaint)
            canvas.drawLine(w * 0.75f, h, w * 0.55f, h * 0.55f, warningPaint)
        }
    }
}