package com.example.adas_fyp

import android.graphics.Rect
import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.core.Rect as CvRect
import org.opencv.imgproc.Imgproc

data class ForwardCollisionResult(
    val objectDetected: Boolean,
    val boundingBox: Rect?,
    val areaRatio: Float,
    val centralObject: Boolean,
    val collisionWarning: Boolean
)

class ForwardCollisionDetector {

    fun detect(
        imageProxy: ImageProxy,
        onResult: (ForwardCollisionResult) -> Unit
    ) {
        val width = imageProxy.width
        val height = imageProxy.height

        val gray = imageProxyToGrayMat(imageProxy)

        val blurred = Mat()
        val edges = Mat()
        val dilated = Mat()
        val mask = Mat.zeros(gray.size(), CvType.CV_8UC1)
        val roiEdges = Mat()
        val hierarchy = Mat()

        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 40.0, 120.0)

        val kernel = Imgproc.getStructuringElement(
            Imgproc.MORPH_RECT,
            Size(7.0, 7.0)
        )

        Imgproc.dilate(edges, dilated, kernel)

        val roiLeft = width * 0.25
        val roiTop = height * 0.20
        val roiRight = width * 0.75
        val roiBottom = height * 0.90

        Imgproc.rectangle(
            mask,
            Point(roiLeft, roiTop),
            Point(roiRight, roiBottom),
            Scalar(255.0),
            -1
        )

        Core.bitwise_and(dilated, mask, roiEdges)

        val contours = ArrayList<MatOfPoint>()
        Imgproc.findContours(
            roiEdges,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        var bestRect: CvRect? = null
        var bestArea = 0.0

        for (contour in contours) {
            val rect = Imgproc.boundingRect(contour)
            val area = rect.width * rect.height.toDouble()

            if (area > bestArea) {
                bestArea = area
                bestRect = rect
            }
        }

        val frameArea = width * height.toFloat()

        if (bestRect == null) {
            onResult(
                ForwardCollisionResult(
                    objectDetected = false,
                    boundingBox = null,
                    areaRatio = 0f,
                    centralObject = false,
                    collisionWarning = false
                )
            )
        } else {
            val rect = bestRect!!

            val areaRatio = (rect.width * rect.height).toFloat() / frameArea

            val centerX = (rect.x + rect.width / 2f) / width
            val centerY = (rect.y + rect.height / 2f) / height

            val isCentral =
                centerX in 0.30f..0.70f &&
                        centerY in 0.20f..0.90f

            val objectWidthRatio = rect.width.toFloat() / width
            val objectHeightRatio = rect.height.toFloat() / height

            val warning =
                isCentral &&
                        areaRatio >= 0.18f &&
                        objectWidthRatio >= 0.25f &&
                        objectHeightRatio >= 0.20f

            val androidRect = Rect(
                rect.x,
                rect.y,
                rect.x + rect.width,
                rect.y + rect.height
            )

            onResult(
                ForwardCollisionResult(
                    objectDetected = true,
                    boundingBox = androidRect,
                    areaRatio = areaRatio,
                    centralObject = isCentral,
                    collisionWarning = warning
                )
            )
        }

        gray.release()
        blurred.release()
        edges.release()
        dilated.release()
        mask.release()
        roiEdges.release()
        hierarchy.release()
        kernel.release()

        contours.forEach { it.release() }

        imageProxy.close()
    }

    private fun imageProxyToGrayMat(imageProxy: ImageProxy): Mat {
        val width = imageProxy.width
        val height = imageProxy.height

        val yPlane = imageProxy.planes[0]
        val buffer = yPlane.buffer
        val rowStride = yPlane.rowStride

        val data = ByteArray(width * height)

        buffer.rewind()

        for (row in 0 until height) {
            val rowStart = row * rowStride

            if (rowStart + width <= buffer.limit()) {
                buffer.position(rowStart)
                buffer.get(data, row * width, width)
            }
        }

        val mat = Mat(height, width, CvType.CV_8UC1)
        mat.put(0, 0, data)

        return mat
    }

    fun close() {
        // No external detector to close for OpenCV version
    }
}