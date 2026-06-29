package com.example.adas_fyp

import androidx.camera.core.ImageProxy
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot

data class LaneDetectionResult(
    val leftDetected: Boolean,
    val rightDetected: Boolean,
    val lineCount: Int,
    val laneWarning: Boolean,
    val laneCenterX: Float? = null,
    val vehicleCenterX: Float = 0f,
    val offsetRatio: Float = 0f
)

class LaneDetector {

    private val minSlope = 0.45
    private val maxSlope = 3.0
    private val warningOffsetRatio = 0.07f

    fun detect(imageProxy: ImageProxy): LaneDetectionResult {
        val width = imageProxy.width
        val height = imageProxy.height

        val gray = imageProxyToGrayMat(imageProxy)

        val blurred = Mat()
        val edges = Mat()
        val mask = Mat.zeros(gray.size(), CvType.CV_8UC1)
        val roiEdges = Mat()
        val lines = Mat()

        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)

        val roiTopY = height * 0.45
        val roiBottomY = height * 0.98

        val roiPoints = MatOfPoint(
            Point(width * 0.00, roiBottomY),
            Point(width * 0.22, roiTopY),
            Point(width * 0.78, roiTopY),
            Point(width * 1.00, roiBottomY)
        )

        Imgproc.fillConvexPoly(mask, roiPoints, Scalar(255.0))
        Core.bitwise_and(edges, mask, roiEdges)

        Imgproc.HoughLinesP(
            roiEdges,
            lines,
            1.0,
            Math.PI / 180,
            45,
            45.0,
            80.0
        )

        val leftLines = mutableListOf<LaneLine>()
        val rightLines = mutableListOf<LaneLine>()

        for (i in 0 until lines.rows()) {
            val line = lines.get(i, 0) ?: continue

            val x1 = line[0]
            val y1 = line[1]
            val x2 = line[2]
            val y2 = line[3]

            val dx = x2 - x1
            val dy = y2 - y1

            if (abs(dx) < 1.0) {
                continue
            }

            val slope = dy / dx

            if (abs(slope) < minSlope || abs(slope) > maxSlope) {
                continue
            }

            val intercept = y1 - slope * x1
            val length = hypot(dx, dy)

            if (slope < 0) {
                leftLines.add(LaneLine(slope, intercept, length))
            } else {
                rightLines.add(LaneLine(slope, intercept, length))
            }
        }

        val leftLane = averageLine(leftLines)
        val rightLane = averageLine(rightLines)

        val leftDetected = leftLane != null
        val rightDetected = rightLane != null
        val validLineCount = leftLines.size + rightLines.size

        if (leftLane == null || rightLane == null) {
            val laneBottomY = height * 0.90
            val vehicleCenterX = width / 2f

            var singleLineWarning = false
            var singleLineOffset = 0f

            if (leftLane != null) {
                val leftBottomX = xAtY(leftLane, laneBottomY)

                singleLineOffset = (vehicleCenterX - leftBottomX) / width

                // Left lane line is too close to vehicle center
                singleLineWarning = leftBottomX > width * 0.38f
            }

            if (rightLane != null) {
                val rightBottomX = xAtY(rightLane, laneBottomY)

                singleLineOffset = (rightBottomX - vehicleCenterX) / width

                // Right lane line is too close to vehicle center
                singleLineWarning = rightBottomX < width * 0.62f
            }

            releaseMats(gray, blurred, edges, mask, roiEdges, lines, roiPoints)

            return LaneDetectionResult(
                leftDetected = leftDetected,
                rightDetected = rightDetected,
                lineCount = validLineCount,
                laneWarning = singleLineWarning,
                laneCenterX = null,
                vehicleCenterX = vehicleCenterX,
                offsetRatio = singleLineOffset
            )
        }

        val laneBottomY = height * 0.90

        val leftBottomX = xAtY(leftLane, laneBottomY)
        val rightBottomX = xAtY(rightLane, laneBottomY)

        val laneWidth = rightBottomX - leftBottomX

        val laneShapeValid =
            leftBottomX in 0f..(width * 0.60f) &&
                    rightBottomX in (width * 0.40f)..width.toFloat() &&
                    laneWidth in (width * 0.30f)..(width * 0.95f)

        if (!laneShapeValid) {
            releaseMats(gray, blurred, edges, mask, roiEdges, lines, roiPoints)

            return LaneDetectionResult(
                leftDetected = true,
                rightDetected = true,
                lineCount = validLineCount,
                laneWarning = false,
                laneCenterX = null,
                vehicleCenterX = width / 2f,
                offsetRatio = 0f
            )
        }

        val laneCenterX = (leftBottomX + rightBottomX) / 2f
        val vehicleCenterX = width / 2f
        val offsetRatio = (vehicleCenterX - laneCenterX) / width

        val laneWarning = abs(offsetRatio) >= warningOffsetRatio

        releaseMats(gray, blurred, edges, mask, roiEdges, lines, roiPoints)

        return LaneDetectionResult(
            leftDetected = true,
            rightDetected = true,
            lineCount = validLineCount,
            laneWarning = laneWarning,
            laneCenterX = laneCenterX,
            vehicleCenterX = vehicleCenterX,
            offsetRatio = offsetRatio
        )
    }

    private fun averageLine(lines: List<LaneLine>): LaneLine? {
        if (lines.isEmpty()) {
            return null
        }

        val totalLength = lines.sumOf { it.length }

        if (totalLength <= 0.0) {
            return null
        }

        val avgSlope = lines.sumOf { it.slope * it.length } / totalLength
        val avgIntercept = lines.sumOf { it.intercept * it.length } / totalLength

        return LaneLine(avgSlope, avgIntercept, totalLength)
    }

    private fun xAtY(
        line: LaneLine,
        y: Double
    ): Float {
        return ((y - line.intercept) / line.slope).toFloat()
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

    private fun releaseMats(
        gray: Mat,
        blurred: Mat,
        edges: Mat,
        mask: Mat,
        roiEdges: Mat,
        lines: Mat,
        roiPoints: MatOfPoint
    ) {
        gray.release()
        blurred.release()
        edges.release()
        mask.release()
        roiEdges.release()
        lines.release()
        roiPoints.release()
    }

    private data class LaneLine(
        val slope: Double,
        val intercept: Double,
        val length: Double
    )
}