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

    private val minSlope = 0.35                 // was 0.45 — catch the shallower line you're crossing
    private val maxSlope = 3.0
    private val warningOffsetRatio = 0.07f      // more sensitive for lane crossing
    private val centerDangerMinX = 0.38f        // straddle zone widened slightly
    private val centerDangerMaxX = 0.62f

    // Cross-frame smoothing. Lane fitting from a phone cam jumps around frame to
    // frame; smoothing the fitted line (and rejecting sudden jumps as outliers) is
    // the single biggest thing that stops LDW from false-triggering.
    private var smoothLeft: LaneLine? = null
    private var smoothRight: LaneLine? = null
    private val laneEma = 0.55                  // was 0.7 — follow a real crossing faster
    private val slopeOutlierTol = 0.35         // ignore a new line whose slope jumps more than this
    private var leftMissStreak = 0
    private var rightMissStreak = 0
    private val maxMissFrames = 5              // hold a smoothed lane through up to N missed frames

    fun detect(imageProxy: ImageProxy): LaneDetectionResult {
        val width = imageProxy.width
        val height = imageProxy.height

        val gray = imageProxyToGrayMat(imageProxy)

        val blurred = Mat()
        val edges = Mat()
        val mask = Mat.zeros(gray.size(), CvType.CV_8UC1)
        val roiEdges = Mat()
        val lines = Mat()

        // White top-hat: isolates bright, thin features (lane markings) regardless of
        // overall brightness, so faint / worn lines -- and the line you're crossing --
        // are detected far more reliably than with raw edges alone.
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(13.0, 13.0))
        val opened = Mat()
        val tophat = Mat()
        Imgproc.morphologyEx(gray, opened, Imgproc.MORPH_OPEN, kernel)
        Core.subtract(gray, opened, tophat)

        // Edges from the raw image (general structure) OR from the enhanced markings.
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 40.0, 120.0)

        val topBlur = Mat()
        val topEdges = Mat()
        Imgproc.GaussianBlur(tophat, topBlur, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(topBlur, topEdges, 20.0, 60.0)

        val allEdges = Mat()
        Core.bitwise_or(edges, topEdges, allEdges)

        val roiTopY = height * 0.45
        val roiBottomY = height * 0.98

        val roiPoints = MatOfPoint(
            Point(width * 0.00, roiBottomY),
            Point(width * 0.22, roiTopY),
            Point(width * 0.78, roiTopY),
            Point(width * 1.00, roiBottomY)
        )

        Imgproc.fillConvexPoly(mask, roiPoints, Scalar(255.0))
        Core.bitwise_and(allEdges, mask, roiEdges)

        Imgproc.HoughLinesP(
            roiEdges,
            lines,
            1.0,
            Math.PI / 180,
            30,
            25.0,
            110.0
        )

        // Release the enhancement intermediates now, before the early returns below.
        opened.release()
        tophat.release()
        topBlur.release()
        topEdges.release()
        allEdges.release()
        kernel.release()

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

        val rawLeft = averageLine(leftLines)
        val rawRight = averageLine(rightLines)
        val validLineCount = leftLines.size + rightLines.size

        // Update the smoothed lanes (outlier-rejected EMA), holding each lane through
        // a few missed frames so one bad frame doesn't drop it.
        if (rawLeft != null) {
            smoothLeft = blendLane(smoothLeft, rawLeft)
            leftMissStreak = 0
        } else if (++leftMissStreak > maxMissFrames) {
            smoothLeft = null
        }

        if (rawRight != null) {
            smoothRight = blendLane(smoothRight, rawRight)
            rightMissStreak = 0
        } else if (++rightMissStreak > maxMissFrames) {
            smoothRight = null
        }

        val leftLane = smoothLeft
        val rightLane = smoothRight
        val leftDetected = leftLane != null
        val rightDetected = rightLane != null

        val vehicleCenterX = width / 2f
        val laneBottomY = height * 0.90

        if (leftLane == null || rightLane == null) {
            // Single line: only warn when the visible line has drifted into the centre
            // danger zone. The old "past 0.35 / 0.65" test sat on the normal driving
            // position and fired constantly.
            var singleLineWarning = false
            var singleLineOffset = 0f

            if (leftLane != null) {
                val leftBottomX = xAtY(leftLane, laneBottomY)
                singleLineOffset = (vehicleCenterX - leftBottomX) / width
                singleLineWarning =
                    leftBottomX in (width * centerDangerMinX)..(width * centerDangerMaxX)
            } else if (rightLane != null) {
                val rightBottomX = xAtY(rightLane, laneBottomY)
                singleLineOffset = (rightBottomX - vehicleCenterX) / width
                singleLineWarning =
                    rightBottomX in (width * centerDangerMinX)..(width * centerDangerMaxX)
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

        val leftBottomX = xAtY(leftLane, laneBottomY)
        val rightBottomX = xAtY(rightLane, laneBottomY)

        val leftLineInCenterDangerZone =
            leftBottomX in (width * centerDangerMinX)..(width * centerDangerMaxX)

        val rightLineInCenterDangerZone =
            rightBottomX in (width * centerDangerMinX)..(width * centerDangerMaxX)

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
                vehicleCenterX = vehicleCenterX,
                offsetRatio = 0f
            )
        }

        val laneCenterX = (leftBottomX + rightBottomX) / 2f
        val offsetRatio = (vehicleCenterX - laneCenterX) / width

        val laneWarning =
            abs(offsetRatio) >= warningOffsetRatio ||
                    leftLineInCenterDangerZone ||
                    rightLineInCenterDangerZone

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

    // Outlier-rejected EMA blend of the new fit into the smoothed lane.
    private fun blendLane(prev: LaneLine?, new: LaneLine): LaneLine {
        if (prev == null) {
            return new
        }

        // A sudden slope jump is almost always a bad fit -> keep the previous smoothed line.
        if (abs(new.slope - prev.slope) > slopeOutlierTol) {
            return prev
        }

        val s = laneEma * prev.slope + (1 - laneEma) * new.slope
        val i = laneEma * prev.intercept + (1 - laneEma) * new.intercept

        return LaneLine(s, i, new.length)
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