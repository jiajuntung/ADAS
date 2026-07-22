package com.example.adas_fyp

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

/** One surviving detection. Lane membership is decided by the caller, in metres. */
data class YoloCandidate(
    val boundingBox: Rect,
    val classId: Int,
    val className: String,
    val confidence: Float,
    val areaRatio: Float
)

data class YoloDetectionResult(
    val objectDetected: Boolean,
    val boundingBox: Rect?,
    val classId: Int?,
    val className: String?,
    val confidence: Float,
    val areaRatio: Float,
    val centralObject: Boolean,
    val collisionWarning: Boolean,
    val imageWidth: Int,
    val imageHeight: Int,
    // All plausible vehicles/VRUs this frame, so the caller can pick the one
    // that is actually IN OUR LANE using distance-based geometry.
    val candidates: List<YoloCandidate> = emptyList()
)

/**
 * YOLOv8n wrapper. Model unchanged; the SELECTION logic around it changed.
 *
 * WHY IT NO LONGER PICKS THE "BEST" OBJECT ITSELF
 * It used to return exactly one detection: the highest
 *     riskScore = confidence * areaRatio * centreWeight
 * among boxes whose centre x fell in 0.33..0.67, and it set collisionWarning
 * from areaRatio >= 0.040.
 *
 * Both of those are decisions this class cannot make correctly, because it has
 * no notion of distance:
 *
 *   - "Centre x in 0.33..0.67" is an ANGLE test, not a lane test. Beyond ~25 m
 *     an adjacent-lane bus projects to x ~0.55-0.59 and lands inside the
 *     window. Worse, being big, it wins riskScore outright and HIDES the real
 *     lead vehicle -- so the caller never even sees the car it should track.
 *     This is the "bus beside me triggers BRAKE" report.
 *   - areaRatio conflates width with height and knows nothing about speed.
 *
 * So this class now does what it can actually do -- decode, threshold, and
 * discard obvious non-objects -- and hands back ALL plausible candidates.
 * Lane membership is decided by MainActivity in metres, where it is a real
 * geometric question with a right answer.
 */
class YoloDetector(context: Context) {

    private val inputSize = 320
    private val outputChannels = 84
    private val outputCount = 2100

    private val confidenceThreshold = 0.25f
    private val maxCandidates = 8

    // Only reject things that cannot be a road user ahead, whatever the lane.
    private val ownVehicleIgnoreTopRatio = 0.58f
    private val ownVehicleIgnoreBottomRatio = 0.90f
    private val wideOwnVehicleAspectRatio = 3.0f
    private val minObjectBottomRatio = 0.30f

    private val targetClasses = mapOf(
        0 to "person",
        1 to "bicycle",
        2 to "car",
        3 to "motorcycle",
        5 to "bus",
        7 to "truck"
    )

    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(loadModelFile(context, "yolov8n_float32.tflite"), options)
    }

    fun detect(imageProxy: ImageProxy, onResult: (YoloDetectionResult) -> Unit) {
        try {
            val originalBitmap = imageProxyToBitmap(imageProxy)
            val resizedBitmap = Bitmap.createScaledBitmap(originalBitmap, inputSize, inputSize, true)
            val inputBuffer = bitmapToInputBuffer(resizedBitmap)

            val output = Array(1) { Array(outputChannels) { FloatArray(outputCount) } }
            interpreter.run(inputBuffer, output)

            val result = parseOutput(output[0], originalBitmap.width, originalBitmap.height)
            onResult(result)

            resizedBitmap.recycle()
            originalBitmap.recycle()
        } catch (e: Exception) {
            e.printStackTrace()
            onResult(noObjectResult())
        } finally {
            imageProxy.close()
        }
    }

    private fun parseOutput(
        output: Array<FloatArray>,
        imageWidth: Int,
        imageHeight: Int
    ): YoloDetectionResult {
        val found = ArrayList<YoloCandidate>()

        for (i in 0 until outputCount) {
            val centerX = output[0][i]
            val centerY = output[1][i]
            val boxWidth = output[2][i]
            val boxHeight = output[3][i]

            var currentClassId: Int? = null
            var currentClassName: String? = null
            var currentConfidence = 0f

            for ((classId, className) in targetClasses) {
                val score = output[4 + classId][i]
                if (score > currentConfidence) {
                    currentConfidence = score
                    currentClassId = classId
                    currentClassName = className
                }
            }

            if (currentConfidence < confidenceThreshold) continue
            if (currentClassId == null || currentClassName == null) continue

            val isNormalizedBox =
                centerX <= 1.5f && centerY <= 1.5f && boxWidth <= 1.5f && boxHeight <= 1.5f

            val scaleX = if (isNormalizedBox) imageWidth.toFloat() else imageWidth.toFloat() / inputSize
            val scaleY = if (isNormalizedBox) imageHeight.toFloat() else imageHeight.toFloat() / inputSize

            val left = ((centerX - boxWidth / 2f) * scaleX).roundToInt().coerceIn(0, imageWidth)
            val top = ((centerY - boxHeight / 2f) * scaleY).roundToInt().coerceIn(0, imageHeight)
            val right = ((centerX + boxWidth / 2f) * scaleX).roundToInt().coerceIn(0, imageWidth)
            val bottom = ((centerY + boxHeight / 2f) * scaleY).roundToInt().coerceIn(0, imageHeight)

            val rect = Rect(left, top, right, bottom)
            if (rect.width() <= 0 || rect.height() <= 0) continue

            val bottomY = rect.bottom.toFloat() / imageHeight
            val topY = rect.top.toFloat() / imageHeight
            val widthRatio = rect.width().toFloat() / imageWidth
            val heightRatio = rect.height().toFloat() / imageHeight
            val aspect = widthRatio / heightRatio.coerceAtLeast(0.001f)

            // Our own bonnet / a wide smear across the bottom is not a vehicle.
            val isOwnVehiclePart =
                topY >= ownVehicleIgnoreTopRatio && bottomY >= ownVehicleIgnoreBottomRatio
            val isWideBottomObject = topY >= 0.50f && aspect >= wideOwnVehicleAspectRatio

            if (isOwnVehiclePart || isWideBottomObject) continue

            // Must stand on the road ahead, not float in the sky.
            if (bottomY < minObjectBottomRatio) continue

            val areaRatio = rect.width() * rect.height().toFloat() / (imageWidth * imageHeight)

            found.add(
                YoloCandidate(rect, currentClassId, currentClassName, currentConfidence, areaRatio)
            )
        }

        // Merge duplicate boxes on the same object, then keep the strongest few.
        val merged = nms(found, 0.45f).sortedByDescending { it.confidence }.take(maxCandidates)

        if (merged.isEmpty()) {
            return noObjectResult(imageWidth, imageHeight)
        }

        // The nearest-looking candidate is only a display default; the real
        // in-lane choice happens in MainActivity, in metres.
        val display = merged.maxByOrNull { it.areaRatio }!!

        return YoloDetectionResult(
            objectDetected = true,
            boundingBox = display.boundingBox,
            classId = display.classId,
            className = display.className,
            confidence = display.confidence,
            areaRatio = display.areaRatio,
            centralObject = false,
            collisionWarning = false,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            candidates = merged
        )
    }

    private fun nms(list: List<YoloCandidate>, iouThreshold: Float): List<YoloCandidate> {
        val sorted = list.sortedByDescending { it.confidence }.toMutableList()
        val keep = ArrayList<YoloCandidate>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            keep.add(best)
            sorted.removeAll { iou(best.boundingBox, it.boundingBox) > iouThreshold }
        }

        return keep
    }

    private fun iou(a: Rect, b: Rect): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)
        val inter = maxOf(0, right - left).toFloat() * maxOf(0, bottom - top).toFloat()
        val union = a.width().toFloat() * a.height() + b.width().toFloat() * b.height() - inter
        return if (union > 0f) inter / union else 0f
    }

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            inputBuffer.putFloat((pixel shr 16 and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixel shr 8 and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val nv21 = imageProxyToNv21(imageProxy)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 90, outputStream)
        val jpegBytes = outputStream.toByteArray()
        val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
        return rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees)
    }

    private fun imageProxyToNv21(imageProxy: ImageProxy): ByteArray {
        val width = imageProxy.width
        val height = imageProxy.height
        val ySize = width * height
        val nv21 = ByteArray(width * height * 3 / 2)

        copyPlane(imageProxy.planes[0], width, height, nv21, 0, 1)
        copyPlane(imageProxy.planes[2], width / 2, height / 2, nv21, ySize, 2)
        copyPlane(imageProxy.planes[1], width / 2, height / 2, nv21, ySize + 1, 2)

        return nv21
    }

    private fun copyPlane(
        plane: ImageProxy.PlaneProxy,
        width: Int,
        height: Int,
        output: ByteArray,
        outputOffset: Int,
        outputPixelStride: Int
    ) {
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        var outputIndex = outputOffset

        for (row in 0 until height) {
            for (col in 0 until width) {
                val inputIndex = row * rowStride + col * pixelStride
                if (inputIndex < buffer.limit() && outputIndex < output.size) {
                    output[outputIndex] = buffer.get(inputIndex)
                }
                outputIndex += outputPixelStride
            }
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees == 0) return bitmap
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun loadModelFile(context: Context, modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun noObjectResult(imageWidth: Int = 0, imageHeight: Int = 0) = YoloDetectionResult(
        objectDetected = false,
        boundingBox = null,
        classId = null,
        className = null,
        confidence = 0f,
        areaRatio = 0f,
        centralObject = false,
        collisionWarning = false,
        imageWidth = imageWidth,
        imageHeight = imageHeight,
        candidates = emptyList()
    )

    fun close() {
        interpreter.close()
    }
}