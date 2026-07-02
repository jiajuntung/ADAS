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
import kotlin.math.abs

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
    val imageHeight: Int
)

class YoloDetector(context: Context) {

    // Model configuration
    private val inputSize = 320
    private val outputChannels = 84
    private val outputCount = 2100

    // Detection thresholds
    private val confidenceThreshold = 0.20f
    private val brakeConfidenceThreshold = 0.50f
    private val minAreaRatio = 0.06f
    private val minWidthRatio = 0.16f
    private val minHeightRatio = 0.16f
    private val egoLaneMinX = 0.30f
    private val egoLaneMaxX = 0.70f
    private val minObjectBottomRatio = 0.35f
    private val ownVehicleIgnoreTopRatio = 0.58f
    private val ownVehicleIgnoreBottomRatio = 0.90f
    private val wideOwnVehicleAspectRatio = 3.0f
    private val brakeLaneMinX = 0.40f
    private val brakeLaneMaxX = 0.60f

    // COCO class IDs used for FCW
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
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }

        interpreter = Interpreter(
            loadModelFile(context, "yolov8n_float32.tflite"),
            options
        )
    }

    fun detect(
        imageProxy: ImageProxy,
        onResult: (YoloDetectionResult) -> Unit
    ) {
        try {
            val originalBitmap = imageProxyToBitmap(imageProxy)

            val resizedBitmap = Bitmap.createScaledBitmap(
                originalBitmap,
                inputSize,
                inputSize,
                true
            )

            val inputBuffer = bitmapToInputBuffer(resizedBitmap)

            val output = Array(1) {
                Array(outputChannels) {
                    FloatArray(outputCount)
                }
            }

            interpreter.run(inputBuffer, output)

            val result = parseOutput(
                output = output[0],
                imageWidth = originalBitmap.width,
                imageHeight = originalBitmap.height
            )

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
        var bestRiskScore = 0f
        var bestConfidence = 0f
        var bestClassId: Int? = null
        var bestClassName: String? = null
        var bestRect: Rect? = null

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

            if (currentConfidence < confidenceThreshold) {
                continue
            }

            val isNormalizedBox =
                centerX <= 1.5f &&
                        centerY <= 1.5f &&
                        boxWidth <= 1.5f &&
                        boxHeight <= 1.5f

            val left: Int
            val top: Int
            val right: Int
            val bottom: Int

            if (isNormalizedBox) {
                left = ((centerX - boxWidth / 2f) * imageWidth)
                    .roundToInt()
                    .coerceIn(0, imageWidth)

                top = ((centerY - boxHeight / 2f) * imageHeight)
                    .roundToInt()
                    .coerceIn(0, imageHeight)

                right = ((centerX + boxWidth / 2f) * imageWidth)
                    .roundToInt()
                    .coerceIn(0, imageWidth)

                bottom = ((centerY + boxHeight / 2f) * imageHeight)
                    .roundToInt()
                    .coerceIn(0, imageHeight)
            } else {
                left = ((centerX - boxWidth / 2f) / inputSize * imageWidth)
                    .roundToInt()
                    .coerceIn(0, imageWidth)

                top = ((centerY - boxHeight / 2f) / inputSize * imageHeight)
                    .roundToInt()
                    .coerceIn(0, imageHeight)

                right = ((centerX + boxWidth / 2f) / inputSize * imageWidth)
                    .roundToInt()
                    .coerceIn(0, imageWidth)

                bottom = ((centerY + boxHeight / 2f) / inputSize * imageHeight)
                    .roundToInt()
                    .coerceIn(0, imageHeight)
            }

            val rect = Rect(left, top, right, bottom)

            if (rect.width() <= 0 || rect.height() <= 0) {
                continue
            }

            val candidateCenterX =
                (rect.left + rect.width() / 2f) / imageWidth

            val candidateBottomY =
                rect.bottom.toFloat() / imageHeight

            val candidateAreaRatio =
                rect.width() * rect.height().toFloat() / (imageWidth * imageHeight)

            val candidateTopY =
                rect.top.toFloat() / imageHeight

            val candidateWidthRatio =
                rect.width().toFloat() / imageWidth

            val candidateHeightRatio =
                rect.height().toFloat() / imageHeight

            val candidateAspectRatio =
                candidateWidthRatio / candidateHeightRatio.coerceAtLeast(0.001f)

            val isOwnVehiclePart =
                candidateTopY >= ownVehicleIgnoreTopRatio &&
                        candidateBottomY >= ownVehicleIgnoreBottomRatio

            val isWideBottomObject =
                candidateTopY >= 0.50f &&
                        candidateAspectRatio >= wideOwnVehicleAspectRatio

            if (isOwnVehiclePart || isWideBottomObject) {
                continue
            }

            val isInEgoLane =
                candidateCenterX in egoLaneMinX..egoLaneMaxX &&
                        candidateBottomY >= minObjectBottomRatio

            if (!isInEgoLane) {
                continue
            }

            val centerWeight =
                1f - (abs(candidateCenterX - 0.5f) * 2f)

            val riskScore =
                currentConfidence * candidateAreaRatio * centerWeight

            if (riskScore <= bestRiskScore) {
                continue
            }

            bestRiskScore = riskScore
            bestConfidence = currentConfidence
            bestClassId = currentClassId
            bestClassName = currentClassName
            bestRect = rect
        }

        val rect = bestRect ?: return noObjectResult(
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )

        val areaRatio =
            rect.width() * rect.height().toFloat() / (imageWidth * imageHeight)

        val objectCenterX =
            (rect.left + rect.width() / 2f) / imageWidth

        val objectCenterY =
            (rect.top + rect.height() / 2f) / imageHeight

        val objectWidthRatio =
            rect.width().toFloat() / imageWidth

        val objectHeightRatio =
            rect.height().toFloat() / imageHeight

        val isCentral =
            objectCenterX in brakeLaneMinX..brakeLaneMaxX &&
                    objectCenterY in 0.20f..0.90f

        val warning =
            isCentral &&
                    bestConfidence >= brakeConfidenceThreshold &&
                    areaRatio >= minAreaRatio &&
                    objectWidthRatio >= minWidthRatio &&
                    objectHeightRatio >= minHeightRatio

        return YoloDetectionResult(
            objectDetected = true,
            boundingBox = rect,
            classId = bestClassId,
            className = bestClassName,
            confidence = bestConfidence,
            areaRatio = areaRatio,
            centralObject = isCentral,
            collisionWarning = warning,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
    }

    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(
            1 * inputSize * inputSize * 3 * 4
        )

        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)

        bitmap.getPixels(
            pixels,
            0,
            inputSize,
            0,
            0,
            inputSize,
            inputSize
        )

        for (pixel in pixels) {
            val red = (pixel shr 16 and 0xFF) / 255.0f
            val green = (pixel shr 8 and 0xFF) / 255.0f
            val blue = (pixel and 0xFF) / 255.0f

            inputBuffer.putFloat(red)
            inputBuffer.putFloat(green)
            inputBuffer.putFloat(blue)
        }

        inputBuffer.rewind()

        return inputBuffer
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val nv21 = imageProxyToNv21(imageProxy)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            imageProxy.width,
            imageProxy.height,
            null
        )

        val outputStream = ByteArrayOutputStream()

        yuvImage.compressToJpeg(
            Rect(0, 0, imageProxy.width, imageProxy.height),
            90,
            outputStream
        )

        val jpegBytes = outputStream.toByteArray()

        val bitmap = BitmapFactory.decodeByteArray(
            jpegBytes,
            0,
            jpegBytes.size
        )

        return rotateBitmap(
            bitmap,
            imageProxy.imageInfo.rotationDegrees
        )
    }

    private fun imageProxyToNv21(imageProxy: ImageProxy): ByteArray {
        val width = imageProxy.width
        val height = imageProxy.height

        val ySize = width * height
        val nv21 = ByteArray(width * height * 3 / 2)

        copyPlane(
            plane = imageProxy.planes[0],
            width = width,
            height = height,
            output = nv21,
            outputOffset = 0,
            outputPixelStride = 1
        )

        copyPlane(
            plane = imageProxy.planes[2],
            width = width / 2,
            height = height / 2,
            output = nv21,
            outputOffset = ySize,
            outputPixelStride = 2
        )

        copyPlane(
            plane = imageProxy.planes[1],
            width = width / 2,
            height = height / 2,
            output = nv21,
            outputOffset = ySize + 1,
            outputPixelStride = 2
        )

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

    private fun rotateBitmap(
        bitmap: Bitmap,
        rotationDegrees: Int
    ): Bitmap {
        if (rotationDegrees == 0) {
            return bitmap
        }

        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())

        return Bitmap.createBitmap(
            bitmap,
            0,
            0,
            bitmap.width,
            bitmap.height,
            matrix,
            true
        )
    }

    private fun loadModelFile(
        context: Context,
        modelName: String
    ): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel

        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }

    private fun noObjectResult(
        imageWidth: Int = 0,
        imageHeight: Int = 0
    ): YoloDetectionResult {
        return YoloDetectionResult(
            objectDetected = false,
            boundingBox = null,
            classId = null,
            className = null,
            confidence = 0f,
            areaRatio = 0f,
            centralObject = false,
            collisionWarning = false,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
    }

    fun close() {
        interpreter.close()
    }
}