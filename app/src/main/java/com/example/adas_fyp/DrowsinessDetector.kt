package com.example.adas_fyp

import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

data class DrowsinessResult(
    val faceDetected: Boolean,
    val leftEyeOpen: Float?,
    val rightEyeOpen: Float?,
    val drowsy: Boolean
)

class DrowsinessDetector {

    private val eyeOpenThreshold = 0.35f

    private val options = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .enableTracking()
        .build()

    private val detector = FaceDetection.getClient(options)

    @OptIn(markerClass = [ExperimentalGetImage::class])
    fun detect(
        imageProxy: ImageProxy,
        onResult: (DrowsinessResult) -> Unit
    ) {
        val mediaImage = imageProxy.image

        if (mediaImage == null) {
            onResult(noFaceResult())
            imageProxy.close()
            return
        }

        val image = InputImage.fromMediaImage(
            mediaImage,
            imageProxy.imageInfo.rotationDegrees
        )

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    onResult(noFaceResult())
                    return@addOnSuccessListener
                }

                val face = faces.first()

                val leftEye = face.leftEyeOpenProbability
                val rightEye = face.rightEyeOpenProbability

                val avgEyeOpen = if (leftEye != null && rightEye != null) {
                    (leftEye + rightEye) / 2f
                } else {
                    null
                }

                val isDrowsy = avgEyeOpen != null && avgEyeOpen < eyeOpenThreshold

                onResult(
                    DrowsinessResult(
                        faceDetected = true,
                        leftEyeOpen = leftEye,
                        rightEyeOpen = rightEye,
                        drowsy = isDrowsy
                    )
                )
            }
            .addOnFailureListener {
                onResult(noFaceResult())
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    private fun noFaceResult(): DrowsinessResult {
        return DrowsinessResult(
            faceDetected = false,
            leftEyeOpen = null,
            rightEyeOpen = null,
            drowsy = false
        )
    }

    fun close() {
        detector.close()
    }
}