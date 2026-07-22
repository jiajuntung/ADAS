package com.example.adas_fyp

import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

data class DrowsinessResult(
    val faceDetected: Boolean,
    val leftEyeOpen: Float?,
    val rightEyeOpen: Float?,
    val drowsy: Boolean,
    // Diagnostics: how many faces were in view, and where the one we picked
    // sits (0 = left edge of image, 1 = right edge). Shown in the status
    // panel so the driver-side setting can be verified rather than guessed.
    val faceCount: Int = 0,
    val driverFaceX: Float = -1f
)

/**
 * Driver monitoring (ML Kit face detection -- model unchanged).
 *
 * WHY DRIVER SELECTION EXISTS NOW
 * It used to call faces.first(), i.e. whatever ML Kit happened to return
 * first. With a passenger in the car that is a coin flip, so a sleeping
 * PASSENGER raised a drowsiness warning at the driver. A DMS that watches the
 * wrong person is worse than none: it trains the driver to ignore it.
 *
 * Malaysia drives on the left, so the car is right-hand drive and the driver
 * sits on the car's RIGHT. But which side of the CAMERA IMAGE that is depends
 * on front-camera mirroring and on the rotation ML Kit applied -- and those
 * vary by device. Rather than hard-code a guess that silently monitors the
 * passenger, the side is a setting, the choice is reported in the status
 * panel (faces + the chosen face's x), and it can be flipped in one tap.
 *
 * Faces are chosen by x-position, and a face on the wrong side is ignored
 * entirely: if the driver's face cannot be seen, the honest output is "no
 * driver face", not "let's watch the passenger instead".
 */
class DrowsinessDetector {

    private val eyeOpenThreshold = 0.35f

    /**
     * Which side of the ML Kit image the driver occupies.
     * Default RIGHT; flip in Settings if the status panel shows it locking
     * onto the passenger.
     */
    var driverOnRight = false

    // Faces nearer the centre than this are ambiguous; with two occupants the
    // driver is clearly on one side, so an ambiguous face is not trusted when
    // another face is present.
    private val sideMarginFromCentre = 0.06f

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

        val rotation = imageProxy.imageInfo.rotationDegrees

        // ML Kit reports boxes in the ROTATED frame, so the effective width is
        // the swapped dimension for 90/270. Getting this wrong would make the
        // side test meaningless.
        val effectiveWidth = if (rotation == 90 || rotation == 270) {
            imageProxy.height
        } else {
            imageProxy.width
        }

        val image = InputImage.fromMediaImage(mediaImage, rotation)

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    onResult(noFaceResult())
                    return@addOnSuccessListener
                }

                val driver = selectDriverFace(faces, effectiveWidth)

                if (driver == null) {
                    // Faces are present, but none on the driver's side.
                    onResult(
                        DrowsinessResult(
                            faceDetected = false,
                            leftEyeOpen = null,
                            rightEyeOpen = null,
                            drowsy = false,
                            faceCount = faces.size,
                            driverFaceX = -1f
                        )
                    )
                    return@addOnSuccessListener
                }

                val leftEye = driver.leftEyeOpenProbability
                val rightEye = driver.rightEyeOpenProbability

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
                        drowsy = isDrowsy,
                        faceCount = faces.size,
                        driverFaceX = faceCentreX(driver, effectiveWidth)
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

    private fun faceCentreX(face: Face, effectiveWidth: Int): Float {
        if (effectiveWidth <= 0) return -1f
        val box = face.boundingBox
        return ((box.left + box.right) / 2f) / effectiveWidth
    }

    /**
     * Pick the face on the driver's side. With several faces, take the
     * outermost one on that side -- the driver sits against their door, so
     * they are the extreme face, not the middle one.
     */
    private fun selectDriverFace(faces: List<Face>, effectiveWidth: Int): Face? {
        if (effectiveWidth <= 0) return null

        val single = faces.size == 1

        var best: Face? = null
        var bestX = if (driverOnRight) -1f else 2f

        for (face in faces) {
            val x = faceCentreX(face, effectiveWidth)
            if (x < 0f) continue

            val onDriverSide = if (driverOnRight) {
                x > 0.5f + sideMarginFromCentre
            } else {
                x < 0.5f - sideMarginFromCentre
            }

            // A lone face slightly off-centre is almost certainly the driver
            // in a solo drive; requiring a hard side test there would disable
            // DMS whenever the phone sits a little off-centre.
            val acceptable = onDriverSide || (single && isOnDriverHalf(x))

            if (!acceptable) continue

            if (driverOnRight) {
                if (x > bestX) { bestX = x; best = face }
            } else {
                if (x < bestX) { bestX = x; best = face }
            }
        }

        return best
    }

    private fun isOnDriverHalf(x: Float): Boolean =
        if (driverOnRight) x >= 0.42f else x <= 0.58f

    private fun noFaceResult() = DrowsinessResult(
        faceDetected = false,
        leftEyeOpen = null,
        rightEyeOpen = null,
        drowsy = false,
        faceCount = 0,
        driverFaceX = -1f
    )

    fun close() {
        detector.close()
    }
}