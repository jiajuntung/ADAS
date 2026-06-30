package com.example.adas_fyp

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.media.AudioManager
import android.media.ToneGenerator
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.util.Size
import org.opencv.android.OpenCVLoader
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ConcurrentCamera.SingleCameraConfig

class MainActivity : ComponentActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var frontPreviewView: PreviewView
    private lateinit var warningText: TextView
    private lateinit var safetyOverlay: SafetyOverlayView
    private lateinit var fpsText: TextView
    private lateinit var cameraExecutor: ExecutorService
    private val toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 50)

    private val laneDetector = LaneDetector()
    private var latestLaneStatus = "Lane: -"
    private var lastLaneWarningTime = 0L
    private val laneWarningCooldownMs = 5000L
    private var laneDepartureStartTime: Long? = null
    private val laneDepartureDurationMs = 1500L
    private var lastLaneIssueDetectedTime = 0L
    private val laneIssueGraceMs = 1000L

    private val drowsinessDetector = DrowsinessDetector()
    private var lastDrowsyWarningTime = 0L
    private var isDmsProcessing = false
    private var latestDmsStatus = "DMS: -"
    private val drowsyCooldownMs = 5000L
    private var eyeClosedStartTime: Long? = null
    private val drowsyDurationMs = 2000L

    private lateinit var yoloDetector: YoloDetector
    private var isFcwProcessing = false
    private var latestFcwStatus = "FCW: -"
    private var fcwWarningCounter = 0
    private var lastFcwWarningTime = 0L
    private var fcwDangerActive = false
    private val fcwWarningThreshold = 3
    private val fcwCooldownMs = 5000L
    private var useBackCamera = true

    private var backFrameCount = 0
    private var frontFrameCount = 0

    private var lastBackFpsTime = System.currentTimeMillis()
    private var lastFrontFpsTime = System.currentTimeMillis()

    private var latestBackFps = 0
    private var latestFrontFps = 0

    private var backAnalysisFrameIndex = 0
    private var frontAnalysisFrameIndex = 0

    private lateinit var switchCameraButton: Button
    private var isDualCameraSupported = false

    private lateinit var statusText: TextView

    private lateinit var btnTestFCW: Button
    private lateinit var btnTestLDW: Button
    private lateinit var btnTestDMS: Button
    private lateinit var btnStatusMode: Button
    private lateinit var btnPerformanceMode: Button

    private var ecoMode = false

    private var fcwFrameInterval = 10
    private var ldwFrameInterval = 3
    private var dmsFrameInterval = 8

    private lateinit var btnSettings: Button
    private lateinit var settingsPanel: View

    private var debugMode = true

    private var fcwSafeCounter = 0
    private val fcwSafeThreshold = 8

    private var leadVehicleSeenStartTime: Long? = null
    private var leadVehicleWaiting = false
    private var leadVehicleReferenceArea = 0f
    private var lastLeadVehicleArea = 0f
    private var lastLeadVehicleDetectedTime = 0L
    private var lastLeadVehicleAlertTime = 0L
    private var latestLeadVehicleStatus = "LVSA: -"

    private val leadVehicleClassIds = setOf(2, 3, 5, 7) // car, motorcycle, bus, truck
    private val leadVehicleStableMs = 3000L
    private val leadVehicleMovedDropRatio = 0.35f
    private val leadVehicleLostGraceMs = 1200L
    private val leadVehicleAlertCooldownMs = 10000L
    private val minLeadVehicleAreaRatio = 0.025f
    private val leadVehicleMinX = 0.38f
    private val leadVehicleMaxX = 0.62f

    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                startDualCamera()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
            }
        }

    private fun resetWarningStates() {
        fcwWarningCounter = 0
        fcwSafeCounter = 0
        fcwDangerActive = false
        laneDepartureStartTime = null
        eyeClosedStartTime = null
        resetLeadVehicleState()
    }

    private fun clearLeadVehicleTracking() {
        leadVehicleSeenStartTime = null
        leadVehicleWaiting = false
        leadVehicleReferenceArea = 0f
        lastLeadVehicleArea = 0f
        lastLeadVehicleDetectedTime = 0L
    }

    private fun resetLeadVehicleState() {
        clearLeadVehicleTracking()
        latestLeadVehicleStatus = "LVSA: -"
    }

    private fun isLeadVehicleCandidate(result: YoloDetectionResult): Boolean {
        val box = result.boundingBox ?: return false

        if (!result.objectDetected) {
            return false
        }

        if (result.classId !in leadVehicleClassIds) {
            return false
        }

        if (result.imageWidth <= 0 || result.imageHeight <= 0) {
            return false
        }

        val centerX =
            (box.left + box.width() / 2f) / result.imageWidth.toFloat()

        val bottomY =
            box.bottom.toFloat() / result.imageHeight.toFloat()

        return centerX in leadVehicleMinX..leadVehicleMaxX &&
                bottomY >= 0.35f &&
                result.areaRatio >= minLeadVehicleAreaRatio
    }

    private fun processLeadVehicleStartAlert(result: YoloDetectionResult) {
        val currentTime = System.currentTimeMillis()
        val leadDetected = isLeadVehicleCandidate(result)

        if (leadDetected) {
            val currentArea = result.areaRatio

            lastLeadVehicleDetectedTime = currentTime

            if (leadVehicleSeenStartTime == null) {
                leadVehicleSeenStartTime = currentTime
                leadVehicleReferenceArea = currentArea
                leadVehicleWaiting = false
            }

            val observedDuration =
                currentTime - (leadVehicleSeenStartTime ?: currentTime)

            if (!leadVehicleWaiting) {
                leadVehicleReferenceArea = maxOf(leadVehicleReferenceArea, currentArea)

                latestLeadVehicleStatus =
                    "LVSA: observing ${observedDuration / 1000.0}s"

                if (observedDuration >= leadVehicleStableMs) {
                    leadVehicleWaiting = true
                    latestLeadVehicleStatus = "LVSA: waiting"
                }
            } else {
                val movedByAreaDrop =
                    leadVehicleReferenceArea > 0f &&
                            currentArea <= leadVehicleReferenceArea * (1f - leadVehicleMovedDropRatio)

                latestLeadVehicleStatus =
                    "LVSA: waiting Area:${String.format("%.3f", currentArea)}"

                if (
                    movedByAreaDrop &&
                    !result.collisionWarning &&
                    currentTime - lastLeadVehicleAlertTime >= leadVehicleAlertCooldownMs
                ) {
                    lastLeadVehicleAlertTime = currentTime
                    latestLeadVehicleStatus = "LVSA: moved"

                    clearLeadVehicleTracking()

                    runOnUiThread {
                        showWarning(
                            "FRONT VEHICLE MOVED",
                            Color.parseColor("#CC2196F3")
                        )
                    }
                }
            }

            lastLeadVehicleArea = currentArea

        } else {
            if (leadVehicleWaiting) {
                val lostDuration = currentTime - lastLeadVehicleDetectedTime

                latestLeadVehicleStatus = "LVSA: waiting / lost"

                if (
                    lostDuration >= leadVehicleLostGraceMs &&
                    currentTime - lastLeadVehicleAlertTime >= leadVehicleAlertCooldownMs
                ) {
                    lastLeadVehicleAlertTime = currentTime
                    latestLeadVehicleStatus = "LVSA: moved"

                    clearLeadVehicleTracking()

                    runOnUiThread {
                        showWarning(
                            "FRONT VEHICLE MOVED",
                            Color.parseColor("#CC2196F3")
                        )
                    }
                }
            } else {
                if (
                    leadVehicleSeenStartTime != null &&
                    currentTime - lastLeadVehicleDetectedTime > leadVehicleLostGraceMs
                ) {
                    clearLeadVehicleTracking()
                }

                latestLeadVehicleStatus = "LVSA: no lead"
            }
        }
    }

    private fun applyDebugMode() {
        val debugVisibility = if (debugMode) View.VISIBLE else View.GONE

        btnTestFCW.visibility = debugVisibility
        btnTestLDW.visibility = debugVisibility
        btnTestDMS.visibility = debugVisibility
        fpsText.visibility = debugVisibility

        btnStatusMode.text = if (debugMode) {
            "Status ON"
        } else {
            "Status OFF"
        }

        statusText.text = if (debugMode) {
            "ADAS Status Mode"
        } else {
            "ADAS Active"
        }

        if (debugMode) {
            updateDualStatusText()
        }
    }

    private fun applyPerformanceMode() {
        if (ecoMode) {
            fcwFrameInterval = 12
            ldwFrameInterval = 5
            dmsFrameInterval = 10
            btnPerformanceMode.text = "Mode: Eco"
            Toast.makeText(this, "Eco Mode enabled", Toast.LENGTH_SHORT).show()
        } else {
            fcwFrameInterval = 10
            ldwFrameInterval = 3
            dmsFrameInterval = 8
            btnPerformanceMode.text = "Mode: Normal"
            Toast.makeText(this, "Normal Mode enabled", Toast.LENGTH_SHORT).show()
        }

        updateDualStatusText()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        yoloDetector = YoloDetector(this)

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_SHORT).show()
        }

        previewView = findViewById(R.id.previewView)

        frontPreviewView = findViewById(R.id.frontPreviewView)

        frontPreviewView.visibility = View.VISIBLE

        warningText = findViewById(R.id.warningText)

        safetyOverlay = findViewById(R.id.safetyOverlay)

        safetyOverlay.setRoadOverlayVisible(useBackCamera)

        fpsText = findViewById(R.id.fpsText)

        cameraExecutor = Executors.newSingleThreadExecutor()

        statusText = findViewById(R.id.statusText)

        btnTestFCW = findViewById(R.id.btnTestFCW)
        btnTestLDW = findViewById(R.id.btnTestLDW)
        btnTestDMS = findViewById(R.id.btnTestDMS)
        btnStatusMode = findViewById(R.id.btnStatusMode)
        btnSettings = findViewById(R.id.btnSettings)
        settingsPanel = findViewById(R.id.settingsPanel)
        btnPerformanceMode = findViewById(R.id.btnPerformanceMode)

        settingsPanel.visibility = View.GONE

        btnSettings.setOnClickListener {
            settingsPanel.visibility =
                if (settingsPanel.visibility == View.VISIBLE) {
                    View.GONE
                } else {
                    View.VISIBLE
                }
        }

        btnTestFCW.setOnClickListener {
            safetyOverlay.setMode("FCW")
            showWarning("BRAKE!", Color.parseColor("#CCFF0000"))
        }

        btnTestLDW.setOnClickListener {
            safetyOverlay.setMode("LDW")
            showWarning("LANE WARNING", Color.parseColor("#CCFFA500"))
        }

        btnTestDMS.setOnClickListener {
            safetyOverlay.setMode("DMS")
            showWarning("DROWSINESS WARNING", Color.parseColor("#CCFF0000"))
        }

        btnStatusMode.setOnClickListener {
            debugMode = !debugMode
            applyDebugMode()
        }

        btnPerformanceMode.setOnClickListener {
            ecoMode = !ecoMode
            applyPerformanceMode()
        }

        applyDebugMode()
        applyPerformanceMode()

        switchCameraButton = findViewById(R.id.btnSwitchCamera)

        switchCameraButton.setOnClickListener {
            if (isDualCameraSupported) {
                Toast.makeText(this, "Dual camera mode is active", Toast.LENGTH_SHORT).show()
            } else {
                useBackCamera = !useBackCamera

                safetyOverlay.setMode("NORMAL")
                safetyOverlay.clearFcwBox()
                safetyOverlay.setRoadOverlayVisible(useBackCamera)
                resetWarningStates()

                if (useBackCamera) {
                    latestLaneStatus = "Lane: -"
                    latestFcwStatus = "FCW: -"
                    latestDmsStatus = "DMS: inactive"
                    Toast.makeText(this, "Rear camera mode", Toast.LENGTH_SHORT).show()
                } else {
                    latestLaneStatus = "Lane: inactive"
                    latestFcwStatus = "FCW: inactive"
                    latestDmsStatus = "DMS: -"
                    Toast.makeText(this, "Front camera mode", Toast.LENGTH_SHORT).show()
                }

                startCamera()
            }
        }

        if (hasCameraPermission()) {
            startDualCamera()
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        updateFps(imageProxy)
                    }
                }

            val cameraSelector = if (useBackCamera) {
                CameraSelector.DEFAULT_BACK_CAMERA
            } else {
                CameraSelector.DEFAULT_FRONT_CAMERA
            }

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )

                frontPreviewView.visibility = View.GONE
                switchCameraButton.visibility = View.VISIBLE
                safetyOverlay.setRoadOverlayVisible(useBackCamera)

            } catch (e: Exception) {
                Toast.makeText(this, "Camera failed to start", Toast.LENGTH_SHORT).show()
                e.printStackTrace()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun startDualCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            var frontCameraSelector: CameraSelector? = null
            var backCameraSelector: CameraSelector? = null

            for (cameraInfos in cameraProvider.availableConcurrentCameraInfos) {
                val frontCameraInfo = cameraInfos.firstOrNull {
                    it.lensFacing == CameraSelector.LENS_FACING_FRONT
                }

                val backCameraInfo = cameraInfos.firstOrNull {
                    it.lensFacing == CameraSelector.LENS_FACING_BACK
                }

                if (frontCameraInfo != null && backCameraInfo != null) {
                    frontCameraSelector = frontCameraInfo.getCameraSelector()
                    backCameraSelector = backCameraInfo.getCameraSelector()
                    break
                }
            }

            if (frontCameraSelector == null || backCameraSelector == null) {
                isDualCameraSupported = false

                frontPreviewView.visibility = View.GONE
                switchCameraButton.visibility = View.VISIBLE

                Toast.makeText(
                    this,
                    "Dual camera not supported. Switch mode enabled.",
                    Toast.LENGTH_LONG
                ).show()

                startCamera()
                return@addListener
            }


            val backPreview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            val frontPreview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(frontPreviewView.surfaceProvider)
                }
            val backAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        analyzeBackCamera(imageProxy)
                    }
                }

            val frontAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(320, 240))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        analyzeFrontCamera(imageProxy)
                    }
                }

            val backUseCaseGroup = UseCaseGroup.Builder()
                .addUseCase(backPreview)
                .addUseCase(backAnalysis)
                .build()

            val frontUseCaseGroup = UseCaseGroup.Builder()
                .addUseCase(frontPreview)
                .addUseCase(frontAnalysis)
                .build()

            val backCameraConfig = SingleCameraConfig(
                backCameraSelector,
                backUseCaseGroup,
                this
            )

            val frontCameraConfig = SingleCameraConfig(
                frontCameraSelector,
                frontUseCaseGroup,
                this
            )

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    listOf(backCameraConfig, frontCameraConfig)
                )

                isDualCameraSupported = true

                frontPreviewView.visibility = View.VISIBLE
                switchCameraButton.visibility = View.GONE
                safetyOverlay.setRoadOverlayVisible(true)

                Toast.makeText(this, "Dual camera started", Toast.LENGTH_SHORT).show()

            } catch (e: Exception) {
                Toast.makeText(this, "Dual camera failed. Using switch mode.", Toast.LENGTH_LONG).show()
                e.printStackTrace()
                startCamera()
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun showWarning(message: String, backgroundColor: Int) {
        warningText.text = message
        warningText.setBackgroundColor(backgroundColor)
        warningText.visibility = View.VISIBLE

        toneGenerator.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 500)

        Handler(Looper.getMainLooper()).postDelayed({
            warningText.visibility = View.GONE
            safetyOverlay.setMode("NORMAL")
        }, 2000)
    }

    private fun updateFps(imageProxy: ImageProxy) {
        if (useBackCamera) {
            analyzeBackCamera(imageProxy)
        } else {
            analyzeFrontCamera(imageProxy)
        }
    }

    private fun analyzeBackCamera(imageProxy: ImageProxy) {
        backFrameCount++
        backAnalysisFrameIndex++

        val currentTime = System.currentTimeMillis()
        val elapsedTime = currentTime - lastBackFpsTime

        if (elapsedTime >= 1000) {
            latestBackFps = backFrameCount
            backFrameCount = 0
            lastBackFpsTime = currentTime

            runOnUiThread {
                updateDualStatusText()
            }
        }

        val shouldRunFcw =
            !isFcwProcessing && backAnalysisFrameIndex % fcwFrameInterval  == 0

        if (shouldRunFcw) {
            isFcwProcessing = true

            yoloDetector.detect(imageProxy) { result ->
                isFcwProcessing = false

                latestFcwStatus =
                    "YOLO ${result.className ?: "-"} " +
                            "Conf:${String.format("%.2f", result.confidence)} " +
                            "Area:${String.format("%.4f", result.areaRatio)} " +
                            "Box:${result.boundingBox?.width() ?: 0}x${result.boundingBox?.height() ?: 0} " +
                            "Warn:${result.collisionWarning}"

                processLeadVehicleStartAlert(result)

                runOnUiThread {
                    safetyOverlay.setFcwBox(
                        result.boundingBox,
                        result.imageWidth,
                        result.imageHeight
                    )
                    updateDualStatusText()
                }

                if (result.collisionWarning) {
                    fcwWarningCounter++
                    fcwSafeCounter = 0
                } else {
                    fcwWarningCounter = 0
                    fcwSafeCounter++

                    if (fcwSafeCounter >= fcwSafeThreshold) {
                        fcwDangerActive = false
                        fcwSafeCounter = 0
                    }
                }

                val currentTimeForWarning = System.currentTimeMillis()

                if (
                    fcwWarningCounter >= fcwWarningThreshold &&
                    !fcwDangerActive &&
                    currentTimeForWarning - lastFcwWarningTime >= fcwCooldownMs
                ) {
                    lastFcwWarningTime = currentTimeForWarning
                    fcwWarningCounter = 0
                    fcwDangerActive = true

                    runOnUiThread {
                        safetyOverlay.setMode("FCW")
                        showWarning("BRAKE!", Color.parseColor("#CCFF0000"))
                    }
                }
            }

            return
        }

        if (backAnalysisFrameIndex % ldwFrameInterval   == 0) {
            val result = laneDetector.detect(imageProxy)

            latestLaneStatus =
                "Lane L:${result.leftDetected} R:${result.rightDetected} " +
                        "Lines:${result.lineCount} " +
                        "Offset:${String.format("%.2f", result.offsetRatio)} " +
                        "Warn:${result.laneWarning}"

            val possibleLaneIssue =
                result.leftDetected &&
                        result.rightDetected &&
                        result.laneWarning

            val currentTimeForWarning = System.currentTimeMillis()

            if (possibleLaneIssue) {
                lastLaneIssueDetectedTime = currentTimeForWarning

                if (laneDepartureStartTime == null) {
                    laneDepartureStartTime = currentTimeForWarning
                }
            }

            val laneIssueStillValid =
                laneDepartureStartTime != null &&
                        currentTimeForWarning - lastLaneIssueDetectedTime <= laneIssueGraceMs

            if (laneIssueStillValid) {
                val departureDuration =
                    currentTimeForWarning - (laneDepartureStartTime ?: currentTimeForWarning)

                latestLaneStatus =
                    "Lane L:${result.leftDetected} R:${result.rightDetected} " +
                            "Lines:${result.lineCount} " +
                            "Offset:${String.format("%.2f", result.offsetRatio)} " +
                            "LDW:${departureDuration / 1000.0}s"

                if (
                    departureDuration >= laneDepartureDurationMs &&
                    currentTimeForWarning - lastLaneWarningTime >= laneWarningCooldownMs
                ) {
                    lastLaneWarningTime = currentTimeForWarning
                    laneDepartureStartTime = null

                    runOnUiThread {
                        safetyOverlay.setMode("LDW")
                        showWarning("LANE WARNING", Color.parseColor("#CCFFA500"))
                    }
                }
            } else {
                laneDepartureStartTime = null
            }
        }

        imageProxy.close()
    }

    private fun analyzeFrontCamera(imageProxy: ImageProxy) {
        frontFrameCount++
        frontAnalysisFrameIndex++

        val currentTime = System.currentTimeMillis()
        val elapsedTime = currentTime - lastFrontFpsTime

        if (elapsedTime >= 1000) {
            latestFrontFps = frontFrameCount
            frontFrameCount = 0
            lastFrontFpsTime = currentTime

            runOnUiThread {
                updateDualStatusText()
            }
        }

        if (!isDmsProcessing && frontAnalysisFrameIndex % dmsFrameInterval == 0) {
            isDmsProcessing = true

            drowsinessDetector.detect(imageProxy) { result ->
                isDmsProcessing = false

                latestDmsStatus = if (result.faceDetected) {
                    "DMS Face:true L:${result.leftEyeOpen} R:${result.rightEyeOpen}"
                } else {
                    "DMS Face:false"
                }

                runOnUiThread {
                    updateDualStatusText()
                }

                val currentTimeForWarning = System.currentTimeMillis()

                if (result.drowsy) {
                    if (eyeClosedStartTime == null) {
                        eyeClosedStartTime = currentTimeForWarning
                    }

                    val closedDuration =
                        currentTimeForWarning - (eyeClosedStartTime ?: currentTimeForWarning)

                    if (
                        closedDuration >= drowsyDurationMs &&
                        currentTimeForWarning - lastDrowsyWarningTime >= drowsyCooldownMs
                    ) {
                        lastDrowsyWarningTime = currentTimeForWarning
                        eyeClosedStartTime = null

                        runOnUiThread {
                            safetyOverlay.setMode("DMS")
                            showWarning("DROWSINESS WARNING", Color.parseColor("#CCFF0000"))
                        }
                    }

                } else {
                    eyeClosedStartTime = null
                }
            }

        } else {
            imageProxy.close()
        }
    }

    private fun updateDualStatusText() {
        if (!debugMode) {
            return
        }

        val performanceText = if (ecoMode) {
            "Eco Mode"
        } else {
            "Normal Mode"
        }

        fpsText.text =
            "System Status\n" +
                    "$performanceText\n" +
                    "Back FPS: $latestBackFps\n" +
                    "Front FPS: $latestFrontFps\n" +
                    "$latestLaneStatus\n" +
                    "$latestFcwStatus\n" +
                    "$latestLeadVehicleStatus\n" +
                    "$latestDmsStatus"
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        toneGenerator.release()
        drowsinessDetector.close()
        yoloDetector.close()
    }
}