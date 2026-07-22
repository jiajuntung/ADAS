package com.example.adas_fyp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Rect
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.core.ConcurrentCamera.SingleCameraConfig
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.sqrt

class MainActivity : ComponentActivity(), SensorEventListener {

    private lateinit var previewView: PreviewView
    private lateinit var frontPreviewView: PreviewView
    private lateinit var warningText: TextView
    private lateinit var safetyOverlay: SafetyOverlayView
    private lateinit var fpsText: TextView
    private lateinit var cameraExecutor: ExecutorService
    private val warningSound = WarningSoundPlayer()

    // ------------------------------------------------------------------
    // Vehicle motion state (vision + accelerometer, no GPS)
    //
    // The accelerometer alone CANNOT tell "constant speed" from "stopped":
    // linear acceleration is ~0 in both cases. In the 13 July road test this
    // single confusion caused:
    //   - FCW suppressed while creeping up behind the lead car
    //     ("Stopped:true Danger:false" with YoloWarn:true),
    //   - LDW suppressed for large parts of normal driving,
    //   - LVSA still armed after the driver had already started moving.
    //
    // Fix: EgoMotionEstimator watches the road texture on the back camera.
    // Moving road => image flows => high score. Stationary => score near
    // sensor noise, even with engine vibration. The accelerometer is kept
    // as a fast secondary signal for accelerating / braking / turning.
    // ------------------------------------------------------------------
    private val egoMotionEstimator = EgoMotionEstimator()
    private var latestEgoScore = -1f
    private var visionMovingStreak = 0
    private var visionStaticStreak = 0

    // CALIBRATE on-device using the "Ego:" value in the status panel:
    // drive slowly and note the score (should sit clearly above egoMovingScore),
    // then stop with the engine running (score should sit below egoStaticScore).
    private val egoMovingScore = 3.0f
    private val egoStaticScore = 1.5f
    private val visionMovingFramesRequired = 2
    private val visionStaticFramesRequired = 6
    // Sustained visual stillness overrides accelerometer jitter (engine idle
    // vibration must not keep resetting the "stopped" state like it did in
    // the road test, where LVSA lost its reference mid-wait).
    private val visionStaticOverrideFrames = 15

    private val MOTION_UNKNOWN = 0
    private val MOTION_MOVING = 1
    private val MOTION_STATIONARY = 2
    private var vehicleMotionState = MOTION_UNKNOWN
    private var stationarySince = 0L
    private var gpsWarned = false

    // LVSA disarm debounce: only stand LVSA down once we are CONFIRMED moving.
    private var lvsaMovingSince = 0L
    private val lvsaDisarmSpeedMps = 1.4f       // ~5 km/h, clearly rolling
    private val lvsaDisarmEgoScore = 5.0f       // vision fallback when no GPS
    private val lvsaDisarmHoldMs = 600L         // sustained, not a flicker
    private val lvsaStoppedSpeedMps = 1.0f      // ~3.6 km/h counts as stopped

    // Own vehicle motion detection (accelerometer part)
    private lateinit var sensorManager: SensorManager
    private var motionSensor: Sensor? = null
    private val gravity = FloatArray(3)

    private var lastOwnMotionTime = 0L
    private var latestOwnMotionStatus = "Motion: -"

    private var ownMotionHitCount = 0
    private val ownVehicleMotionHitThreshold = 4

    private val ownVehicleMotionSuppressMs = 2500L
    private val ownVehicleStationaryRequiredMs = 1000L
    private val ownVehicleAccelThreshold = 0.60f

    // LDW
    private val laneDetector = LaneDetector()
    private var latestLaneStatus = "Lane: -"
    private var lastLaneWarningTime = 0L
    private val laneWarningCooldownMs = 5000L
    private var laneDepartureStartTime: Long? = null
    private val laneDepartureDurationMs = 450L
    private var lastLaneIssueDetectedTime = 0L
    private val laneIssueGraceMs = 1000L

    // Evidence persistence: the departure condition must be SEEN on at least
    // this many LDW frames inside the grace window before a warning fires.
    // Without this, one noisy frame armed the timer and the grace period
    // alone carried it past laneDepartureDurationMs -- the 14 July false
    // alarms (offset spike / seam misread / shadow clutter) all fired off
    // 1-3 noisy frames exactly this way ("Offset:0.00 ... LDW:0.278s").
    private var laneWarnHitCount = 0
    // Lowered from 3: with the smoothed, rate-limited p a real drift only spends
    // a few frames above the departure line before the driver corrects, and
    // one-line-dropout frames break a long streak. 2 confirmed frames is enough
    // evidence while still rejecting single-frame noise.
    private val laneWarnHitsRequired = 2

    // LDW low-speed cutoff. Production cars use ~60 km/h because their LDW
    // actively steers; a WARNING-ONLY system like this should stay useful in
    // slow traffic. On the 19 Jul drive (max 39 km/h, 83% below 20 km/h) the
    // old 20 km/h floor disabled LDW for almost the whole trip, so low-speed
    // lane crossings never warned. Lowered to ~8 km/h: below that the driver
    // is parking/manoeuvring and lane-keeping is meaningless, but ordinary
    // slow city driving is covered.
    private val ldwMinSpeedMps = 2.2f

    // DMS
    private val drowsinessDetector = DrowsinessDetector()
    private var lastDrowsyWarningTime = 0L
    private var isDmsProcessing = false
    private var latestDmsStatus = "DMS: -"
    private val drowsyCooldownMs = 5000L
    private var eyeClosedStartTime: Long? = null
    private val drowsyDurationMs = 2000L

    // ------------------------------------------------------------------
    // Physical sensing: GPS speed, gyro yaw rate, monocular distance.
    //
    // These replace the guesswork that produced the 16 Jul false BRAKEs.
    // FCW no longer fires on "the box looks big" (Area >= 0.040 fired on
    // cars 20-30 m away at Area:0.056 / 0.072). It fires on metres and
    // seconds: how far is it, and how long until we get there.
    // ------------------------------------------------------------------
    private lateinit var speedProvider: SpeedProvider
    private lateinit var distanceEstimator: DistanceEstimator
    private val stripCalibrator = RoadStripCalibrator()

    // Cornering: a parked or oncoming car sweeping through frame centre
    // mid-turn is in front of the CAMERA, not in our PATH. This was the
    // single most common false BRAKE. Yaw rate makes it explicit.
    private val turnYawRateRadS = 0.25f
    private val turnHoldMs = 700L
    private var lastTurningTime = 0L

    // FCW
    private lateinit var yoloDetector: YoloDetector
    private var isFcwProcessing = false
    private var latestFcwStatus = "FCW: -"
    private var fcwWarningCounter = 0
    private var lastFcwWarningTime = 0L
    private var fcwDangerActive = false
    // Require the SAME object to be dangerous on 2 consecutive FCW frames.
    // In the road test a parked car sweeping through the frame centre during
    // a turn fired BRAKE off a single frame; persistence + identity kills
    // most of those while adding only ~0.3-0.5 s of latency.
    private val fcwWarningThreshold = 2
    private val fcwCooldownMs = 3500L

    // --- Physics-based FCW thresholds (metres / seconds) -------------------
    // BRAKE when the gap is about to close: either time-to-collision is
    // short, or headway is so small that any lead braking is unsurvivable.
    private val brakeTtcSec = 2.0f

    // Pure-headway BRAKE (no closing required) is the "you are simply far too
    // close" case. 0.6 s at 80 km/h is a 13 m gap; Malaysian highway traffic
    // sits nearer than that routinely, and firing on it constantly is what
    // made highway FCW feel trigger-happy. Tightened, and now it must persist
    // (fcwWarningThreshold frames on the SAME object) before it speaks.
    private val brakeHeadwaySec = 0.45f
    // BRAKE speed floor. 11 km/h left BRAKE dead for 50% of the slow 19 Jul
    // drive. A low-speed rear-end (creeping into the car ahead in traffic) is
    // still worth a warning, so this is lowered to ~4 km/h. Below that,
    // closing is so slow that TTC/headway logic handles it without a hard
    // BRAKE.
    private val brakeMinSpeedMps = 1.1f      // ~4 km/h
    private val brakeMaxDistanceM = 40f      // sanity clamp on the estimate

    // TTC-based BRAKE only when the gap is genuinely small; a noisy low TTC at
    // 25 m is not an emergency. And the TTC danger must persist across a
    // majority of a short voting window, so a single jittery dip can't fire.
    private val brakeTtcMaxDistanceM = 15f
    private val brakeTtcVoteFraction = 0.6f
    private val ttcVote = BooleanArray(5)
    private var ttcVoteIdx = 0
    private var ttcVoteFilled = 0

    // Ego-lane test, in METRES. A lane is ~3.5 m wide, so anything more than
    // ~1.7 m off the camera axis is in another lane. This replaces the image-x
    // window that made buses beside us trigger BRAKE: at 30 m an adjacent-lane
    // vehicle projects to image x ~0.59, inside any "centre of frame" gate,
    // but its true lateral offset is 3.4 m at every range.
    // Ego-lane half-widths, in metres. A lane is ~3.5 m, so its half is ~1.75 m.
    // BRAKE keeps the safety-margin width (a car drifting in from the side is
    // still a hazard). TOO CLOSE -- a softer reminder -- uses a TIGHTER width so
    // it only fires on a car genuinely ahead: on the 21 Jul drive it kept
    // warning on cars 1.1-1.6 m to the side (the next lane's near edge).
    private val egoLaneHalfWidthM = 1.7f          // BRAKE / lead selection
    private val proxLaneHalfWidthM = 1.05f        // TOO CLOSE: centred only

    // If the phone is not mounted on the car's centreline, set this to the
    // offset in metres (positive = mounted right of centre). Malaysia RHD with
    // a centre-windscreen cradle is ~0.
    private val cameraLateralOffsetM = 0.0f
    private var latestLateralM = Float.NaN
    private var smoothedLateralM = Float.NaN
    private val lateralEma = 0.5f

    // TOO CLOSE: the familiar two-second rule, as a gentle heads-up well
    // before BRAKE. This is the "warn me a bit earlier" tier.
    private val proxHeadwaySec = 1.5f

    // TOO CLOSE speed floor. The old 40 km/h floor made this tier dead for the
    // ENTIRE 19 Jul drive (100% of samples below 40), which is exactly the
    // "slow, very close to the car ahead, no TOO CLOSE" report. The headway
    // rule already scales with speed, so the floor only needs to exclude
    // crawling/parking. Lowered to ~8 km/h. To avoid nagging in dense stop-go
    // traffic, a SHORT ABSOLUTE GAP is additionally required at low speed
    // (below proxLowSpeedMps) via proxLowSpeedMaxDistanceM.
    private val proxMinSpeedMps = 8.3f
    private val proxLowSpeedMps = 8.0f          // ~29 km/h
    private val proxLowSpeedMaxDistanceM = 8f   // at crawl, only warn if really close

    // Distance-only fallback for when GPS speed is unavailable: warn if a lead
    // is within this range AND the gap is actively closing.
    private val proxFallbackDistanceM = 12f
    private val proxFallbackClosingMps = 1.2f

    // --- Target identity + robust closing speed ----------------------------
    //
    // TWO FAULTS CAUSED THE 17 JUL FALSE BRAKES.
    //
    // 1. NO IDENTITY. selectInLaneCandidate() ran independently every frame,
    //    so with several candidates in view the "closest in-lane" object could
    //    be a different vehicle each frame. Logged at 35-36s:
    //        D:12.3m -> D:34.2m -> D:3.0m  in one second -> TTC:0.1s -> BRAKE
    //    That is not motion, it is the target teleporting between objects.
    //
    // 2. DIFFERENTIATING NOISE. Distance comes from box width, so its absolute
    //    error grows with range: +/-2 px of box jitter is +/-0.2 m at 10 m but
    //    +/-1.4 m at 25 m. Frame-differencing that over 0.25 s invents ~11 m/s
    //    of closing speed from nothing -- which is precisely why FCW felt
    //    twitchy on the highway and calm in town.
    //
    // Fixes: lock the target by IoU and drop the history when it changes; fit
    // a least-squares line to the last ~2 s of (t, distance) instead of
    // differencing consecutive frames; and reject closing speeds that are
    // physically impossible.
    private var fcwTargetBox: Rect? = null
    private val fcwHistTime = LongArray(10)
    private val fcwHistDist = FloatArray(10)
    private var fcwHistCount = 0
    private val fcwHistMinSamples = 4
    private val fcwHistMaxAgeMs = 2500L
    private val fcwTargetSameIou = 0.25f

    // A lead vehicle cannot approach faster than we are driving at it (it would
    // have to be reversing). Anything beyond ego speed + this margin is noise.
    private val closingSanityMarginMps = 3.0f

    private var fcwSmoothClosingMps = 0f
    private var latestDistanceM = -1f
    private var latestHeadwaySec = -1f
    private var latestTtcSec = -1f
    private var fcwSafeCounter = 0
    private val fcwSafeThreshold = 8
    private var fcwLastWarnedAreaRatio = 0f
    private val fcwRewarnAreaIncreaseRatio = 1.45f
    private var fcwLastDangerBox: Rect? = null

    // Early proximity reminder ("TOO CLOSE"): a soft heads-up that fires
    // BEFORE the BRAKE thresholds while closing in on the lead vehicle.
    // Requested after the 15 Jul test: BRAKE alone felt slightly late as a
    // first notification.
    private val proxPersistFrames = 2
    private val proxCooldownMs = 8000L
    private val proxSuppressAfterBrakeMs = 3000L
    private var proxCounter = 0
    private var proxLastBox: Rect? = null
    private var lastProxWarningTime = 0L

    // Minimum IoU for two frames' boxes to count as the same object. Used by
    // the FCW/proximity persistence checks so a car sweeping through frame
    // centre during a turn cannot accumulate consecutive "danger" frames.
    private val fcwSameObjectIou = 0.2f

    // NOTE: the old box-height TTC tracker lived here. It inferred closing
    // speed from how fast the box grew, which conflated distance with object
    // size and needed magic constants. It is replaced by updateDistanceTtc(),
    // which differentiates a real distance in metres.

    // Camera
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

    // UI settings
    private lateinit var statusText: TextView
    private lateinit var btnTestFCW: Button
    private lateinit var btnTestLDW: Button
    private lateinit var btnTestDMS: Button
    private lateinit var btnStatusMode: Button
    private lateinit var btnPerformanceMode: Button
    private lateinit var btnDriverSide: Button
    private lateinit var btnSettings: Button
    private lateinit var settingsPanel: View

    private var debugMode = true
    private var ecoMode = false

    private var fcwFrameInterval = 5
    private var ldwFrameInterval = 3
    private var dmsFrameInterval = 8

    // LVSA - Lead Vehicle Start Alert
    private var leadVehicleSeenStartTime: Long? = null
    private var leadVehicleWaiting = false
    private var leadVehicleReferenceArea = 0f
    private var lastLeadVehicleArea = 0f
    private var lastLeadVehicleDetectedTime = 0L
    private var lastLeadVehicleAlertTime = 0L
    private var latestLeadVehicleStatus = "LVSA: -"

    // A real "front car pulled away" shows up as the box shrinking over
    // several frames, not as a single jittery small frame or a lost
    // detection.
    private var leadShrinkStreak = 0
    private var leadWasShrinkingBeforeLost = false

    // Identity tracking: the YOLO stage returns only the single best object,
    // and in traffic that can momentarily switch to a different vehicle
    // (adjacent lane). Without an identity check the switch corrupts the
    // reference area and the shrink streak.
    private var lastLeadBox: Rect? = null
    private var leadIdentityMissCount = 0
    private val leadSameObjectIou = 0.15f
    private val leadIdentityMissLimit = 3

    // COCO class ids: car, motorcycle, bus, truck
    private val leadVehicleClassIds = setOf(2, 3, 5, 7)

    private val leadVehicleStableMs = 1200L
    private val leadVehicleMovedDropRatio = 0.12f
    private val leadVehicleLostGraceMs = 900L
    private val leadVehicleLostMaxMs = 3000L
    private val leadVehicleAlertCooldownMs = 10000L
    private val minLeadVehicleAreaRatio = 0.025f
    private val leadShrinkStreakRequired = 2

    // Instant motion veto: at the very moment an LVSA alert is about to
    // fire, any fresh evidence that the ego car is already rolling cancels
    // it. The 15 Jul test showed a photo-finish race (217.5s: stopped
    // Ego:1.0 -> 218.0s: moving Ego:4.0 AND "LVSA: moved" in the same
    // instant): the state machine needs 2 vision frames to flip to MOVING,
    // but the alert decision happens inside that window. The veto reads the
    // freshest single-frame signals instead of waiting for the state.
    private val lvsaVetoEgoScore = 2.5f
    private val lvsaVetoAccelMs = 500L

    private var currentWarningPriority = 0
    private var currentWarningEndTime = 0L
    private var warningToken = 0
    private var latestFcwDangerNow = false
    private var lastFcwDangerTime = 0L
    private val fcwDangerHoldMs = 3000L

    private val PRIORITY_LVSA = 1
    private val PRIORITY_LDW = 2
    private val PRIORITY_PROXIMITY = 3
    private val PRIORITY_DMS = 4
    private val PRIORITY_FCW = 5

    private val requestPermissions =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { result ->
            if (result[Manifest.permission.CAMERA] == true) {
                startDualCamera()
            } else {
                Toast.makeText(this, "Camera permission is required", Toast.LENGTH_SHORT).show()
            }

            if (result[Manifest.permission.ACCESS_FINE_LOCATION] == true) {
                speedProvider.start()
            } else {
                // No GPS -> vision ego-motion still drives the motion state,
                // but FCW cannot compute headway and falls back to a
                // conservative distance-only rule.
                Toast.makeText(
                    this,
                    "Location denied: FCW timing will be less accurate",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // IMPORTANT: keep this before findViewById()
        setContentView(R.layout.activity_main)

        initOwnVehicleMotionSensor()

        yoloDetector = YoloDetector(this)
        speedProvider = SpeedProvider(this)
        distanceEstimator = DistanceEstimator(this)

        if (!OpenCVLoader.initDebug()) {
            Toast.makeText(this, "OpenCV failed to load", Toast.LENGTH_SHORT).show()
        }

        previewView = findViewById(R.id.previewView)

        // Show the WHOLE analysis frame instead of centre-cropping it.
        //
        // The default FILL_CENTER crops the top and bottom 20% of a 4:3 camera
        // image in this 2.2:1 landscape view. Everything the detectors do down
        // there is then invisible in the preview -- which is exactly how the
        // LDW strip ended up analysing the car's own bonnet for three road
        // tests with no way to see it. FIT_CENTER letterboxes instead, so the
        // preview and the detectors look at the same pixels and the debug
        // overlay lines up 1:1.
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER
        frontPreviewView = findViewById(R.id.frontPreviewView)
        warningText = findViewById(R.id.warningText)
        safetyOverlay = findViewById(R.id.safetyOverlay)
        fpsText = findViewById(R.id.fpsText)
        statusText = findViewById(R.id.statusText)

        btnTestFCW = findViewById(R.id.btnTestFCW)
        btnTestLDW = findViewById(R.id.btnTestLDW)
        btnTestDMS = findViewById(R.id.btnTestDMS)
        btnStatusMode = findViewById(R.id.btnStatusMode)
        btnPerformanceMode = findViewById(R.id.btnPerformanceMode)
        btnDriverSide = findViewById(R.id.btnDriverSide)
        btnSettings = findViewById(R.id.btnSettings)
        settingsPanel = findViewById(R.id.settingsPanel)
        switchCameraButton = findViewById(R.id.btnSwitchCamera)

        frontPreviewView.visibility = View.VISIBLE
        settingsPanel.visibility = View.GONE
        safetyOverlay.setRoadOverlayVisible(useBackCamera)

        cameraExecutor = Executors.newSingleThreadExecutor()

        btnSettings.setOnClickListener {
            settingsPanel.visibility =
                if (settingsPanel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        btnTestFCW.setOnClickListener {
            showAdasWarning(
                message = "BRAKE!",
                backgroundColor = Color.parseColor("#CCFF0000"),
                overlayMode = "FCW",
                priority = PRIORITY_FCW,
                durationMs = 2500L
            )
        }

        btnTestLDW.setOnClickListener {
            showAdasWarning(
                message = "LANE WARNING",
                backgroundColor = Color.parseColor("#CCFFA500"),
                overlayMode = "LDW",
                priority = PRIORITY_LDW
            )
        }

        btnTestDMS.setOnClickListener {
            showAdasWarning(
                message = "DROWSINESS WARNING",
                backgroundColor = Color.parseColor("#CCFF0000"),
                overlayMode = "DMS",
                priority = PRIORITY_DMS
            )
        }

        btnStatusMode.setOnClickListener {
            debugMode = !debugMode
            applyDebugMode()
        }

        btnPerformanceMode.setOnClickListener {
            ecoMode = !ecoMode
            applyPerformanceMode()
        }

        // Malaysia is right-hand drive, so the driver sits on the car's RIGHT
        // -- but which side of the ML Kit image that becomes depends on front
        // camera mirroring and rotation, which vary by device. The status
        // panel reports the chosen face's x; if DMS locks onto the passenger,
        // flip this once and it is correct for good.
        btnDriverSide.setOnClickListener {
            drowsinessDetector.driverOnRight = !drowsinessDetector.driverOnRight
            eyeClosedStartTime = null
            applyDriverSideLabel()
            Toast.makeText(
                this,
                if (drowsinessDetector.driverOnRight) "Driver: RIGHT of front image"
                else "Driver: LEFT of front image",
                Toast.LENGTH_SHORT
            ).show()
        }

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

        applyDebugMode()
        applyPerformanceMode()
        applyDriverSideLabel()

        if (hasCameraPermission()) {
            startDualCamera()
            if (hasLocationPermission()) speedProvider.start()
        }

        val missing = mutableListOf<String>()
        if (!hasCameraPermission()) missing.add(Manifest.permission.CAMERA)
        if (!hasLocationPermission()) missing.add(Manifest.permission.ACCESS_FINE_LOCATION)
        if (missing.isNotEmpty()) requestPermissions.launch(missing.toTypedArray())
    }

    private fun initOwnVehicleMotionSensor() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        motionSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
            ?: sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        lastOwnMotionTime = System.currentTimeMillis()

        latestOwnMotionStatus = if (motionSensor == null) {
            "Motion: no accel (vision only)"
        } else {
            "Motion: settling"
        }
    }

    override fun onResume() {
        super.onResume()

        if (::sensorManager.isInitialized) {
            motionSensor?.let {
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
            }
        }

        if (::speedProvider.isInitialized && hasLocationPermission()) {
            speedProvider.start()
        }
    }

    override fun onPause() {
        super.onPause()

        if (::sensorManager.isInitialized) {
            sensorManager.unregisterListener(this)
        }

        if (::speedProvider.isInitialized) {
            speedProvider.stop()
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val now = System.currentTimeMillis()

        val accelMagnitude = if (event.sensor.type == Sensor.TYPE_LINEAR_ACCELERATION) {
            val x = event.values[0]
            val y = event.values[1]
            val z = event.values[2]
            sqrt(x * x + y * y + z * z)
        } else {
            // Fallback for normal accelerometer: remove gravity using low-pass filter.
            val alpha = 0.8f

            gravity[0] = alpha * gravity[0] + (1 - alpha) * event.values[0]
            gravity[1] = alpha * gravity[1] + (1 - alpha) * event.values[1]
            gravity[2] = alpha * gravity[2] + (1 - alpha) * event.values[2]

            val x = event.values[0] - gravity[0]
            val y = event.values[1] - gravity[1]
            val z = event.values[2] - gravity[2]

            sqrt(x * x + y * y + z * z)
        }

        // The accelerometer only feeds a timestamp into the motion state
        // machine. It must NOT clear LVSA or flip states on its own: engine
        // idle vibration produced spurious "moving" flips in the road test,
        // which wiped the LVSA reference mid-wait and caused missed alerts.
        if (accelMagnitude > ownVehicleAccelThreshold) {
            ownMotionHitCount++

            if (ownMotionHitCount >= ownVehicleMotionHitThreshold) {
                lastOwnMotionTime = now
            }
        } else {
            ownMotionHitCount = 0
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not required.
    }

    // ------------------------------------------------------------------
    // Motion state machine: fuse vision (primary) with accel (secondary).
    // Called on every analyzed back frame, right after the ego-motion score
    // is updated.
    //
    //   MOVING      -> FCW/LDW active, LVSA disarmed.
    //   STATIONARY  -> FCW/LDW suppressed, LVSA may arm.
    //   UNKNOWN     -> fail safe: FCW/LDW stay ACTIVE, LVSA stays disarmed.
    //
    // Note the fail-safe inversion: previously the driving warnings were OFF
    // unless the accelerometer proved motion, which disabled them for most
    // of the drive. Now they are ON unless the vision clearly proves the car
    // is stationary.
    // ------------------------------------------------------------------
    private fun updateVehicleMotionState(now: Long) {
        val accelRecentlyStrong = now - lastOwnMotionTime <= ownVehicleMotionSuppressMs
        val visionMoving = visionMovingStreak >= visionMovingFramesRequired
        val visionStatic = visionStaticStreak >= visionStaticFramesRequired
        val lowLight = egoMotionEstimator.isLowLight()

        // GPS answers "are we moving" directly and needs no calibration, so
        // it wins whenever the fix is fresh. Vision and the accelerometer
        // remain the fallback for tunnels / underground car parks.
        val gpsSpeed = speedProvider.speedMps()

        val newState = when {
            gpsSpeed >= 1.5f -> MOTION_MOVING
            gpsSpeed in 0f..0.4f && !visionMoving -> MOTION_STATIONARY

            visionMoving -> MOTION_MOVING

            accelRecentlyStrong && !visionStatic -> MOTION_MOVING

            !lowLight && visionStatic &&
                    (!accelRecentlyStrong || visionStaticStreak >= visionStaticOverrideFrames) ->
                MOTION_STATIONARY

            else -> MOTION_UNKNOWN
        }

        // Mark the moment we became stationary. Trust GPS first: at a light
        // GPS reads ~0 steadily even while the vision state flickers, so this
        // keeps a stable "stopped since" for LVSA instead of resetting on every
        // flicker.
        val gpsNow = speedProvider.speedMps()
        val stoppedNow =
            if (gpsNow >= 0f) gpsNow <= lvsaStoppedSpeedMps
            else newState == MOTION_STATIONARY

        if (stoppedNow) {
            if (stationarySince == 0L) stationarySince = now
        } else {
            stationarySince = 0L
        }

        // LVSA DISARM MUST BE DEBOUNCED.
        //
        // At a traffic-light stop the motion state flickers MOVING<->STATIONARY
        // every few seconds (GPS jitters 0<->a few km/h; vision ego-motion sees
        // the car ahead roll and heat shimmer). The old code disarmed LVSA on
        // the FIRST MOTION_MOVING sample, so on the 19 Jul drive LVSA was
        // constantly reset: when the front car actually left, LVSA had usually
        // just been wiped and never alerted; and stray 'waiting' windows caught
        // the front car creeping and alerted even as the driver also moved.
        //
        // Real departure is sustained. We only disarm when we are CONFIRMED
        // moving: GPS speed clearly above a walking pace, held for a short time.
        // A vision-only 'moving' flicker no longer disarms LVSA.
        val gpsForLvsa = speedProvider.speedMps()
        val confirmedMovingOff =
            gpsForLvsa >= lvsaDisarmSpeedMps ||
                    (gpsForLvsa < 0f && newState == MOTION_MOVING && latestEgoScore >= lvsaDisarmEgoScore)

        if (confirmedMovingOff) {
            if (lvsaMovingSince == 0L) lvsaMovingSince = now
        } else {
            lvsaMovingSince = 0L
        }

        val egoDepartureConfirmed =
            lvsaMovingSince != 0L && now - lvsaMovingSince >= lvsaDisarmHoldMs

        if (egoDepartureConfirmed && (leadVehicleSeenStartTime != null || leadVehicleWaiting)) {
            clearLeadVehicleTracking()
            latestLeadVehicleStatus = "LVSA: disabled / own moving"
        }

        vehicleMotionState = newState

        val egoText = if (latestEgoScore >= 0f) {
            String.format("%.1f", latestEgoScore)
        } else {
            "-"
        }

        val darkText = if (lowLight) " dark" else ""

        val gpsText = speedProvider.statusText()

        // Make "GPS off" impossible to miss: it silently disables the accurate
        // speed path and forces the coarse vision fallback.
        if (gpsText.contains("off") || gpsText.contains("noperm")) {
            if (!gpsWarned) {
                gpsWarned = true
                runOnUiThread {
                    Toast.makeText(
                        this,
                        "Location is OFF - turn on GPS for accurate FCW timing",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        } else {
            gpsWarned = false
        }

        latestOwnMotionStatus = when (newState) {
            MOTION_MOVING -> "Motion: moving Ego:$egoText $gpsText$darkText"
            MOTION_STATIONARY -> {
                val stoppedFor = if (stationarySince > 0L) (now - stationarySince) / 1000.0 else 0.0
                "Motion: stopped ${String.format("%.1f", stoppedFor)}s $gpsText$darkText"
            }
            else -> "Motion: settling Ego:$egoText $gpsText$darkText"
        }
    }

    /**
     * Time-to-collision from a robust fit to the recent distance history.
     *
     * The history is cleared whenever the tracked object changes identity, so
     * a target switch can no longer masquerade as 30 m/s of closing speed.
     * The closing rate is the negated slope of a least-squares fit over ~2 s,
     * which is far steadier than a frame-to-frame difference of a signal whose
     * noise grows with range.
     *
     * Returns 99 when not approaching or when there is not enough evidence.
     */
    private fun updateDistanceTtc(distanceM: Float, box: Rect?, egoSpeed: Float): Float {
        val now = System.currentTimeMillis()

        if (distanceM <= 0f || box == null) {
            fcwHistCount = 0
            fcwTargetBox = null
            fcwSmoothClosingMps = 0f
            return 99f
        }

        // Identity check: same object as last frame?
        val prevBox = fcwTargetBox
        val sameTarget = prevBox != null && iou(prevBox, box) >= fcwTargetSameIou
        fcwTargetBox = box

        if (!sameTarget) {
            fcwHistCount = 0
            fcwSmoothClosingMps = 0f
        }

        // Drop samples older than the window, then append.
        var w = 0
        for (i in 0 until fcwHistCount) {
            if (now - fcwHistTime[i] <= fcwHistMaxAgeMs) {
                fcwHistTime[w] = fcwHistTime[i]
                fcwHistDist[w] = fcwHistDist[i]
                w++
            }
        }
        fcwHistCount = w

        if (fcwHistCount == fcwHistTime.size) {
            for (i in 1 until fcwHistCount) {
                fcwHistTime[i - 1] = fcwHistTime[i]
                fcwHistDist[i - 1] = fcwHistDist[i]
            }
            fcwHistCount--
        }

        fcwHistTime[fcwHistCount] = now
        fcwHistDist[fcwHistCount] = distanceM
        fcwHistCount++

        if (fcwHistCount < fcwHistMinSamples) return 99f

        // Least-squares slope of distance vs time (m/s).
        val t0 = fcwHistTime[0]
        var sx = 0.0
        var sy = 0.0
        var sxx = 0.0
        var sxy = 0.0
        val n = fcwHistCount

        for (i in 0 until n) {
            val x = (fcwHistTime[i] - t0) / 1000.0
            val y = fcwHistDist[i].toDouble()
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
        }

        val denom = n * sxx - sx * sx
        if (denom < 1e-6) return 99f

        val slope = (n * sxy - sx * sy) / denom     // m/s, negative = closing
        val closing = (-slope).toFloat()

        // Physical sanity: the gap cannot shrink faster than we drive at it.
        // With a coarse vision speed the margin is widened so a valid approach
        // is not clipped by speed-estimate error.
        val margin = if (hasTrueSpeed()) closingSanityMarginMps else closingSanityMarginMps * 2f
        val maxClosing = (if (egoSpeed > 0f) egoSpeed else 0f) + margin
        if (closing > maxClosing) return 99f

        fcwSmoothClosingMps = closing

        // Require the closing rate to be meaningful relative to the noise: at
        // range, a fraction of a m/s is indistinguishable from box jitter.
        val minClosing = maxOf(0.8f, 0.06f * distanceM)

        return if (closing > minClosing) distanceM / closing else 99f
    }

    private fun isVehicleClearlyStationary(): Boolean {
        return vehicleMotionState == MOTION_STATIONARY
    }

    private fun isOwnVehicleStoppedEnough(): Boolean {
        // LVSA arms after the car has been stationary for a short while.
        //
        // The state must not flicker it off: at a light, GPS reading 0-2 km/h
        // is still "stopped" for LVSA purposes even if the motion state briefly
        // flips to MOVING on a vision glitch. So when GPS is available we trust
        // it directly and only require the car to be essentially still; the
        // debounced disarm (lvsaMovingSince) is what ends the wait, not this
        // instantaneous check.
        val gps = speedProvider.speedMps()

        if (gps in 0f..lvsaStoppedSpeedMps) {
            return System.currentTimeMillis() - stationarySince >= ownVehicleStationaryRequiredMs ||
                    stationarySince == 0L
        }

        if (gps > lvsaStoppedSpeedMps) return false

        // No GPS: fall back to the vision motion state.
        return vehicleMotionState == MOTION_STATIONARY &&
                System.currentTimeMillis() - stationarySince >= ownVehicleStationaryRequiredMs
    }

    // Last-instant check before an LVSA alert: if ANY fresh signal says we
    // are already rolling, stay quiet. Reads raw signals rather than the
    // motion state, because the state machine needs a couple of frames to
    // flip and the alert decision happens inside that window (15 Jul, 218s:
    // "stopped 70.7s" and "moving Ego:4.0" were 0.5 s apart, and the alert
    // landed in between).
    private fun lvsaInstantMotionVeto(): Boolean {
        val gps = speedProvider.speedMps()
        if (gps >= 0.8f) return true

        val accelJustNow =
            System.currentTimeMillis() - lastOwnMotionTime <= lvsaVetoAccelMs

        return latestEgoScore >= lvsaVetoEgoScore || accelJustNow
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }

    // GPS is the primary speed source. When it is unavailable (the 18 Jul
    // drive had Location switched OFF for all 162 s, so speed was -1 the whole
    // time and every speed-gated warning -- including TOO CLOSE -- went dark),
    // fall back to a COARSE speed from the vision ego-motion score so the
    // driving warnings still function. This is deliberately rough: it only
    // needs to answer "roughly town speed vs roughly road speed", not to be
    // accurate. hasTrueSpeed() lets callers know which they are getting.
    private fun egoSpeedMps(): Float {
        val gps = speedProvider.speedMps()
        if (gps >= 0f) return gps

        // No GPS: map the ego-motion score to a rough speed. The score rises
        // with road flow; these breakpoints were chosen so "clearly moving"
        // reads as ~town speed and a high score as ~road speed. Better an
        // approximate speed than no FCW at all.
        if (latestEgoScore < 0f) return -1f
        return when {
            latestEgoScore >= 6f -> 16f    // ~58 km/h
            latestEgoScore >= 4f -> 11f    // ~40 km/h
            latestEgoScore >= egoMovingScore -> 7f   // ~25 km/h
            else -> 0f
        }
    }

    private fun hasTrueSpeed(): Boolean = speedProvider.speedMps() >= 0f

    /**
     * Choose the vehicle that is actually IN OUR LANE.
     *
     * YoloDetector now hands back every plausible candidate instead of
     * pre-picking the biggest one, because "biggest" was frequently a bus in
     * the next lane -- which both raised false BRAKEs and HID the real lead
     * car behind it. Lane membership is decided here, in metres:
     *
     *     distance   from box width + camera intrinsics
     *     lateral    = (box centre - image centre) * distance / focal
     *
     * Among in-lane candidates the closest one is the lead vehicle.
     */
    private fun selectInLaneCandidate(result: YoloDetectionResult): InLaneTarget? {
        var best: InLaneTarget? = null
        var sticky: InLaneTarget? = null

        for (c in result.candidates) {
            val d = distanceEstimator.distanceMeters(c.boundingBox, c.classId, result.imageWidth)
            if (d <= 0f || d > brakeMaxDistanceM) continue

            val lat = distanceEstimator.lateralOffsetMeters(c.boundingBox, d, result.imageWidth)
            if (lat.isNaN()) continue

            val latFromCarCentre = lat + cameraLateralOffsetM
            if (kotlin.math.abs(latFromCarCentre) > egoLaneHalfWidthM) continue

            val cand = InLaneTarget(c, d, latFromCarCentre)

            if (best == null || d < best!!.distanceM) best = cand

            // Is this the same object we tracked last frame?
            val prev = fcwTargetBox
            if (prev != null && iou(prev, c.boundingBox) >= fcwTargetSameIou) {
                sticky = cand
            }
        }

        // Prefer continuing to track the SAME lead vehicle: TTC needs several
        // consistent distance samples on one object, and constantly switching
        // to "whatever is closest this frame" reset that history so TTC rarely
        // computed (mostly "TTC:-"). Only switch away from the sticky target if
        // something is clearly closer (>1.5 m nearer).
        if (sticky != null && best != null) {
            return if (best!!.distanceM < sticky!!.distanceM - 1.5f) best else sticky
        }

        return best
    }

    private data class InLaneTarget(
        val candidate: YoloCandidate,
        val distanceM: Float,
        val lateralM: Float
    )

    private fun pushTtcVote(danger: Boolean) {
        ttcVote[ttcVoteIdx] = danger
        ttcVoteIdx = (ttcVoteIdx + 1) % ttcVote.size
        if (ttcVoteFilled < ttcVote.size) ttcVoteFilled++
    }

    private fun ttcVoteFraction(): Float {
        if (ttcVoteFilled == 0) return 0f
        var c = 0
        for (i in 0 until ttcVoteFilled) if (ttcVote[i]) c++
        return c.toFloat() / ttcVoteFilled
    }

    private fun isTurningNow(): Boolean {
        val now = System.currentTimeMillis()
        if (speedProvider.isTurning(turnYawRateRadS)) {
            lastTurningTime = now
            return true
        }
        return now - lastTurningTime <= turnHoldMs
    }

    private fun resetWarningStates() {
        fcwWarningCounter = 0
        fcwSafeCounter = 0
        fcwDangerActive = false
        fcwLastWarnedAreaRatio = 0f
        resetFcwTracker()
        laneDepartureStartTime = null
        laneWarnHitCount = 0
        eyeClosedStartTime = null
        resetLeadVehicleState()

        egoMotionEstimator.reset()
        laneDetector.reset()
        latestEgoScore = -1f
        fcwHistCount = 0
        fcwTargetBox = null
        fcwSmoothClosingMps = 0f
        latestDistanceM = -1f
        latestHeadwaySec = -1f
        latestTtcSec = -1f
        visionMovingStreak = 0
        visionStaticStreak = 0
        vehicleMotionState = MOTION_UNKNOWN
        stationarySince = 0L

        latestFcwDangerNow = false
        lastFcwDangerTime = 0L
        lvsaMovingSince = 0L

        currentWarningPriority = 0
        currentWarningEndTime = 0L
        warningToken++
        warningText.visibility = View.GONE
        safetyOverlay.setMode("NORMAL")
    }

    private fun resetFcwTracker() {
        fcwHistCount = 0
        fcwTargetBox = null
        fcwSmoothClosingMps = 0f
        smoothedLateralM = Float.NaN
        ttcVoteIdx = 0
        ttcVoteFilled = 0
        fcwLastDangerBox = null
        proxCounter = 0
        proxLastBox = null
    }

    private fun clearLeadVehicleTracking() {
        leadVehicleSeenStartTime = null
        leadVehicleWaiting = false
        leadVehicleReferenceArea = 0f
        lastLeadVehicleArea = 0f
        lastLeadVehicleDetectedTime = 0L
        leadShrinkStreak = 0
        leadWasShrinkingBeforeLost = false
        lastLeadBox = null
        leadIdentityMissCount = 0
    }

    private fun resetLeadVehicleState() {
        clearLeadVehicleTracking()
        latestLeadVehicleStatus = "LVSA: -"
    }

    /**
     * The lead vehicle is the IN-LANE one, decided in metres by
     * selectInLaneCandidate(). The old test used the image-x window
     * (0.38..0.62), which has the same flaw as the FCW gate had: at range, a
     * vehicle in the next lane sits inside it, so LVSA could arm on a bus
     * beside us and then announce "front vehicle moved" when that bus pulled
     * away.
     */
    private fun isLeadVehicleCandidate(target: InLaneTarget?, result: YoloDetectionResult): Boolean {
        if (target == null) return false
        if (target.candidate.classId !in leadVehicleClassIds) return false
        if (result.imageHeight <= 0) return false

        val box = target.candidate.boundingBox
        val bottomY = box.bottom.toFloat() / result.imageHeight.toFloat()

        return bottomY >= 0.35f && target.candidate.areaRatio >= minLeadVehicleAreaRatio
    }

    private fun processLeadVehicleStartAlert(
        target: InLaneTarget?,
        result: YoloDetectionResult
    ) {
        val currentTime = System.currentTimeMillis()

        // LVSA only works after the ego vehicle has been stationary long
        // enough. The motion state machine (vision-based) makes this both
        // fast to disarm when the driver pulls away, and robust against
        // engine vibration while waiting.
        if (!isOwnVehicleStoppedEnough()) {
            latestLeadVehicleStatus = "LVSA: disabled / own moving"

            if (leadVehicleSeenStartTime != null || leadVehicleWaiting) {
                clearLeadVehicleTracking()
            }

            return
        }

        val leadDetected = isLeadVehicleCandidate(target, result)

        if (leadDetected && target != null) {
            val currentArea = target.candidate.areaRatio
            val newBox = target.candidate.boundingBox
            lastLeadVehicleDetectedTime = currentTime

            // Identity check: is this the same vehicle we have been watching?
            // If the best YOLO object momentarily switches to another car,
            // skip the shrink logic for this frame instead of corrupting the
            // reference; if it stays a different car, restart observation.
            if (newBox != null && lastLeadBox != null) {
                val sameLead = iou(lastLeadBox!!, newBox) >= leadSameObjectIou

                if (!sameLead) {
                    leadIdentityMissCount++

                    if (leadIdentityMissCount >= leadIdentityMissLimit) {
                        clearLeadVehicleTracking()
                        leadVehicleSeenStartTime = currentTime
                        leadVehicleReferenceArea = currentArea
                        lastLeadVehicleArea = currentArea
                        lastLeadVehicleDetectedTime = currentTime
                        lastLeadBox = newBox
                        latestLeadVehicleStatus = "LVSA: new lead"
                    } else {
                        latestLeadVehicleStatus =
                            "LVSA: id? $leadIdentityMissCount/$leadIdentityMissLimit"
                    }

                    return
                }
            }

            leadIdentityMissCount = 0
            lastLeadBox = newBox

            if (leadVehicleSeenStartTime == null) {
                leadVehicleSeenStartTime = currentTime
                leadVehicleReferenceArea = currentArea
                leadVehicleWaiting = false
            }

            val observedDuration = currentTime - (leadVehicleSeenStartTime ?: currentTime)

            if (!leadVehicleWaiting) {
                // Smooth the reference area instead of taking the max. Using
                // the max lets a single jittery large frame inflate the
                // reference, after which normal box jitter looks like a 25%
                // drop and fires a false "moved" alert.
                leadVehicleReferenceArea =
                    0.7f * leadVehicleReferenceArea + 0.3f * currentArea
                latestLeadVehicleStatus =
                    "LVSA: observing ${String.format("%.1f", observedDuration / 1000.0)}s"

                if (observedDuration >= leadVehicleStableMs) {
                    leadVehicleWaiting = true
                    leadShrinkStreak = 0
                    latestLeadVehicleStatus = "LVSA: waiting"
                }
            } else {
                // Count frames where the car is getting smaller (moving
                // away). The 0.985 factor ignores tiny jitter. A single
                // non-shrinking frame only DECAYS the streak instead of
                // resetting it to zero -- in the road test, YOLO jitter kept
                // the streak stuck at 0 through a real departure.
                if (currentArea < lastLeadVehicleArea * 0.985f) {
                    leadShrinkStreak++
                } else if (leadShrinkStreak > 0) {
                    leadShrinkStreak--
                }
                leadWasShrinkingBeforeLost = leadShrinkStreak >= 2

                val movedAway =
                    leadShrinkStreak >= leadShrinkStreakRequired &&
                            leadVehicleReferenceArea > 0f &&
                            currentArea <= leadVehicleReferenceArea * (1f - leadVehicleMovedDropRatio)

                latestLeadVehicleStatus =
                    "LVSA: waiting Area:${String.format("%.3f", currentArea)} shrink:$leadShrinkStreak"

                if (
                    movedAway &&
                    !latestFcwDangerNow &&
                    currentTime - lastLeadVehicleAlertTime >= leadVehicleAlertCooldownMs
                ) {
                    if (lvsaInstantMotionVeto()) {
                        // Driver is already rolling: alerting now would only
                        // annoy. Cancel quietly.
                        clearLeadVehicleTracking()
                        latestLeadVehicleStatus = "LVSA: cancelled / own moving"
                    } else {
                        lastLeadVehicleAlertTime = currentTime
                        latestLeadVehicleStatus = "LVSA: moved"
                        clearLeadVehicleTracking()

                        runOnUiThread {
                            showAdasWarning(
                                message = "FRONT VEHICLE MOVED",
                                backgroundColor = Color.parseColor("#CC2196F3"),
                                overlayMode = "NORMAL",
                                priority = PRIORITY_LVSA
                            )
                        }
                    }
                }
            }

            lastLeadVehicleArea = currentArea
        } else {
            // Detection lost. A car doesn't only disappear by driving off --
            // occlusion, flicker and low light drop the box too. Only call it
            // "moved" if the box was already clearly shrinking right before
            // it vanished; otherwise it's a dropout.
            if (leadVehicleWaiting) {
                val lostDuration = currentTime - lastLeadVehicleDetectedTime
                latestLeadVehicleStatus = "LVSA: waiting / lost"

                if (
                    leadWasShrinkingBeforeLost &&
                    lostDuration in leadVehicleLostGraceMs..leadVehicleLostMaxMs &&
                    currentTime - lastLeadVehicleAlertTime >= leadVehicleAlertCooldownMs
                ) {
                    if (lvsaInstantMotionVeto()) {
                        clearLeadVehicleTracking()
                        latestLeadVehicleStatus = "LVSA: cancelled / own moving"
                    } else {
                        lastLeadVehicleAlertTime = currentTime
                        latestLeadVehicleStatus = "LVSA: moved"
                        clearLeadVehicleTracking()

                        runOnUiThread {
                            showAdasWarning(
                                message = "FRONT VEHICLE MOVED",
                                backgroundColor = Color.parseColor("#CC2196F3"),
                                overlayMode = "NORMAL",
                                priority = PRIORITY_LVSA
                            )
                        }
                    }
                } else if (lostDuration > leadVehicleLostMaxMs) {
                    // Gone too long without a prior shrink -> treat as a
                    // dropout and stop quietly.
                    clearLeadVehicleTracking()
                    latestLeadVehicleStatus = "LVSA: lost"
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

    private fun iou(a: Rect, b: Rect): Float {
        val left = maxOf(a.left, b.left)
        val top = maxOf(a.top, b.top)
        val right = minOf(a.right, b.right)
        val bottom = minOf(a.bottom, b.bottom)

        val inter = maxOf(0, right - left).toFloat() * maxOf(0, bottom - top).toFloat()
        val union = a.width().toFloat() * a.height() + b.width().toFloat() * b.height() - inter

        return if (union > 0f) inter / union else 0f
    }

    private fun applyDebugMode() {
        val debugVisibility = if (debugMode) View.VISIBLE else View.GONE

        btnTestFCW.visibility = debugVisibility
        btnTestLDW.visibility = debugVisibility
        btnTestDMS.visibility = debugVisibility
        fpsText.visibility = debugVisibility

        safetyOverlay.setLaneDebugEnabled(debugMode)
        btnStatusMode.text = if (debugMode) "Status ON" else "Status OFF"
        statusText.text = if (debugMode) "ADAS Status Mode" else "ADAS Active"

        if (debugMode) {
            updateDualStatusText()
        }
    }

    private fun applyDriverSideLabel() {
        btnDriverSide.text =
            if (drowsinessDetector.driverOnRight) "Driver: Right" else "Driver: Left"
    }

    private fun applyPerformanceMode() {
        if (ecoMode) {
            fcwFrameInterval = 6
            ldwFrameInterval = 5
            dmsFrameInterval = 10
            btnPerformanceMode.text = "Mode: Eco"
            Toast.makeText(this, "Eco Mode enabled", Toast.LENGTH_SHORT).show()
        } else {
            fcwFrameInterval = 5
            ldwFrameInterval = 3
            dmsFrameInterval = 8
            btnPerformanceMode.text = "Mode: Normal"
            Toast.makeText(this, "Normal Mode enabled", Toast.LENGTH_SHORT).show()
        }

        updateDualStatusText()
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

    private fun showAdasWarning(
        message: String,
        backgroundColor: Int,
        overlayMode: String,
        priority: Int,
        durationMs: Long = 2000L
    ) {
        val now = System.currentTimeMillis()

        // If a higher-priority warning is still active, do not replace it
        if (now < currentWarningEndTime && priority < currentWarningPriority) {
            return
        }

        currentWarningPriority = priority
        currentWarningEndTime = now + durationMs
        warningToken++

        val thisWarningToken = warningToken

        safetyOverlay.setMode(overlayMode)

        warningText.text = message
        warningText.setBackgroundColor(backgroundColor)
        warningText.visibility = View.VISIBLE

        warningSound.playForMessage(message)

        Handler(Looper.getMainLooper()).postDelayed({
            // Only hide if this is still the latest warning
            if (thisWarningToken == warningToken) {
                warningText.visibility = View.GONE
                safetyOverlay.setMode("NORMAL")
                currentWarningPriority = 0
                currentWarningEndTime = 0L
            }
        }, durationMs)
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

        // Pure-vision ego-motion: update on EVERY back frame (cheap, reads a
        // sparse grid of the luma plane only, does not close the proxy).
        // This runs before the detectors so FCW / LDW / LVSA all gate on the
        // resulting motion state.
        val egoScore = egoMotionEstimator.update(imageProxy)
        latestEgoScore = egoScore

        // Intrinsics depend on the analysis width; set once it is known.
        if (distanceEstimator.focalPx() <= 0f) {
            distanceEstimator.setImageWidth(imageProxy.width)
        }

        if (egoScore >= 0f) {
            if (egoScore >= egoMovingScore) {
                visionMovingStreak++
                visionStaticStreak = 0
            } else if (egoScore <= egoStaticScore) {
                visionStaticStreak++
                visionMovingStreak = 0
            }
            // Scores inside the hysteresis band leave both streaks unchanged.
        }

        updateVehicleMotionState(currentTime)

        // Self-calibrate where the road is. Only meaningful while driving:
        // the bonnet is static relative to the camera, the road is not.
        stripCalibrator.update(imageProxy, vehicleMotionState == MOTION_MOVING)

        val shouldRunFcw =
            !isFcwProcessing && backAnalysisFrameIndex % fcwFrameInterval == 0

        if (shouldRunFcw) {
            isFcwProcessing = true

            yoloDetector.detect(imageProxy) { result ->
                isFcwProcessing = false

                // ---- Pick the vehicle IN OUR LANE, in metres ---------------
                // Not "the biggest box near the middle of the frame": that was
                // routinely a bus in the next lane.
                val target = selectInLaneCandidate(result)
                val box = target?.candidate?.boundingBox
                val targetClassName = target?.candidate?.className
                val targetConfidence = target?.candidate?.confidence ?: 0f
                val targetArea = target?.candidate?.areaRatio ?: 0f

                // ---- Physics: how far, how fast are we closing -------------
                val speedMps = egoSpeedMps()

                val distanceM = target?.distanceM ?: -1f
                latestLateralM = target?.lateralM ?: Float.NaN

                // Smooth the lateral estimate: it wobbles with box width (on
                // 21 Jul the SAME approach read 0.8->1.1->1.2 m over 3 frames),
                // and a single frame dipping inside the lane must not fire a
                // warning on an adjacent-lane car.
                smoothedLateralM = when {
                    latestLateralM.isNaN() -> Float.NaN
                    smoothedLateralM.isNaN() -> latestLateralM
                    else -> lateralEma * smoothedLateralM + (1f - lateralEma) * latestLateralM
                }

                val validDistance = distanceM in 0.5f..brakeMaxDistanceM
                latestDistanceM = if (validDistance) distanceM else -1f

                val ttc = if (validDistance) {
                    updateDistanceTtc(distanceM, box, speedMps)
                } else {
                    updateDistanceTtc(-1f, null, speedMps)
                }
                latestTtcSec = ttc

                val headway = if (validDistance && speedMps > 0f) {
                    distanceEstimator.headwaySeconds(distanceM, speedMps)
                } else -1f
                latestHeadwaySec = headway

                val turning = isTurningNow()
                val ownVehicleStopped = isVehicleClearlyStationary()

                // Show the in-lane target if we have one, else the nearest
                // detection, so the overlay never looks "blind".
                val displayBox = box ?: result.boundingBox

                // ---- BRAKE: only on real, imminent closing ------------------
                // Replaces YoloDetector.collisionWarning (Area >= 0.040),
                // which fired on Area:0.056/0.072 = distant cars, and
                // earlyFrontWarning, which was the same mistake with a
                // different constant. Neither knew metres or speed.
                //
                // Three independent gates, all physical:
                //   - we must be driving fast enough for a collision to matter
                //   - we must not be cornering (the object is in front of the
                //     camera, not necessarily in our path)
                //   - the gap must actually be closing (TTC), or already be
                //     unsurvivably small (headway)
                val movingFastEnough = speedMps >= brakeMinSpeedMps

                // TTC is noisy at range (box-width jitter): on 19 Jul it swung
                // 8.8 -> 6.3 -> 2.6 -> 74 s during ONE smooth approach. A single
                // dip below the threshold must not fire BRAKE. Require the TTC
                // danger to be present on a MAJORITY of recent frames (voting
                // buffer), and never BRAKE on TTC alone when the true gap is
                // still large.
                val ttcDangerRaw =
                    validDistance && ttc <= brakeTtcSec && distanceM <= brakeTtcMaxDistanceM

                pushTtcVote(ttcDangerRaw)
                val ttcDanger = ttcVoteFraction() >= brakeTtcVoteFraction

                // Pure-headway BRAKE needs a TRUE speed; a coarse vision speed
                // is not accurate enough to trust a bare headway number.
                val headwayDanger = hasTrueSpeed() && headway in 0f..brakeHeadwaySec

                // The target already passed the 1.7 m ego-lane test in
                // selectInLaneCandidate(). Require the SMOOTHED lateral to still
                // be within it, so a car drifting across the boundary with a
                // jittery estimate can't fire a single-frame BRAKE.
                val laterallyInLaneForBrake =
                    smoothedLateralM.isNaN() ||
                            kotlin.math.abs(smoothedLateralM) <= egoLaneHalfWidthM

                val dangerNow =
                    validDistance &&
                            movingFastEnough &&
                            laterallyInLaneForBrake &&
                            !turning &&
                            !ownVehicleStopped &&
                            (ttcDanger || headwayDanger)

                latestFcwDangerNow = dangerNow

                if (dangerNow) {
                    lastFcwDangerTime = System.currentTimeMillis()
                }

                latestFcwStatus =
                    "FCW ${targetClassName ?: "-"} " +
                            "n:${result.candidates.size} " +
                            "Conf:${String.format("%.2f", targetConfidence)} " +
                            "D:${if (validDistance) String.format("%.1fm", distanceM) else "-"} " +
                            "Lat:${if (!latestLateralM.isNaN()) String.format("%.1fm", latestLateralM) else "-"} " +
                            "HW:${if (headway >= 0f) String.format("%.1fs", headway) else "-"} " +
                            "TTC:${if (ttc < 90f) String.format("%.1fs", ttc) else "-"} " +
                            "Turn:$turning " +
                            "spd:${if (hasTrueSpeed()) "gps" else "vis"} " +
                            "Danger:$dangerNow"

                processLeadVehicleStartAlert(target, result)

                runOnUiThread {
                    safetyOverlay.setFcwBox(
                        displayBox,
                        result.imageWidth,
                        result.imageHeight
                    )
                    updateDualStatusText()
                }

                if (dangerNow) {
                    // Persistence with identity: only count consecutive
                    // danger frames when they belong to the SAME object.
                    // A parked car sweeping through the frame centre during
                    // a turn does not keep IoU across FCW frames, so it can
                    // no longer fire BRAKE off a single glimpse.
                    val previousDangerBox = fcwLastDangerBox

                    val samePreviousDanger =
                        previousDangerBox != null &&
                                box != null &&
                                iou(previousDangerBox, box) >= fcwSameObjectIou

                    fcwWarningCounter = if (samePreviousDanger) {
                        fcwWarningCounter + 1
                    } else {
                        1
                    }

                    fcwLastDangerBox = box
                    fcwSafeCounter = 0
                } else {
                    fcwWarningCounter = 0
                    fcwLastDangerBox = null
                    fcwSafeCounter++

                    if (fcwSafeCounter >= fcwSafeThreshold) {
                        fcwDangerActive = false
                        fcwLastWarnedAreaRatio = 0f
                        fcwSafeCounter = 0
                    }
                }

                val currentTimeForWarning = System.currentTimeMillis()

                val objectMuchCloserAfterWarning =
                    fcwDangerActive &&
                            fcwLastWarnedAreaRatio > 0f &&
                            result.areaRatio >= fcwLastWarnedAreaRatio * fcwRewarnAreaIncreaseRatio

                if (
                    fcwWarningCounter >= fcwWarningThreshold &&
                    (!fcwDangerActive || objectMuchCloserAfterWarning) &&
                    currentTimeForWarning - lastFcwWarningTime >= fcwCooldownMs
                ) {
                    lastFcwWarningTime = currentTimeForWarning
                    fcwWarningCounter = 0
                    fcwDangerActive = true
                    fcwLastWarnedAreaRatio = result.areaRatio

                    runOnUiThread {
                        showAdasWarning(
                            message = "BRAKE!",
                            backgroundColor = Color.parseColor("#CCFF0000"),
                            overlayMode = "FCW",
                            priority = PRIORITY_FCW,
                            durationMs = 2500L
                        )
                    }
                }

                // --- Early proximity reminder ("TOO CLOSE") -----------------
                // Primary rule: the two-second headway, meaningful at any
                // speed. Fallback rule: when speed is unavailable (GPS off),
                // headway cannot be computed, so warn on a short, CLOSING gap
                // instead -- otherwise the whole tier goes dark, which is
                // exactly what happened on 18 Jul.
                val movingForProx =
                    vehicleMotionState == MOTION_MOVING && !ownVehicleStopped

                // Headway rule. At low speed (crawling traffic) headway alone
                // nags, so below proxLowSpeedMps additionally require the gap to
                // be genuinely short. This keeps the "slow but very close"
                // warning you asked for without firing on every stop-go metre.
                val headwayProx =
                    hasTrueSpeed() &&
                            speedMps >= proxMinSpeedMps &&
                            headway in 0f..proxHeadwaySec &&
                            (speedMps >= proxLowSpeedMps ||
                                    distanceM in 0.5f..proxLowSpeedMaxDistanceM)

                val distanceOnlyProx =
                    !hasTrueSpeed() &&
                            movingForProx &&
                            distanceM in 0.5f..proxFallbackDistanceM &&
                            fcwSmoothClosingMps >= proxFallbackClosingMps

                // TOO CLOSE additionally requires the target to be genuinely
                // centred (tighter lane, smoothed lateral), so it stops firing
                // on cars in the adjacent lane.
                val laterallyCentredForProx =
                    !smoothedLateralM.isNaN() &&
                            kotlin.math.abs(smoothedLateralM) <= proxLaneHalfWidthM

                val proximityNow =
                    validDistance &&
                            laterallyCentredForProx &&
                            !dangerNow &&
                            !turning &&
                            !ownVehicleStopped &&
                            (headwayProx || distanceOnlyProx)

                if (proximityNow) {
                    val samePrevProx =
                        proxLastBox != null && box != null &&
                                iou(proxLastBox!!, box) >= fcwSameObjectIou

                    proxCounter = if (samePrevProx) proxCounter + 1 else 1
                    proxLastBox = box
                } else {
                    proxCounter = 0
                    proxLastBox = null
                }

                val nowForProx = System.currentTimeMillis()

                if (
                    proxCounter >= proxPersistFrames &&
                    nowForProx - lastProxWarningTime >= proxCooldownMs &&
                    nowForProx - lastFcwWarningTime >= proxSuppressAfterBrakeMs
                ) {
                    lastProxWarningTime = nowForProx
                    proxCounter = 0

                    runOnUiThread {
                        showAdasWarning(
                            message = "TOO CLOSE",
                            backgroundColor = Color.parseColor("#CCFF8C00"),
                            overlayMode = "NORMAL",
                            priority = PRIORITY_PROXIMITY,
                            durationMs = 1800L
                        )
                    }
                }
            }

            return
        }

        if (backAnalysisFrameIndex % ldwFrameInterval == 0) {

            val nowForLdw = System.currentTimeMillis()

            // LDW is a driving warning: suppress ONLY when the vehicle is
            // CLEARLY stationary. Previously the accel-based check reported
            // "stopped" for 30-40% of actual driving time (constant speed =
            // zero acceleration), so real lane crossings never warned.
            // LDW only matters at road speed. Below ~20 km/h (car parks,
            // junctions, crawling traffic) production LDW stays silent, and
            // GPS makes that trivial to honour.
            // LDW low-speed suppression only trusts a TRUE GPS speed. When
            // GPS is off, egoSpeedMps() returns a coarse vision estimate that
            // reads low even while driving normally -- on the 20 Jul drive that
            // fallback suppressed LDW as 'slow' for 93 frames with GPS off.
            // Without a real speed we do NOT suppress on slowness; the motion
            // state (stationary) and turning gates still apply.
            val ldwSuppressedBecauseSlow =
                hasTrueSpeed() &&
                        speedProvider.speedMps() < ldwMinSpeedMps &&
                        vehicleMotionState != MOTION_MOVING

            val ldwSuppressedBecauseStopped = isVehicleClearlyStationary()

            // Cornering makes "which side of the line am I on" meaningless.
            val ldwSuppressedBecauseTurn = isTurningNow()

            // FCW priority previously starved LDW: on 16 Jul the false BRAKEs
            // held this gate for 38 s of a 293 s drive (13%), and LDW was
            // inactive for 59% of the drive overall. The hold is now short and
            // keyed to an ACTUAL brake event, not to every frame that merely
            // looked dangerous.
            val ldwSuppressedBecauseFcw =
                nowForLdw - lastFcwWarningTime <= fcwDangerHoldMs

            if (
                ldwSuppressedBecauseStopped ||
                ldwSuppressedBecauseFcw ||
                ldwSuppressedBecauseSlow ||
                ldwSuppressedBecauseTurn
            ) {
                laneDepartureStartTime = null
                lastLaneIssueDetectedTime = 0L
                laneWarnHitCount = 0

                latestLaneStatus = when {
                    ldwSuppressedBecauseFcw -> "Lane: suppressed / FCW priority"
                    ldwSuppressedBecauseTurn -> "Lane: suppressed / turning"
                    ldwSuppressedBecauseSlow -> "Lane: suppressed / slow"
                    else -> "Lane: suppressed / stopped"
                }

                runOnUiThread {
                    safetyOverlay.setRoadOverlayVisible(false)
                    updateDualStatusText()
                }

                imageProxy.close()
                return
            } else {
                runOnUiThread {
                    safetyOverlay.setRoadOverlayVisible(true)
                }
            }

            // The strip position comes from measurement, not a guess. The
            // hand-picked 0.78-0.94 used until now sat on the car's bonnet,
            // which is why LDW never fired once in 293 s on 16 Jul.
            val result = laneDetector.detect(
                imageProxy,
                stripCalibrator.stripTop(),
                stripCalibrator.stripBottom()
            )

            runOnUiThread {
                safetyOverlay.setLaneDebug(
                    result.stripTop,
                    result.stripBottom,
                    result.bandXs,
                    result.rawBandXs,
                    stripCalibrator.statusText() + "  " + result.debugText
                )
            }

            val possibleLaneIssue = result.laneWarning

            // p is the normalised lane position: 0 = centred, +/-1 = on a
            // line. Watching this is how you tell a real departure from the
            // old "band in the middle" false alarms.
            latestLaneStatus =
                "Lane L:${result.leftDetected} R:${result.rightDetected} " +
                        "p:${String.format("%.2f", result.offsetRatio)} " +
                        "Lines:${result.lineCount} " +
                        "Cross:${result.crossingDetected} " +
                        "Cplx:${result.complexRoadMarking} " +
                        "Warn:${result.laneWarning}"

            val currentTimeForWarning = System.currentTimeMillis()

            if (possibleLaneIssue) {
                lastLaneIssueDetectedTime = currentTimeForWarning
                laneWarnHitCount++

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
                            "p:${String.format("%.2f", result.offsetRatio)} " +
                            "Lines:${result.lineCount} " +
                            "LDW:${departureDuration / 1000.0}s " +
                            "Hits:$laneWarnHitCount"

                if (
                    departureDuration >= laneDepartureDurationMs &&
                    laneWarnHitCount >= laneWarnHitsRequired &&
                    currentTimeForWarning - lastLaneWarningTime >= laneWarningCooldownMs
                ) {
                    lastLaneWarningTime = currentTimeForWarning
                    laneDepartureStartTime = null
                    laneWarnHitCount = 0

                    runOnUiThread {
                        showAdasWarning(
                            message = "LANE WARNING",
                            backgroundColor = Color.parseColor("#CCFFA500"),
                            overlayMode = "LDW",
                            priority = PRIORITY_LDW
                        )
                    }
                }
            } else {
                laneDepartureStartTime = null
                laneWarnHitCount = 0
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

                val sideText = if (drowsinessDetector.driverOnRight) "R" else "L"

                latestDmsStatus = if (result.faceDetected) {
                    "DMS drv[$sideText]@${String.format("%.2f", result.driverFaceX)} " +
                            "faces:${result.faceCount} " +
                            "eye:${String.format("%.2f", result.leftEyeOpen ?: -1f)}/" +
                            "${String.format("%.2f", result.rightEyeOpen ?: -1f)}"
                } else {
                    "DMS drv[$sideText] faces:${result.faceCount} no driver face"
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
                            showAdasWarning(
                                message = "DROWSINESS WARNING",
                                backgroundColor = Color.parseColor("#CCFF0000"),
                                overlayMode = "DMS",
                                priority = PRIORITY_DMS
                            )
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

        val performanceText = if (ecoMode) "Eco Mode" else "Normal Mode"

        fpsText.text =
            "System Status\n" +
                    "$performanceText\n" +
                    "Back FPS: $latestBackFps\n" +
                    "Front FPS: $latestFrontFps\n" +
                    "$latestOwnMotionStatus\n" +
                    "${stripCalibrator.statusText()}\n" +
                    "$latestLaneStatus\n" +
                    "$latestFcwStatus\n" +
                    "$latestLeadVehicleStatus\n" +
                    "$latestDmsStatus"
    }

    override fun onDestroy() {
        super.onDestroy()

        if (::sensorManager.isInitialized) {
            sensorManager.unregisterListener(this)
        }

        if (::speedProvider.isInitialized) {
            speedProvider.stop()
        }

        cameraExecutor.shutdown()
        warningSound.release()
        drowsinessDetector.close()
        yoloDetector.close()
    }
}