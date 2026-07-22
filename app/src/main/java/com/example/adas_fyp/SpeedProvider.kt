package com.example.adas_fyp

import android.annotation.SuppressLint
import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Bundle
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Ego-vehicle speed (GPS) and yaw rate (gyroscope).
 *
 * WHY THIS EXISTS
 * Every failure across the 13/14/15/16 July road tests traces back to the app
 * not knowing two simple physical quantities:
 *
 *   1. HOW FAST ARE WE GOING?
 *      Linear acceleration is ~0 both when parked and when cruising at
 *      constant speed, so the accelerometer literally cannot tell them
 *      apart. Vision ego-motion helped but needed hand calibration per
 *      phone. GPS reports speed directly, in m/s, with no calibration.
 *      Speed is also what turns "apparent size" into real danger: a car 6 m
 *      ahead is fine at 10 km/h and an emergency at 60 km/h.
 *
 *   2. ARE WE TURNING?
 *      Most false BRAKEs happen mid-corner, when a parked or oncoming car
 *      sweeps through the middle of the frame. It is in front of the camera
 *      but not in our path. A gyro tells us we are cornering, so we can
 *      stand the driving warnings down.
 *
 * The yaw rate is taken as the component of the gyro vector along gravity,
 * which makes it independent of how the phone is mounted (portrait, tilted,
 * angled in a cradle) -- no per-device setup.
 *
 * GPS is used as the primary speed source and vision/accelerometer stay as
 * fallbacks: in tunnels, multi-storey car parks and urban canyons the fix
 * can drop out, and callers must degrade gracefully rather than trust a
 * stale speed.
 */
class SpeedProvider(private val context: Context) : SensorEventListener, LocationListener {

    // ---- GPS ----------------------------------------------------------------
    private var locationManager: LocationManager? = null
    private var lastFixTime = 0L
    private var speedMps = -1f          // <0 = unknown

    // A fix older than this is not trusted (tunnel / underground car park).
    private val fixStaleMs = 3000L

    // Distinguishable failure modes -- "no speed" alone told us nothing.
    private val STATE_NO_PERMISSION = 0
    private val STATE_DISABLED = 1
    private val STATE_WAITING = 2
    private val STATE_OK = 3
    private val STATE_NETWORK_ONLY = 4
    private var gpsState = STATE_WAITING

    // GPS speed is noisy near zero; below this we do not claim to be moving.
    private val gpsNoiseFloorMps = 0.6f

    // ---- Gyroscope ----------------------------------------------------------
    private var sensorManager: SensorManager? = null
    private var gyroSensor: Sensor? = null
    private var gravitySensor: Sensor? = null

    private val gravityVec = FloatArray(3)
    private var haveGravity = false

    private var yawRateRadS = 0f        // smoothed |yaw rate|
    private val yawEma = 0.7f

    fun start() {
        startGps()
        startSensors()
    }

    @SuppressLint("MissingPermission")
    private fun startGps() {
        // The 16 Jul evening run showed "GPS:-" for the whole drive, which was
        // indistinguishable from "permission denied", "no fix yet" and "this
        // phone has no GPS". Those need different fixes, so the status now
        // says which one it is.
        //
        // Subscribing to GPS_PROVIDER alone was also part of the problem: raw
        // GPS can take 30-60 s for a first fix and drops out under cover.
        // NETWORK_PROVIDER is subscribed alongside it as a coarse backstop;
        // whichever reports a usable speed first is used.
        try {
            val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
            locationManager = lm

            var any = false

            for (provider in listOf(
                LocationManager.GPS_PROVIDER,
                LocationManager.NETWORK_PROVIDER
            )) {
                try {
                    if (!lm.isProviderEnabled(provider)) continue
                    lm.requestLocationUpdates(provider, 500L, 0f, this)
                    any = true
                } catch (e: IllegalArgumentException) {
                    // provider not present on this device
                }
            }

            gpsState = if (any) STATE_WAITING else STATE_DISABLED
        } catch (e: SecurityException) {
            locationManager = null
            gpsState = STATE_NO_PERMISSION
        }
    }

    private fun startSensors() {
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        sensorManager = sm

        gyroSensor = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        gravitySensor = sm.getDefaultSensor(Sensor.TYPE_GRAVITY)
            ?: sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        gyroSensor?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
        gravitySensor?.let { sm.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) }
    }

    fun stop() {
        try {
            locationManager?.removeUpdates(this)
        } catch (e: SecurityException) {
            // ignore
        }
        sensorManager?.unregisterListener(this)
    }

    // ---- Public API ---------------------------------------------------------

    /** Ego speed in m/s, or -1 if unknown / stale. */
    fun speedMps(): Float {
        if (speedMps < 0f) return -1f
        if (System.currentTimeMillis() - lastFixTime > fixStaleMs) return -1f
        return speedMps
    }

    fun speedKmh(): Float {
        val s = speedMps()
        return if (s < 0f) -1f else s * 3.6f
    }

    fun hasSpeed(): Boolean = speedMps() >= 0f

    /** Smoothed absolute yaw rate in rad/s (0 when the gyro is unavailable). */
    fun yawRateRadS(): Float = yawRateRadS

    /** True while cornering hard enough that "straight ahead" is meaningless. */
    fun isTurning(thresholdRadS: Float): Boolean = yawRateRadS >= thresholdRadS

    fun statusText(): String {
        val s = speedKmh()

        val spd = when {
            s >= 0f -> "GPS:${String.format("%.0f", s)}km/h"
            gpsState == STATE_NO_PERMISSION -> "GPS:noperm"
            gpsState == STATE_DISABLED -> "GPS:off"
            gpsState == STATE_NETWORK_ONLY -> "GPS:netonly"
            lastFixTime == 0L -> "GPS:nofix"
            else -> "GPS:stale"
        }

        return "$spd Yaw:${String.format("%.2f", yawRateRadS)}"
    }

    // ---- Callbacks ----------------------------------------------------------

    override fun onLocationChanged(location: Location) {
        // ONLY GPS_PROVIDER MAY SET SPEED.
        //
        // NETWORK_PROVIDER was added as a backstop for slow GPS first-fixes,
        // and it poisoned the speed source: cell/wifi fixes routinely report
        // hasSpeed()=true with speed=0. On the 16 Jul run that produced
        // GPS:0km/h while the car was clearly moving (13% of speed samples
        // contradicted the vision motion state), which fed the LDW "slow"
        // gate and shut lane detection off for 28% of the drive.
        //
        // Network fixes are still useful as proof the location stack works,
        // but their speed is discarded.
        val isGps = location.provider == LocationManager.GPS_PROVIDER

        if (!isGps) {
            if (gpsState == STATE_WAITING) gpsState = STATE_NETWORK_ONLY
            return
        }

        lastFixTime = System.currentTimeMillis()

        if (location.hasSpeed()) {
            val v = location.speed
            speedMps = if (v < gpsNoiseFloorMps) 0f else v
            gpsState = STATE_OK
        } else {
            speedMps = -1f
            gpsState = STATE_WAITING
        }
    }

    @Deprecated("Required by LocationListener on older APIs")
    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {
    }

    override fun onProviderEnabled(provider: String) {
    }

    override fun onProviderDisabled(provider: String) {
        speedMps = -1f
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GRAVITY -> {
                gravityVec[0] = event.values[0]
                gravityVec[1] = event.values[1]
                gravityVec[2] = event.values[2]
                haveGravity = true
            }

            Sensor.TYPE_ACCELEROMETER -> {
                if (gravitySensor?.type == Sensor.TYPE_ACCELEROMETER) {
                    // Low-pass the accelerometer to recover the gravity direction.
                    val a = 0.8f
                    gravityVec[0] = a * gravityVec[0] + (1 - a) * event.values[0]
                    gravityVec[1] = a * gravityVec[1] + (1 - a) * event.values[1]
                    gravityVec[2] = a * gravityVec[2] + (1 - a) * event.values[2]
                    haveGravity = true
                }
            }

            Sensor.TYPE_GYROSCOPE -> {
                if (!haveGravity) return

                val gx = gravityVec[0]
                val gy = gravityVec[1]
                val gz = gravityVec[2]
                val norm = sqrt(gx * gx + gy * gy + gz * gz)

                if (norm < 1e-3f) return

                // Yaw rate = rotation about the vertical (gravity) axis.
                // Projecting the gyro vector onto gravity makes this work for
                // any phone mounting angle without calibration.
                val raw = (event.values[0] * gx + event.values[1] * gy + event.values[2] * gz) / norm

                yawRateRadS = yawEma * yawRateRadS + (1f - yawEma) * abs(raw)
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
    }
}