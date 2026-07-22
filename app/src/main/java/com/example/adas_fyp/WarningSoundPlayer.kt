package com.example.adas_fyp

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Handler
import android.os.Looper
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sin

class WarningSoundPlayer {

    private val sampleRate = 44100
    private val handler = Handler(Looper.getMainLooper())

    /** Master volume 0..1 */
    private val masterVolume = 0.85f

    // -----------------------------------------------------------------------

    fun playForMessage(message: String) {
        when {
            message.contains("BRAKE", ignoreCase = true)         -> playFcwBrake()
            message.contains("TOO CLOSE", ignoreCase = true)     -> playTooClose()
            message.contains("LANE", ignoreCase = true)          -> playLdw()
            message.contains("FRONT VEHICLE", ignoreCase = true) -> playLvsa()
            message.contains("DROWSINESS", ignoreCase = true)    -> playDms()
            else                                                   -> playDefault()
        }
    }

    // -----------------------------------------------------------------------
    // BRAKE — Honda CMBS auto-brake alert style
    //
    // Honda CMBS fires a "constant beeping sound" that escalates across stages.
    // The signature the user recognises from Honda V-Sensing videos is:
    //   • A pair of quick double-beeps (BI-BI ... BI-BI) at ~2000 Hz as stage 1
    //     FCW fires, then immediately rolls into
    //   • A rapid burst of single beeps (BI-BI-BI-BI-BI) as CMBS stage 2
    //     kicks in — this is what auto-brake sounds like.
    // We combine both into one escalating sequence.
    // -----------------------------------------------------------------------
    private fun playFcwBrake() {
        // Stage 1 FCW — two double-beeps (BI-BI  pause  BI-BI)
        val stage1 = listOf(
            Beep(2000f, 90, 1.00f),
            Beep(2000f, 90, 1.00f),    // pair 1
            GAP(130),
            Beep(2000f, 90, 1.00f),
            Beep(2000f, 90, 1.00f)     // pair 2
        )
        // Stage 2 CMBS — rapid burst (BI-BI-BI-BI-BI), tighter gaps, louder
        val stage2 = listOf(
            GAP(160),
            Beep(2000f, 80, 1.00f),
            Beep(2000f, 80, 1.00f),
            Beep(2000f, 80, 1.00f),
            Beep(2000f, 80, 1.00f),
            Beep(2000f, 80, 1.00f)
        )
        scheduleSequence(stage1 + stage2, defaultGapMs = 60)
    }

    // TOO CLOSE — Honda FCW early alert: softer 2-pair double-beep
    private fun playTooClose() {
        scheduleSequence(
            listOf(
                Beep(1800f, 100, 0.70f),
                Beep(1800f, 100, 0.70f),
                GAP(170),
                Beep(1800f, 100, 0.70f),
                Beep(1800f, 100, 0.70f)
            ),
            defaultGapMs = 65
        )
    }

    // LDW — Honda RDM/LKAS lane departure: bi-bip, mid pitch
    private fun playLdw() {
        scheduleSequence(
            listOf(
                Beep(1200f, 150, 0.85f),
                Beep(1200f, 150, 0.85f)
            ),
            defaultGapMs = 190
        )
    }

    // LVSA — two ascending beeps
    private fun playLvsa() {
        scheduleSequence(
            listOf(
                Beep(900f,  190, 0.75f),
                Beep(1300f, 190, 0.85f)
            ),
            defaultGapMs = 145
        )
    }

    // DMS — 3 mid beeps, softer than BRAKE
    private fun playDms() {
        scheduleSequence(
            listOf(
                Beep(1500f, 110, 0.80f),
                Beep(1500f, 110, 0.80f),
                Beep(1500f, 110, 0.80f)
            ),
            defaultGapMs = 85
        )
    }

    private fun playDefault() {
        scheduleSequence(
            listOf(Beep(1000f, 200, 0.75f)),
            defaultGapMs = 0
        )
    }

    // -----------------------------------------------------------------------
    // Sequence DSL: Beep or explicit GAP items
    // -----------------------------------------------------------------------

    private data class Beep(val freqHz: Float, val durationMs: Int, val vol: Float)
    private data class GAP(val ms: Int)

    private fun scheduleSequence(items: List<Any>, defaultGapMs: Int) {
        var delayMs = 0L
        for (item in items) {
            when (item) {
                is Beep -> {
                    val b = item
                    handler.postDelayed({
                        playBeep(b.freqHz, b.durationMs, masterVolume * b.vol)
                    }, delayMs)
                    delayMs += b.durationMs + defaultGapMs
                }
                is GAP -> delayMs += item.ms
            }
        }
    }

    // -----------------------------------------------------------------------
    // Core synthesis: sine wave + ADSR envelope via AudioTrack (static mode)
    // -----------------------------------------------------------------------

    private fun playBeep(freqHz: Float, durationMs: Int, volume: Float) {
        val totalSamples = sampleRate * durationMs / 1000
        val attackSamples  = min(sampleRate * 7  / 1000, totalSamples / 4)
        val releaseSamples = min(sampleRate * 12 / 1000, totalSamples / 4)

        val pcm = ShortArray(totalSamples)
        val twoPiFOverSr = (2.0 * PI * freqHz / sampleRate).toFloat()

        for (i in 0 until totalSamples) {
            val sine = sin((twoPiFOverSr * i).toDouble()).toFloat()
            val env = when {
                i < attackSamples ->
                    i.toFloat() / attackSamples
                i >= totalSamples - releaseSamples -> {
                    val t = (i - (totalSamples - releaseSamples)).toFloat() / releaseSamples
                    exp((-4.0 * t).toDouble()).toFloat()
                }
                else -> 1.0f
            }
            pcm[i] = (sine * env * volume * Short.MAX_VALUE)
                .toInt()
                .coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                .toShort()
        }

        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setTransferMode(AudioTrack.MODE_STATIC)
            .setBufferSizeInBytes(pcm.size * 2)
            .build()

        track.write(pcm, 0, pcm.size)
        track.play()

        handler.postDelayed({
            track.stop()
            track.release()
        }, (durationMs + 120).toLong())
    }

    fun release() {
        handler.removeCallbacksAndMessages(null)
    }

    fun close() {
        release()
    }
}