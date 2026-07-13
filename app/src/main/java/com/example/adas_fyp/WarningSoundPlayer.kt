package com.example.adas_fyp

import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper

class WarningSoundPlayer {

    private val toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 20)
    private val handler = Handler(Looper.getMainLooper())

    fun playForMessage(message: String) {
        when {
            message.contains("BRAKE", ignoreCase = true) -> playFcwWarning()
            message.contains("LANE", ignoreCase = true) -> playLdwWarning()
            message.contains("DROWSINESS", ignoreCase = true) -> playDmsWarning()
            message.contains("FRONT VEHICLE", ignoreCase = true) -> playLvsaWarning()
            else -> playDefaultWarning()
        }
    }

    private fun playFcwWarning() {
        toneGenerator.startTone(
            ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD,
            700
        )
    }

    private fun playLdwWarning() {
        toneGenerator.startTone(
            ToneGenerator.TONE_PROP_BEEP,
            180
        )

        handler.postDelayed({
            toneGenerator.startTone(
                ToneGenerator.TONE_PROP_BEEP,
                180
            )
        }, 300)
    }

    private fun playDmsWarning() {
        toneGenerator.startTone(
            ToneGenerator.TONE_SUP_ERROR,
            900
        )
    }

    private fun playLvsaWarning() {
        toneGenerator.startTone(
            ToneGenerator.TONE_PROP_ACK,
            400
        )
    }

    private fun playDefaultWarning() {
        toneGenerator.startTone(
            ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD,
            500
        )
    }

    fun release() {
        handler.removeCallbacksAndMessages(null)
        toneGenerator.release()
    }

    fun close() {
        release()
    }
}