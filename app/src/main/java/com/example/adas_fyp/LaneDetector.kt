package com.example.adas_fyp

import androidx.camera.core.ImageProxy
import kotlin.math.abs

data class LaneDetectionResult(
    val leftDetected: Boolean,
    val rightDetected: Boolean,
    val lineCount: Int,
    val laneWarning: Boolean,
    val laneCenterX: Float? = null,
    val vehicleCenterX: Float = 0f,
    val offsetRatio: Float = 0f,
    val crossingDetected: Boolean = false,
    val complexRoadMarking: Boolean = false,
    // debug: everything needed to DRAW what the detector sees
    val stripTop: Float = 0f,
    val stripBottom: Float = 0f,
    val bandXs: FloatArray = FloatArray(0),
    val rawBandXs: FloatArray = FloatArray(0),
    val debugText: String = ""
)

/**
 * Lane departure detector, v5 -- 1D top-hat band extraction.
 *
 * WHY v4 FOUND NOTHING (the 16 Jul evening log: "Lines:0" or "Lines:1" for
 * the whole drive, Cross:true zero times, no warning through real crossings)
 *
 * v4 tried to remove hand-tuned constants by deriving thresholds from each
 * strip's own median + MAD. That was backwards. MAD measures the spread of
 * the profile, and lane markings ARE most of that spread -- so the more paint
 * was in the strip, the higher MAD rose, the higher the threshold rose, and
 * the more certainly the paint was rejected. The threshold was computed from
 * the thing it was supposed to detect.
 *
 * Measured on the actual recording's pixels:
 *     rows containing markings -> MAD 15-19  -> threshold 45-57 luma
 *     rows of plain asphalt    -> MAD  2-4   -> threshold  6-12 luma
 * No evening-light marking clears a 50-luma bar, so markings were invisible
 * and the only "bands" ever reported were noise on empty asphalt.
 *
 * WHAT v5 DOES INSTEAD
 * A 1D white top-hat along the profile:
 *     background = openings(profile) = runningMax(runningMin(profile, W), W)
 *     residual   = profile - background
 * with W (121 px) wider than any lane marking. The residual answers "how much
 * brighter is this than the road immediately beside it", so overall
 * brightness, shade, dusk and camera gain all cancel -- and, critically,
 * nothing in the threshold is derived from the markings themselves.
 *
 * The noise scale comes from the MEDIAN ABSOLUTE FIRST DIFFERENCE of the
 * residual. Paint bands are wide and smooth, so they contribute almost
 * nothing to |diff|; this measures true pixel-to-pixel noise.
 *
 * VERIFIED on the 16 Jul recording before being written (this is the step
 * that was skipped in v3/v4):
 *     noise floor        0.30 - 0.63 luma
 *     plain-road bands   8 - 16 luma
 *     real paint bands   27, 33, 54, 56, 58, 62, 66, 75, 90, 150 luma
 * Paint sits 3-10x above road residual across the whole drive.
 *
 * Running min/max use a monotonic deque, so the top-hat is O(n), not O(n*W).
 */
class LaneDetector {

    private val subRows = 4

    private val stripLeft = 0.10f
    private val stripRight = 0.90f

    // --- Top-hat band extraction ---------------------------------------------
    // Opening window must exceed the widest plausible marking, or it would eat
    // the band it is meant to isolate.
    private val tophatWindow = 121
    private val bandMinWidthPx = 3
    private val bandMaxWidthPx = 70

    // Threshold on the top-hat residual. The absolute floor dominates in
    // practice (noise*4 is ~1.2-2.5); it is kept well below the 27+ luma of
    // real paint and well above the 8-16 of road speckle.
    private val bandAbsThreshold = 10f
    private val bandNoiseMult = 4f

    // --- Vertical chaining ---------------------------------------------------
    private val chainMaxStepFrac = 0.06f
    private val chainMinRows = 3

    // --- Complex-paint suppression -------------------------------------------
    // RAW BAND COUNT IS A BAD SIGNAL and nearly killed LDW again.
    // complexTotalBands was 12, chosen while v4 was effectively blind and
    // reported 0-1 bands. Once the top-hat started working, an ORDINARY road
    // (2 lane lines + tar seams + cracks + specks, summed over 4 sub-rows)
    // reports 14-24 raw bands -- so Cplx fired on 83% of frames, tracks were
    // permanently coasted, and Cross:true never happened.
    //
    // Raw bands count noise. CHAINS -- bands that line up vertically across
    // sub-rows -- are the meaningful unit: a normal road has 2-3 (left line,
    // right line, maybe a kerb), while a yellow-box crosshatch or bump
    // chevrons produce many. Chain count is now the primary criterion; the
    // raw count is only a last-resort net for pathological frames.
    // Raw band count REMOVED as a complex trigger: on the 20 Jul concrete
    // highway an ordinary road produced 24-55 raw bands (expansion joints,
    // texture, worn patches), so Cplx latched true on nearly every frame and
    // zeroed LDW for the whole drive. Only CHAINS (bands that line up
    // vertically across sub-rows = actual painted stripes) indicate a
    // yellow-box / crosshatch. A normal road has 2-3 chains; a crosshatch many.
    private val complexTotalBands = 999      // effectively disabled
    private val complexChainCount = 6
    private val complexBrightFraction = 0.42f
    private val complexHoldFrames = 3
    private var complexHoldCounter = 0

    // --- Temporal tracking ---------------------------------------------------
    private val trackAssocMaxFrac = 0.05f
    private val trackEma = 0.5f
    private val trackMaxMiss = 3
    private val trackMaxCount = 8
    private val minTrackAgeForWarning = 2

    // --- Departure decision ---------------------------------------------------
    //
    // "ANY BAND NEAR THE MIDDLE OF THE IMAGE = CROSSING" WAS THE WRONG QUESTION.
    // On the 17 Jul highway run it warned while the car was demonstrably
    // centred:
    //     321.5s  L:true R:true  Offset:0.01  Cross:true
    //     345.0s  L:true R:true  Offset:0.00  Cross:true
    // Both real lane lines were visible in their normal outboard positions --
    // the car was in the middle of its lane -- but a tar seam / crack sealant /
    // the worn strip between the wheel tracks produced a band inside the
    // 0.38..0.62 window and that alone was called a departure.
    //
    // A lane departure is not "something is painted near the middle of the
    // frame". It is "the car has moved too close to one of ITS OWN lane
    // lines". So the criterion is now the standard normalised lane position:
    //
    //     p = (imageCentre - laneCentre) / (laneWidth / 2)
    //
    //     p = 0   centred        p = +/-1  camera is on a line
    //
    // p is scale-free: it needs no camera height, no pitch, no metres, and it
    // self-calibrates to whatever the lane width happens to be.
    // Lowered from 0.75: on the 19 Jul drive real lane drifts peaked at
    // p = 0.67-0.70 and NEVER reached 0.75, so no crossing ever warned. The
    // rate-limit below is also relaxed so a genuine drift is not smoothed away
    // before it reaches this line. 0.58 = camera ~58% of the way from lane
    // centre to the line; combined with the hit counter in MainActivity this
    // still rejects in-lane wobble (which stays under ~0.3).
    private val departureRatio = 0.58f

    // A lane line is bright paint. Tar seams and sealant are much weaker: the
    // top-hat residual measured on real road was 27-150 for paint and 8-16 for
    // road speckle, so only strong bands are allowed to act as lane lines.
    private val laneLineMinStrength = 18f

    // Sanity on the bracketing pair, as a fraction of frame width. A pair
    // narrower than this is two seams; wider is two different lanes.
    private val laneWidthMinFrac = 0.25f
    private val laneWidthMaxFrac = 1.10f
    private val laneWidthTolerance = 0.35f

    // Running estimate of this road's lane width (px), so a single line is
    // still usable and rogue pairs can be rejected.
    private var laneWidthPx = -1f
    private val laneWidthEma = 0.9f

    // The running width is only trustworthy after several CONSISTENT pairs.
    // The 18 Jul false alarm came from single-line frames using a bad width to
    // guess the lane centre, so p leapt to -0.97 the instant the second line
    // appeared with the true (smaller) width. Until the width has settled, a
    // lone line does NOT drive a warning.
    private var laneWidthConfidence = 0
    private val laneWidthConfidenceForSingle = 6

    // Temporal smoothing of p. A lane departure is a smooth, monotonic drift;
    // p jumping -0.43 -> -0.97 -> -0.36 between frames is a measurement
    // artefact (which lines were visible), not the car moving. p is rate-
    // limited so no single frame can slam it to the warn threshold, and the
    // warn must then hold for several consecutive frames of the SAME sign.
    private var smoothedP = 0f
    private var havePrevP = false
    private val pMaxStepPerFrame = 0.45f
    private val pSmoothEma = 0.4f

    // --- Centered-band ("straddle") detector -- the user's suggestion --------
    //
    // The bracket-both-lines method returns p:0.00 whenever the road doesn't
    // give two clean lines at once, which on the 20 Jul concrete highways was
    // almost always. This second path needs only ONE line: the one passing
    // under the car. A STRONG (paint-level) band that sits in the centre of the
    // strip for a sustained time means we are straddling a marking.
    //
    // False-alarm control (the user's own concern):
    //   - STRONG only: tar seams / sealant (residual 8-16) are rejected; paint
    //     is 27-150. laneLineMinStrength already encodes this.
    //   - CENTRE band only: a line at the strip edge is a normal lane boundary.
    //   - must PERSIST centred for straddleHoldFrames (~0.7s): rejects specks,
    //     shadows and single-frame noise.
    //   - a band that has been centred CONTINUOUSLY for too long is treated as
    //     a permanent feature (a seam that runs down the centre) and is NOT
    //     warned repeatedly: it fires ONCE on arrival, then must clear the
    //     centre before it can fire again.
    private val straddleCenterMinFrac = 0.42f   // strip-relative
    private val straddleCenterMaxFrac = 0.58f
    private val straddleHoldFrames = 3          // ~0.5s at ~6-7 Hz LDW
    private val straddleReleaseFrames = 3       // centre must clear this long
    private var straddleFrames = 0
    private var straddleClearFrames = straddleReleaseFrames
    private var straddleLatched = false

    private class BandTrack(
        var x: Float,
        var strength: Float,
        var age: Int = 1,
        var miss: Int = 0
    )

    private val tracks = mutableListOf<BandTrack>()

    // Reusable buffers: LDW runs ~7x/s, so avoid re-allocating each call.
    private var bufProfile = FloatArray(0)
    private var bufResidual = FloatArray(0)
    private var bufMin = FloatArray(0)
    private var bufBg = FloatArray(0)
    private var bufDeque = IntArray(0)

    fun detect(
        imageProxy: ImageProxy,
        stripTopF: Float,
        stripBottomF: Float
    ): LaneDetectionResult {
        val width = imageProxy.width
        val height = imageProxy.height
        val plane = imageProxy.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val limit = buffer.limit()

        val y0 = (height * stripTopF).toInt().coerceIn(0, height - subRows - 1)
        val y1 = (height * stripBottomF).toInt().coerceIn(y0 + subRows, height)
        val x0 = (width * stripLeft).toInt().coerceIn(0, width - 16)
        val x1 = (width * stripRight).toInt().coerceIn(x0 + 16, width)
        val stripW = x1 - x0
        val stripH = y1 - y0

        ensureBuffers(stripW)

        val rowBands = ArrayList<List<Band>>(subRows)
        var totalBands = 0
        var transverseBar = false
        val allRaw = ArrayList<Float>()

        for (k in 0 until subRows) {
            val r0 = y0 + k * stripH / subRows
            val r1 = y0 + ((k + 1) * stripH / subRows).coerceAtMost(stripH)

            // Column means over this sub-row.
            for (c in 0 until stripW) {
                var sum = 0
                var n = 0
                for (y in r0 until r1) {
                    val p = y * rowStride + (x0 + c) * pixelStride
                    if (p in 0 until limit) {
                        sum += buffer.get(p).toInt() and 0xFF
                        n++
                    }
                }
                bufProfile[c] = if (n > 0) sum.toFloat() / n else 0f
            }

            topHat(stripW)

            val noise = noiseScale(stripW)
            val threshold = maxOf(bandAbsThreshold, bandNoiseMult * noise)
            val bands = findBands(stripW, threshold)

            rowBands.add(bands)
            totalBands += bands.size
            for (b in bands) allRaw.add((x0 + b.x) / width)

            if (brightFraction(stripW, threshold) >= complexBrightFraction) {
                transverseBar = true
            }
        }

        val chains = buildChains(rowBands, width)

        val complexNow =
            totalBands >= complexTotalBands ||
                    chains.size >= complexChainCount ||
                    transverseBar

        if (complexNow) complexHoldCounter = complexHoldFrames
        else if (complexHoldCounter > 0) complexHoldCounter--

        val complexActive = complexNow || complexHoldCounter > 0

        if (complexActive) coastTracks() else updateTracks(chains, width)

        // --- Decision: normalised lane position -------------------------------
        //
        // Bracket the image centre with the nearest CONFIRMED, BRIGHT bands on
        // each side. Those are the car's own lane lines. Everything between
        // them (tar seams, sealant, worn wheel-track strips) is ignored -- it
        // was exactly that debris in the 0.38..0.62 window that produced the
        // 17 Jul false alarms while the car sat centred with Offset:0.01.
        val vehicleCenterX = width / 2f
        val stripCentreX = vehicleCenterX - x0   // centre, in strip coords

        var leftLine: BandTrack? = null
        var rightLine: BandTrack? = null
        val confirmed = ArrayList<Float>()

        for (t in tracks) {
            if (t.age < minTrackAgeForWarning) continue

            confirmed.add((x0 + t.x) / width)

            if (t.strength < laneLineMinStrength) continue

            if (t.x <= stripCentreX) {
                if (leftLine == null || t.x > leftLine!!.x) leftLine = t
            } else {
                if (rightLine == null || t.x < rightLine!!.x) rightLine = t
            }
        }

        // Maintain a running lane width from good pairs, so a single visible
        // line is still usable and implausible pairs can be rejected.
        var pairWidth = -1f

        if (leftLine != null && rightLine != null) {
            val wpx = rightLine!!.x - leftLine!!.x
            val frac = wpx / width

            if (frac in laneWidthMinFrac..laneWidthMaxFrac) {
                val agrees = laneWidthPx <= 0f ||
                        abs(wpx - laneWidthPx) / laneWidthPx <= laneWidthTolerance

                if (agrees) {
                    pairWidth = wpx
                    laneWidthPx = if (laneWidthPx <= 0f) wpx
                    else laneWidthEma * laneWidthPx + (1f - laneWidthEma) * wpx
                    if (laneWidthConfidence < laneWidthConfidenceForSingle) {
                        laneWidthConfidence++
                    }
                } else {
                    // A pair that disagrees with the running width is evidence
                    // the width is unreliable; decay confidence so the single-
                    // line fallback stops trusting it.
                    if (laneWidthConfidence > 0) laneWidthConfidence--
                }
            }
        }

        // p = 0 centred, p = +/-1 camera sitting on a line.
        var p = 0f
        var havePosition = false

        if (pairWidth > 0f) {
            val laneCentre = (leftLine!!.x + rightLine!!.x) / 2f
            p = (stripCentreX - laneCentre) / (pairWidth / 2f)
            havePosition = true
        } else if (
            laneWidthPx > 0f &&
            laneWidthConfidence >= laneWidthConfidenceForSingle &&
            (leftLine != null || rightLine != null)
        ) {
            // One line visible: infer the lane centre from the known width --
            // but only once that width has been confirmed by several
            // consistent pairs, so a stale/wrong width can no longer
            // manufacture a departure.
            val half = laneWidthPx / 2f
            val laneCentre = if (leftLine != null) leftLine!!.x + half else rightLine!!.x - half
            p = (stripCentreX - laneCentre) / half
            havePosition = true
        }

        // Rate-limit then smooth p, so a one-frame artefact cannot reach the
        // warn threshold. When we have no position this frame, decay toward 0
        // rather than holding a stale value.
        if (havePosition) {
            if (!havePrevP) {
                smoothedP = p
                havePrevP = true
            } else {
                val step = (p - smoothedP).coerceIn(-pMaxStepPerFrame, pMaxStepPerFrame)
                val limited = smoothedP + step
                smoothedP = pSmoothEma * smoothedP + (1f - pSmoothEma) * limited
            }
        } else {
            smoothedP *= 0.6f
            if (abs(smoothedP) < 0.05f) havePrevP = false
        }

        // --- Centered-band straddle evaluation --------------------------------
        // Is there a STRONG confirmed band sitting in the centre of the strip
        // right now?
        var strongCenterBand = false
        for (t in tracks) {
            if (t.age < minTrackAgeForWarning) continue
            if (t.strength < laneLineMinStrength) continue
            val fx = t.x / stripW.toFloat()
            if (fx in straddleCenterMinFrac..straddleCenterMaxFrac) {
                strongCenterBand = true
                break
            }
        }

        if (strongCenterBand) {
            straddleFrames++
            straddleClearFrames = 0
        } else {
            straddleClearFrames++
            if (straddleClearFrames >= straddleReleaseFrames) {
                straddleFrames = 0
                straddleLatched = false   // centre cleared: re-arm
            }
        }

        // Fires ONCE when a band has been centred long enough, then latches so
        // a marking that stays centred (e.g. a seam that slipped the strength
        // filter, or genuinely riding the line) does not spam warnings until it
        // clears and returns.
        val straddleDeparture =
            straddleFrames >= straddleHoldFrames && !straddleLatched
        if (straddleDeparture) straddleLatched = true

        val departing = havePosition && abs(smoothedP) >= departureRatio
        // The p-based path is guarded by complexActive (crosshatch fools the
        // bracket method). The STRADDLE path is NOT: a strong, persistent,
        // centred paint band under the car is a real line-crossing even on a
        // textured/complex road, and this is the only method that works when
        // the road gives just one worn line. It has its own strength+persistence
        // guards against false positives.
        val crossingDetected = (departing && !complexActive) || straddleDeparture

        val laneCenterX: Float? =
            if (havePosition && pairWidth > 0f) x0 + (leftLine!!.x + rightLine!!.x) / 2f else null

        val leftDetected = leftLine != null
        val rightDetected = rightLine != null

        return LaneDetectionResult(
            leftDetected = leftDetected,
            rightDetected = rightDetected,
            lineCount = totalBands,
            laneWarning = crossingDetected,
            laneCenterX = laneCenterX,
            vehicleCenterX = vehicleCenterX,
            // offsetRatio reports the SMOOTHED lane position p:
            // 0 = centred, +/-1 = on a line. Smoothed so a single-frame
            // line-visibility change cannot look like a swerve.
            offsetRatio = smoothedP,
            crossingDetected = crossingDetected,
            complexRoadMarking = complexActive,
            stripTop = stripTopF,
            stripBottom = stripBottomF,
            bandXs = confirmed.toFloatArray(),
            rawBandXs = allRaw.toFloatArray(),
            debugText = "raw:$totalBands chain:${chains.size} trk:${tracks.size} " +
                    "str:$straddleFrames " +
                    "lw:${if (laneWidthPx > 0f) String.format("%.0f", laneWidthPx) else "-"}"
        )
    }

    // ------------------------------------------------------------------------
    private class Band(val x: Float, val peak: Float)

    private fun ensureBuffers(n: Int) {
        if (bufProfile.size != n) {
            bufProfile = FloatArray(n)
            bufResidual = FloatArray(n)
            bufMin = FloatArray(n)
            bufBg = FloatArray(n)
            bufDeque = IntArray(n)
        }
    }

    /**
     * residual = profile - opening(profile), where opening = runningMax(runningMin).
     * Both passes use a monotonic deque, so this is O(n) rather than O(n*window).
     */
    private fun topHat(n: Int) {
        slidingMin(bufProfile, bufMin, n, tophatWindow)
        slidingMax(bufMin, bufBg, n, tophatWindow)
        for (i in 0 until n) bufResidual[i] = bufProfile[i] - bufBg[i]
    }

    private fun slidingMin(src: FloatArray, dst: FloatArray, n: Int, w: Int) {
        val half = w / 2
        var head = 0
        var tail = 0
        var next = 0

        for (i in 0 until n) {
            val hi = (i + half).coerceAtMost(n - 1)

            while (next <= hi) {
                while (tail > head && src[bufDeque[tail - 1]] >= src[next]) tail--
                bufDeque[tail++] = next
                next++
            }

            val lo = i - half
            while (bufDeque[head] < lo) head++

            dst[i] = src[bufDeque[head]]
        }
    }

    private fun slidingMax(src: FloatArray, dst: FloatArray, n: Int, w: Int) {
        val half = w / 2
        var head = 0
        var tail = 0
        var next = 0

        for (i in 0 until n) {
            val hi = (i + half).coerceAtMost(n - 1)

            while (next <= hi) {
                while (tail > head && src[bufDeque[tail - 1]] <= src[next]) tail--
                bufDeque[tail++] = next
                next++
            }

            val lo = i - half
            while (bufDeque[head] < lo) head++

            dst[i] = src[bufDeque[head]]
        }
    }

    /**
     * Median |first difference| of the residual. Paint bands are wide and
     * smooth so they barely contribute; this is the real pixel noise, not the
     * feature amplitude that misled v4.
     */
    private fun noiseScale(n: Int): Float {
        if (n < 8) return 1f

        val d = FloatArray(n - 1)
        for (i in 0 until n - 1) d[i] = abs(bufResidual[i + 1] - bufResidual[i])
        d.sort()

        return d[d.size / 2].coerceAtLeast(0.3f)
    }

    private fun findBands(n: Int, threshold: Float): List<Band> {
        val out = ArrayList<Band>()
        var i = 0

        while (i < n) {
            if (bufResidual[i] > threshold) {
                var j = i
                var peak = 0f
                while (j < n && bufResidual[j] > threshold) {
                    if (bufResidual[j] > peak) peak = bufResidual[j]
                    j++
                }

                val w = j - i
                if (w in bandMinWidthPx..bandMaxWidthPx) {
                    out.add(Band(i + w / 2f, peak))
                }
                i = j
            } else {
                i++
            }
        }

        return out
    }

    private fun brightFraction(n: Int, threshold: Float): Float {
        var bright = 0
        for (i in 0 until n) if (bufResidual[i] > threshold) bright++
        return bright.toFloat() / n
    }

    private class Chain(val x: Float, val strength: Float)

    private fun buildChains(rowBands: List<List<Band>>, width: Int): List<Chain> {
        val chains = ArrayList<Chain>()
        val maxStep = width * chainMaxStepFrac

        for (seed in rowBands[subRows - 1]) {
            var x = seed.x
            var sum = seed.x
            var count = 1
            var weakest = seed.peak

            for (k in subRows - 2 downTo 0) {
                var best: Band? = null
                var bestDist = Float.MAX_VALUE

                for (b in rowBands[k]) {
                    val d = abs(b.x - x)
                    if (d <= maxStep && d < bestDist) {
                        best = b
                        bestDist = d
                    }
                }

                if (best == null) break

                x = best.x
                sum += best.x
                count++
                if (best.peak < weakest) weakest = best.peak
            }

            // Strength = the weakest link, so a chain only counts as bright
            // paint if it is bright the whole way up the strip.
            if (count >= chainMinRows) chains.add(Chain(sum / count, weakest))
        }

        return chains
    }

    private fun updateTracks(chains: List<Chain>, width: Int) {
        val used = BooleanArray(chains.size)
        val maxAssoc = width * trackAssocMaxFrac

        for (t in tracks) {
            var bestIdx = -1
            var bestDist = Float.MAX_VALUE

            for (i in chains.indices) {
                if (used[i]) continue
                val d = abs(chains[i].x - t.x)
                if (d <= maxAssoc && d < bestDist) {
                    bestIdx = i
                    bestDist = d
                }
            }

            if (bestIdx >= 0) {
                used[bestIdx] = true
                t.x = trackEma * t.x + (1f - trackEma) * chains[bestIdx].x
                t.strength = trackEma * t.strength + (1f - trackEma) * chains[bestIdx].strength
                t.age++
                t.miss = 0
            } else {
                t.miss++
            }
        }

        tracks.removeAll { it.miss > trackMaxMiss }

        for (i in chains.indices) {
            if (!used[i] && tracks.size < trackMaxCount) {
                tracks.add(BandTrack(chains[i].x, chains[i].strength))
            }
        }
    }

    private fun coastTracks() {
        for (t in tracks) t.miss++
        tracks.removeAll { it.miss > trackMaxMiss }
    }

    fun reset() {
        tracks.clear()
        complexHoldCounter = 0
        laneWidthPx = -1f
        laneWidthConfidence = 0
        smoothedP = 0f
        havePrevP = false
        straddleFrames = 0
        straddleClearFrames = straddleReleaseFrames
        straddleLatched = false
    }
}