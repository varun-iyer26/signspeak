# Aligning a frame-by-frame ASL keypoint dataset with a 5-flex + IMU glove

*Research note for the DubHacks 2026 glove build. Written 2026-10-17. The reference code is in [`research/hand_state.py`](../../research/hand_state.py), and its synthetic self-checks are in [`research/test_hand_state.py`](../../research/test_hand_state.py).*

**Hardware assumed:** each glove has an Arduino Nano 33 BLE Rev2 (BMI270 accelerometer and gyro, BMM150 magnetometer), 5 × SparkFun 2.2" flex sensors in 47 kΩ dividers, and copper-tape fingertip pads. The firmware samples at 100 Hz and runs Madgwick fusion. **Dataset assumed:** per-frame hand keypoints, most likely MediaPipe's 21 landmarks, covering most of the ASL vocabulary.

> Markers used below: **[est.]** is our own engineering estimate, not a published number. **[unverified]** is a claim we could not confirm from a primary source.

---

## Executive summary

1. **Convert both sides into one shared "hand state", instead of converting one side into the other.** Each frame becomes a 12-number vector per hand:
   - 5 finger bends: MCP + PIP angle sum, which is what a back-of-finger flex sensor measures;
   - the gravity direction in the hand frame (3 numbers);
   - the angular velocity in the hand frame (3 numbers);
   - the yaw since the start of the sign (1 number).

   Keypoints produce all of these with geometry. The glove produces them through a short per-wearer calibration. All four groups are immune to IMU drift except yaw, and yaw only has to hold for one sign.
2. **Nobody has published flex-sensor synthesis from sign video.** That is a gap, so the plan below combines established parts. The closest prior work all converts video into *IMU* signals: IMUTube, SignRing, Vi2IMU, and Leite et al. 2024. Their shared lesson: **training only on virtual data loses 5–50% of performance, and a small amount of real data recovers most of it.** Examples:
   - SignRing (934 ASL signs): 8.3% WER on real data alone, 6.3% WER on 50% virtual + 50% real.
   - Leite 2024: F1 of 0.44–0.62 on virtual data alone, about 0.75 with 10% real data.
3. **Calibrate the glove into "MediaPipe's view of the hand", not true anatomy.** MediaPipe finger angles have large errors against motion capture: 10–22° RMSE, and worse at the PIP joints. If the dataset was also made with MediaPipe, regressing the flex readings onto MediaPipe angles cancels MediaPipe's systematic bias rather than adding it.
4. **For the 20-hour build: first ship a small-vocabulary demo, then extend it.**
   - **Ship first:** a 1-nearest-neighbour DTW recognizer on templates the team records with the glove itself, using 20–40 signs. Pick those signs with the dataset-driven confusability ranking described below.
   - **Stretch goal:** train a small GRU/1D-CNN on the *converted* dataset with glove-style augmentation, fine-tune it on 3–5 real repetitions per sign, and output top-5 candidates. Claude (already in this repo) then chooses the sentence.
5. **Some signs are physically out of reach for 5 flex sensors and 1 IMU per hand.**
   - **Location-only pairs:** MOTHER/FATHER, SUMMER/UGLY/DRY, APPLE/ONION.
   - **Spread-only and crossing contrasts:** U/V/R, B/4.
   - **Thumb-placement letters:** A/S/T/M/N/E.
   - **Anything carried by the face.**

   The cheapest fixes:
   - **A webcam** (free; you already have MediaPipe) for location relative to the face.
   - **Contact pads:** thumb-to-fingertip pads, plus one pad on the side of the index finger and one on the side of the middle finger.
   - **A second IMU on the forearm** for coarse hand height.

   These additions can be ranked before building them: compute each one from the dataset's keypoints and count how many confusable pairs it separates.
6. **Data to collect per wearer per session: about 5 minutes.**
   - 10 s of flat hand ↔ fist for the flex gain/offset.
   - 2 static poses for R_mount.
   - 1–2 min of paired glove + webcam free motion for the full flex fit, R_mount refinement and time lag.
   - 3–5 repetitions of each demo sign for templates and fine-tuning.

   The flex fit must be **redone every time the glove is put on**: sensor position shifts, and Spectra Symbol sensors creep (see §3.1).

---

## 1. Recommended pipeline

```
                      ┌──────────── OFFLINE (dataset) ─────────────┐
 ASL dataset          │                                            │
 21 kp / frame ──► canonicalize ──► hand frame R(t) ──► hand state │
 (2D/3D, ~30 fps)  (mirror left,    joint angles θ      s_kp(t)    │
                    fix aspect)                          (12/hand) │
                      │                                    │       │
                      │                     glove-style    ▼       │
                      │                     augmentation ─► train GRU / build templates
                      └────────────────────────────────────┬───────┘
                                                           │ weights / templates
            ┌──────── CALIBRATION (per wearer, ~5 min) ────┼────────┐
            │  webcam + MediaPipe ──► s_kp(t) ─┐           │        │
            │                                  ├─► lag τ (xcorr |ω|)│
            │  glove (100 Hz) ─────► raw(t) ───┘   R_mount (Wahba)  │
            │                                     flex fit f_i(R)   │
            └──────────────────────────────┬──────────────┼─────────┘
                                           ▼              ▼
 LIVE   glove 100 Hz ─► divider inverse ─► f_i(R) ─► s_gl(t) ─► segment ─► time-normalize (32 fr)
        (flex, acc, gyro, q)  R_mount, bias                                       │
                                                                                  ▼
                                     DTW 1-NN (handshape prefilter) or GRU ─► top-k ─► Claude ─► speech
```

### 1.1 The shared hand state (per hand, per frame)

| Index | Name | Units | Keypoints → | Glove → |
|---|---|---|---|---|
| 0–4 | `bend` (thumb…pinky) | rad | θ_MCP + θ_PIP (thumb: θ_MCP + θ_IP) | f_i(R_i), a calibrated polynomial of resistance |
| 5–7 | `gravity` (unit, pointing down, hand frame) | – | Rᵀ · g_cam | Rₘ · R(q)ᵀ · (0,0,−1), or −Rₘ·LPF(accel) |
| 8–10 | `omega` (hand frame) | rad/s | log(R_kᵀ R_{k+1}) / Δt | Rₘ · (gyro − bias) |
| 11 | `yaw` since sign start | rad | twist of R_k R_0ᵀ about the vertical | twist of R(q_k)R(q_0)ᵀ about the vertical |

Hand frame for a canonical **right** hand. Left hands are mirrored into right hands first.
- origin: the wrist (landmark 0);
- **+y**: wrist → middle MCP (landmark 9);
- **+x**: pinky MCP (17) → index MCP (5), orthogonalized against y;
- **+z** = x × y: points **out of the palm**, the side the fingers curl toward.

### 1.2 Keypoints → hand state (step by step)

Landmarks: `p[0..20]`, with indices 0 wrist, 1–4 thumb, 5–8 index, 9–12 middle, 13–16 ring, 17–20 pinky. **Step 0: make the coordinates isotropic.**
- **MediaPipe world landmarks (meters):** already isotropic; use them as-is.
- **Normalized image landmarks:** multiply y by H/W.
- **Unknown frame size** (for example Kaggle parquet files): search for the aspect ratio that keeps bone lengths most constant across frames. This is `fix_aspect()`; it recovers 1.6 to within 0.05 on the synthetic test.

**Step 1: canonicalize handedness.** If the *physical* hand is left, or the image is mirrored, but not both, set `x ← −x`. Do this *before* any cross product, because a reflection flips the sign of cross products.

MediaPipe's handedness label assumes a mirrored, selfie-style image ([legacy Hands docs](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md)). On un-mirrored video, swap Left and Right. Settle this once per dataset by checking a few frames by eye.

**Step 2: build the hand frame.**
```
y = unit(p9 − p0)
x = unit((p5 − p17) − ((p5 − p17)·y) y)
z = x × y
R = [x y z]          # camera ← hand;   v_hand = Rᵀ v_cam
```

**Step 3: compute joint flexion angles.** For each finger chain (0 → MCP → PIP → DIP → tip), take consecutive bone vectors u and v.
- **PIP and DIP:** θ = arccos(û·v̂).
- **MCP:** use *signed* flexion in the finger's flexion plane, so that spread does not register as bend:
  ```
  u⊥ = u − (u·x)x,  v⊥ = v − (v·x)x
  θ_MCP = atan2((u⊥ × v⊥)·x,  u⊥·v⊥)     # >0 = toward the palm, <0 = hyperextension
  ```
- **Thumb:** the chain is 1 → 2 → 3 → 4, which gives θ_MCP at 2 and θ_IP at 3, both unsigned. The thumb CMC (opposition and abduction) is invisible to a single dorsal flex sensor; §5 covers the consequences.

**Step 4: flex-equivalent bend.** `bend_f = θ_MCP + θ_PIP` (thumb: `θ_MCP + θ_IP`). The reasoning:
- A resistive bend sensor responds to curvature along its length.
- Curvature integrated along the sensor equals the total change in direction between its two ends.
- If the sensor spans several joints, that total is the **sum of the spanned joint angles**.
- The SparkFun 2.2" sensor's active length is 55.4 mm ([datasheet](https://cdn.sparkfun.com/assets/9/5/b/f/7/FLEX_SENSOR_-_SPECIAL_EDITION_DATA_SHEET_v2019__Rev_A_.pdf)). Starting over the metacarpal head, it spans the MCP and PIP joints of an adult index finger, and ends near or before the DIP.

The "sum of angles" model is physics-motivated and consistent with Pratap et al. 2024, but we found no paper that validates it formally **[unverified as a formal model]**. That is why the calibration below fits the response rather than assuming it. Joint geometry also gives you a free check: DIP flexion is roughly ⅔ of PIP flexion in natural motion, so leaving it out loses little information.

**Step 5: gravity in the hand frame.** `g_hand = Rᵀ g_cam`. Here `g_cam` is the camera-frame down direction, `(0, 1, 0)` for an upright, level camera (image y points down).
- If the dataset includes body pose (Kaggle data is MediaPipe Holistic), estimate camera roll from the shoulder line and rotate `g_cam` to match.
- If you record with a phone, read its accelerometer for `g_cam`.

**Step 6: angular velocity.**
```
ω_k = log( R_{k−1}ᵀ R_{k+1} )^∨ / (t_{k+1} − t_{k−1})      # body (hand) frame, rad/s
```
This is exactly what a gyro rigidly fixed to the hand measures once rotated into the hand frame. Because it is a derivative, it amplifies landmark jitter: about 2° of per-frame orientation noise at 30 fps becomes roughly 1 rad/s of noise **[est.]**. So:
- smooth the landmarks first, with a 5-frame moving average or Savitzky–Golay;
- give `omega` a lower weight in distances.

**Step 7: yaw since the start of the sign.** The world-frame delta rotation is `R_k R_0ᵀ`. Take its twist angle about the vertical (swing–twist decomposition) so that tilting the hand does not leak into yaw. Video has no drift at all. On the glove, gyro drift over a 1–3 s sign is a few degrees once the gyro bias is calibrated.

### 1.3 Glove → hand state (step by step)

**Step 1: undo the voltage divider.** With the flex sensor on top (Vcc → flex → Aₙ → 47 k → GND):
```
V/Vcc = 47k / (47k + R)   ⇒   R = 47k · (ADC_max / ADC − 1)
```
SparkFun's hookup guide gives about 30 kΩ flat and about 70 kΩ at 90°, so the divider spans roughly 0.61 → 0.40 of Vcc. The divider output is nonlinear in R; resistance itself is close to linear in angle. **Fit in R, not in ADC counts.** Set `analogReadResolution(12)` on the Nano 33 BLE to get 12-bit readings.

**Step 2: flex calibration.** `bend_f = c₀ + c₁R + c₂R²` per finger (`fit_flex_calibration`). Fit it on paired data, described in §3. Clip predicted bend at ≥ 0: the sensor barely responds to backward bending ([SparkFun](https://www.sparkfun.com/flex-sensor-2-2.html)).

**Step 3: rotate the IMU readings into the hand frame.**
```
ω_hand   = Rₘ (gyro − bias)
g_hand   = Rₘ R(q)ᵀ (0, 0, −1)        # tilt from Madgwick; accel-corrected, so drift-free
           (fallback: −unit(Rₘ · LPF(accel)), wrong during fast motion)
yaw_k    = twist( R(q_k)Rₘᵀ · (R(q_0)Rₘᵀ)ᵀ , about (0,0,−1) )
```
Madgwick's quaternion maps sensor → earth with earth z pointing up. Check this once: with the glove at rest, `R(q) · accel ≈ (0, 0, +1 g)`. Use the 6-axis (IMU) Madgwick update indoors. The BMM150 is easily disturbed near laptops and metal tables, and relative yaw does not need it.

**Step 4: mirror the left glove.** Its R_mount must target the left-hand frame: y toward the fingers, z out of the palm, x = y × z (toward the pinky). Then apply `mirror_state`, which uses M = diag(−1, 1, 1):
- `g ← M g`;
- `ω ← −M ω` (angular velocity is a pseudovector);
- `yaw ← −yaw`;
- bends are unchanged.

`test_left_glove_matches_left_keypoints` checks this end to end.

**Step 5: frame-rate matching.** Glove data arrives at 100 Hz and video at about 30 fps, with variable timing in a browser.
- Low-pass the glove signal with a moving average of about 3 samples, then interpolate it onto the video timestamps for calibration.
- For recognition, resample *each segmented sign* to a fixed 32 frames on both sides (`time_normalize`). After that step the original rates no longer matter.
- Always use timestamps, never frame indices.

### 1.4 Training and recognition

1. Convert every dataset clip to `s_kp(t)` and segment it into signs; most isolated-sign datasets are already one sign per clip. Time-normalize to 32 frames.
2. **Glove-style augmentation** (`simulate_glove`):
   - clip negative bend values;
   - apply a random per-finger gain of 0.85–1.15 (placement and fit vary);
   - add Gaussian noise;
   - add a yaw ramp to imitate drift;
   - optionally time-warp and speed-scale each clip.
3. Train the recognizer (§4), then fine-tune or enrol with real glove recordings run through the same `glove_to_state`.

---

## 2. Prior work: moving between video/pose and wearable sensors

### 2.1 Virtual IMU from video (well established)

| Work | What it does | Key result for us |
|---|---|---|
| **IMUTube**, Kwon et al., IMWUT 2020 ([arXiv 2006.05675](https://arxiv.org/abs/2006.05675)) | YouTube video → 2D/3D pose → virtual accelerometer → "distribution mapping" to match real IMU | Virtual-only training reaches 94–98% of real-only F1. Virtual+real beats real-only by up to 12%. Calibration converges with **< 100 s of real IMU**. Without distribution mapping, Opportunity F1 falls from 0.776 to 0.295. |
| Kwon et al., *Sensors* 2021 ([doi](https://doi.org/10.3390/s21248337)) | 41 h of virtual IMU trains large models, then calibrates on 36 min of real data | Large virtual corpora make bigger models trainable |
| Fortes Rey et al., UbiComp/ISWC '19 Adj., "Let there be IMU data" (pp. 699–708) | Monocular video → pose → simulated IMU | Early demonstration; numbers not retrieved |
| **Vi2IMU**, Santhalingam et al., IMWUT 2023 ([doi](https://doi.org/10.1145/3596261)) | **ASL videos** → wrist orientation from hand joints → smartwatch accelerometer + gyro | A recognizer trained only on IMU translated from ASL video performs "reasonably well" against one trained on real IMU (exact numbers not retrieved) |
| **SignRing**, Li et al., IMWUT 2023 ([doi](https://doi.org/10.1145/3610881)) | Two-view sign videos → 3D hand → virtual IMU at the ring positions | 934 signs: real only 8.3% WER, virtual only 14.7%, **50/50 mix 6.3%** |
| **Leite et al.**, *Sci. Rep.* 2024 ([doi](https://doi.org/10.1038/s41598-024-65069-2)) | MediaPipe Hands 3D → IK → simulated hand IMUs, and optimizes sensor placement | Simulated-only training tested on real: F1 0.62 / 0.44. With 90% simulated + 10% real (about 25 s per gesture): **0.76 / 0.75**. Simulation error 0.64 m/s² and 0.155 rad/s. |
| IMUGPT, Leng et al., 2023 ([arXiv 2305.03187](https://arxiv.org/abs/2305.03187)); IMUGPT 2.0 ([arXiv 2402.01049](https://arxiv.org/abs/2402.01049)) | Text → motion model → virtual accelerometer | Even text-generated motion helps when mixed with real data |
| Takeda et al., UbiComp/ISWC 2018 Adj. ([doi](https://doi.org/10.1145/3267305.3267509)); Xia & Sugiura, ISWC 2022 | Mocap → virtual accelerometer; spring-joint augmentation | Background |

**Gap:** we found **no published work that synthesizes flex or bend-sensor glove data from sign video or pose.** Leite 2024 is the closest, and it simulates only IMUs. This project would be new on that point. Every comparable system needed some real data from the target sensor, so plan for it.

### 2.2 Shared embeddings and contrastive learning (pose/video ↔ IMU)

- **IMU2CLIP**, Moon et al. ([arXiv 2210.14395](https://arxiv.org/abs/2210.14395)): aligns an IMU encoder to CLIP video/text embeddings using Ego4D.
- **ImageBind**, Girdhar et al., CVPR 2023 ([arXiv 2305.05665](https://arxiv.org/abs/2305.05665)): includes an IMU modality aligned with head-IMU and video pairs. C3T ([arXiv 2407.16803](https://arxiv.org/abs/2407.16803)) reports that this IMU encoder transfers poorly to third-person activity recognition.
- **COCOA** ([arXiv 2208.00467](https://arxiv.org/abs/2208.00467)) and **CroSSL** ([arXiv 2307.16847](https://arxiv.org/abs/2307.16847)): cross-modal self-supervised learning for sensor time series.
- **Tong, Ge, Lane**, IMWUT 2021 ([doi](https://doi.org/10.1145/3494995)): uses video action embeddings as the semantic space for zero-shot IMU activity recognition.

**Relevance for a hackathon:** low. All of these need many *paired* samples, from hours to thousands of clips. An explicit geometric hand state needs none, and stays interpretable and debuggable on a demo floor. A contrastive encoder is worth adding later, once real glove data has been collected over weeks: train glove-encoder ↔ keypoint-encoder with InfoNCE, then retrieve signs by nearest neighbour in the joint space.

### 2.3 Glove → hand pose, and glove calibration

- **Glauser et al.**, SIGGRAPH 2019 ([doi](https://doi.org/10.1145/3306346.3322957), [PDF](https://igl.ethz.ch/projects/stretch-glove/Stretch-Sensing-Glove-2019.pdf)): 44 stretch sensors.
  - Ground truth came from a depth-camera hand tracker; the thin glove does not disturb it. This is the same idea as webcam + MediaPipe here.
  - Mean error was 6.8°, against 10.5° for CyberGlove and 11.9° for Manus.
  - **A new user needs only per-sensor min/max normalization** with the generic model, reaching 7.6° mean error. This supports the 10 s flat ↔ fist session-recalibration step.
- **Wang & Neff**, SCA 2013: Gaussian-process regression from glove sensors to joint angles.
- **Kahlesz et al.**, CGI 2004: visual-fidelity glove calibration.
- **Kessler et al. 1995**: CyberGlove evaluation (Georgia Tech technical report).
- We found no peer-reviewed paper that uses **MediaPipe as ground truth to calibrate a flex glove**. The closest is Leite 2024, which goes from MediaPipe to simulated IMUs.

### 2.4 Glove sign-recognition baselines (calibrating expectations)

| System | Sensors | Vocabulary | Result |
|---|---|---|---|
| Gao et al., IJPRAI 2000 ([doi](https://doi.org/10.1142/S0218001400000386)) | 2 CyberGloves + Polhemus 3D trackers | 5,177 CSL signs | About 94.8% isolated (via the companion FG 2000 paper **[which paper reports 5,177 unverified]**) |
| Zhou et al. (UCLA), *Nat. Electron.* 2020 ([doi](https://doi.org/10.1038/s41928-020-0428-6)) | Yarn stretch sensors | 660 signs | 98.63% (4 signers) |
| **Lee & Lee**, *IEEE Sensors J.* 2018 ([doi](https://doi.org/10.1109/JSEN.2017.2779466)) | 5 flex + 2 pressure + IMU | ASL alphabet | **65.7% without pressure sensors → 98.2% with them** |
| **Faisal et al.** (DU), *Sci. Rep.* 2022 ([doi](https://doi.org/10.1038/s41598-022-25108-2)) | **5 × 2.2" flex + MPU-6050**, the same as ours | 40 ASL (24 static, 16 dynamic) | Leave-one-subject-out F1: 82% static, 97% dynamic |
| SignSpeak (Makkar et al.) ([arXiv 2407.12020](https://arxiv.org/abs/2407.12020)) | 5 flex, no IMU, 36 Hz | 36 (A–Z, 1–10) | GRU, about 92% |
| FinGTrAC, Liu et al., IMWUT 2020 ([doi](https://doi.org/10.1145/3414117)) | Ring IMU + smartwatch | 100 words in sentences | 94.2%, falling to about 90% at 200 words, *using sentence context* |
| Athitsos et al. 2010 ([PDF](https://www.cs.bu.edu/groups/ivc/pubs/Athitsos_cslt2010_a4.pdf)) | Video hand trajectories, DTW | 921 ASL classes | Only 66.8% have the correct sign in the top 20 |

What these numbers show:
- The DU glove is your hardware. Its static signs (handshape only) are harder than its dynamic ones, because motion adds information.
- Large vocabularies were only reached with **3D position trackers** (Gao) or dense custom sensors (UCLA).
- FinGTrAC shows that **sentence-level context** can carry a weak sensor to roughly 100–200 words. Claude can do this job here.

---

## 3. The math, sensor by sensor (pitfalls included)

### 3.1 Flex sensor ↔ finger angles

- **Which joints:** MCP + PIP; see §1.2, step 4. The thumb sensor sees thumb MCP + IP only, never CMC opposition.
- **Linear or nonlinear:**
  - Resistance vs angle is roughly monotone and mildly nonlinear. The nonlinearity is worst at small angles, and the response depends on bend radius and on where the sensor sits over the joint (Tor Vergata 2018; ICAT).
  - The divider output adds a second nonlinearity, which `flex_resistance()` removes.
  - A **quadratic in R** per finger is enough **[est.]**.
- **Spec spread:**
  - Flat resistance is 25 kΩ ± 30%.
  - The datasheet only guarantees "≥ 2× flat at a 180° pinch".
  - Measured values at 90° range from about 48 kΩ to about 70 kΩ across batches.
  - So **calibrate every sensor**.
- **Creep (important for static handshapes):**
  - Simone & Kamper 2005 ([doi](https://doi.org/10.1186/1743-0003-2-5)) measured a **Spectra Symbol sensor losing 31.8% of its step change in resistance within 30 s**.
  - Mitigations:
    - take the handshape from the *first ~300 ms* of a hold;
    - include temporal context in the features;
    - recalibrate every session.
- **Sliding:** a sensor in a loose fabric sleeve slides over the knuckle and drifts in ways that are hard to detect. Anchor the *base* (the pin end) firmly and let the tip slide in a pocket. Don't let the base flex ([SparkFun](https://www.sparkfun.com/flex-sensor-2-2.html)).
- **Hyperextension:** almost no response. Clip simulated bends at 0 (`simulate_glove`).

### 3.2 Palm orientation → gravity vector, compared with the accelerometer

- Keypoint side: `g_hand = Rᵀ g_cam`. Glove side: `g_hand = Rₘ R(q)ᵀ ĝ_down`.
- Both use a direction, not a full rotation, so **heading drops out**: only tilt (pitch and roll of the palm) is compared, and that is what the accelerometer measures without drift.
- Prefer Madgwick's tilt to raw accelerometer during signing: hand accelerations of 0.5–2 g corrupt raw accel **[est.]**.
- On the keypoint side, the palm normal from three landmarks (0, 5, 17) is the most reliable orientation cue MediaPipe gives. Fingers are noisier than the palm.

### 3.3 Angular velocity, compared with the gyro

- `ω = log(R_{k−1}ᵀ R_{k+1})^∨ / 2Δt`. The body-frame form `Rᵀ dR` is correct for a hand-mounted gyro. The world-frame form `dR Rᵀ` would be wrong here.
- The norm |ω| does not depend on the frame. **Time alignment can therefore use |ω_gyro| against |ω_keypoints| before R_mount is known.**
- Band-limit the gyro to match the keypoint smoothing before comparing the two.

### 3.4 Yaw drift

The design uses no absolute yaw. Only yaw relative to the start of the sign is kept, along with tilt (drift-free) and rate (bias-corrected).
- Calibrate the gyro bias at startup. The BMI270 zero-rate offset is about ±0.5 °/s typical, and users report about 3 °/s, so measure it.
- Re-zero the bias whenever the hand is still for more than 1 s.
- After calibration, drift is a few degrees over a 2 s sign. The `simulate_glove` yaw-ramp augmentation teaches the model to tolerate it.

### 3.5 IMU mounting offset R_mount

The board will not be aligned with the hand frame. Solve **Wahba's problem** with an SVD (Kabsch; `solve_rotation`):
```
Rₘ = argmin_R Σ wᵢ ‖bᵢ − R aᵢ‖²,   B = Σ wᵢ bᵢ aᵢᵀ = U S Vᵀ,   Rₘ = U diag(1, 1, det(UVᵀ)) Vᵀ
```
At least two non-parallel vector pairs are needed. Sources of pairs, from cheapest to most accurate:
- **Two known static poses (about 20 s total).**
  - Palm flat on a table, palm down: `a = −accel`, `b = (0, 0, 1)`, since down is out of the palm.
  - Arm hanging, fingers pointing at the floor: `a = −accel`, `b = (0, 1, 0)`.
  - Add a third pose (thumb up, palm facing left) for robustness.
- **Paired recording (best):** stack every frame's `(−accel or tilt, g_hand_kp)` and `(gyro, ω_kp)` pair. This also averages out how the wearer actually holds "flat".

`test_glove_matches_keypoints` recovers R_mount to within 1e-3.

### 3.6 Handedness and mirroring

- Canonicalize everything to a right hand: reflect keypoints in x before building the frame, and mirror glove states with `mirror_state`.
- Left-handed signers then share templates with right-handed ones. PopSign's baseline does the same, flipping left-handed videos.
- Two-handed signs: keep the dominant hand in slot 0 and the non-dominant hand in slot 1, with each canonicalized separately. The model then sees the non-dominant hand as a mirrored right hand, which is consistent across the dataset and the gloves.

### 3.7 Frame-rate mismatch (≈30 fps ↔ 100 Hz)

- Calibration: low-pass and interpolate the glove onto the webcam timestamps.
- Recognition: resample each sign to 32 frames (≈ 1 frame per 30–60 ms for a 1–2 s sign).
- Don't upsample video-derived ω to 100 Hz and compare it sample-for-sample with raw gyro: the keypoint signal has no content above about 5–8 Hz **[est.]**.

### 3.8 2D-only keypoints

In order of preference:
1. **If the dataset includes video, re-run MediaPipe HandLandmarker** and use `worldLandmarks`: meters, origin at the hand's geometric centre ([docs](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)).
2. **MediaPipe "2.5D" landmarks** (x, y plus relative z, which is what Kaggle and most exports contain): run `fix_aspect` and use them as 3D. Depth is the least reliable axis. Orientation is fine for gross palm direction; bend estimates are worse for fingers pointing at the camera.
3. **Truly 2D landmarks (for example OpenPose 2D; the same 21-point order):**
   - Orientation is ambiguous: the palm normal's sign cannot be recovered from a single 2D view.
   - `finger_straightness_2d` gives a chord/arc ratio per finger. It is monotone in curl but depends on viewing angle.
   - Learn a small regressor from it to `bend`, trained on frames where you also have MediaPipe 3D (your own webcam captures).
   - Or lift 2D → 3D with a small MLP trained on MediaPipe 3D. Expect noticeably worse orientation features **[est.]**.

---

## 4. Calibration procedure (paired glove + webcam)

The calibration recorder should save `result.worldLandmarks` and `handedness` per frame, together with `performance.now()` timestamps, in the same file as the glove stream. The repo's `js/hands.js` currently returns only normalized `landmarks`, so add this in a separate capture page; this note does not change the app. Stream the glove over BLE with the **Arduino's `micros()` timestamp in each packet**, because BLE delivery jitter is tens of milliseconds.

**Per session (each time the glove goes on), about 5 minutes:**

| Step | Time | Wearer does | Solves |
|---|---|---|---|
| 0 | 3 s | Hold still on the table | Gyro bias |
| 1 | 10 s | Flat hand ↔ tight fist, 5× | Per-finger min/max, which gives a quick linear flex map (the Glauser-style min/max normalization) |
| 2 | 20 s | Palm-down flat, fingers-down hang, thumb-up | R_mount (static Wahba) |
| 3 | 5 s | Three sharp wrist twists, facing the webcam | Time lag τ: cross-correlation of \|ω_gyro\| and \|ω_kp\| (`estimate_lag`). Seen from the front, a twist moves the palm a lot, so its keypoint ω is easy to detect. |
| 4 | 60–120 s | Free signing / fingerspelling / finger waves facing the webcam, slowly, at several palm orientations | Full quadratic flex fit, R_mount refinement (dynamic Wahba), and a check |
| 5 | per sign | 3–5 repetitions of each demo sign | Templates / fine-tuning data |

**Fitting details:**
- **Lag first:** use rotation-invariant signals (|ω| and Σ bend), so no other parameter is needed yet. Search ±1 s. Re-estimate it at the end of the recording too, to catch clock drift; crystals are about ±50 ppm, roughly 3 ms per minute **[est.]**, which is negligible for short recordings.
- **Flex regression:** `bend_kp,f(t) ≈ f_f(R_f(t − τ))`.
  - Use only frames where MediaPipe confidence is high and the finger is not pointing at the camera; reject frames where `|z|` of the finger bones dominates.
  - Use a robust (Huber) or plain least-squares quadratic.
  - MediaPipe joint angles have about 10–22° RMSE against mocap, with PIP worst (ring PIP over 40°) (Sprague et al. 2025, [PMC12473350](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473350/)). Lean on many frames and the full range of motion, not on precision per frame.
- **R_mount:** start from the static solution, then refine with weighted Wahba over the dynamic frames, weighting frames by `|ω_kp|` for the gyro pairs.
- **Validate:** after fitting, overlay `s_gl(t)` and `s_kp(t)` for a held-out 20 s. As targets **[est.]**:
  - bend RMSE under 15°;
  - gravity angle under 10°;
  - |ω| correlation above 0.8.

**How much paired data is enough?**
- **Flex:** 3 parameters per finger. The step-1 min/max covers about 80% of the benefit (Glauser: a generic model plus min/max reached 7.6°). 1–2 minutes of varied paired motion is ample for a quadratic.
- **R_mount:** 3 degrees of freedom. 2–3 static poses solve it; the dynamic refinement converges within 30 s.
- **Lag:** 1 parameter; 5–10 s with sharp rotations.
- **Recognizer fine-tuning:** this is where real data matters most.
  - IMUTube's distribution mapping converged with under 100 s of real data.
  - Leite 2024 used about 25 s of real data per gesture to go from F1 0.44–0.62 to 0.75.
  - SignRing mixed real and virtual 1:1.
  - **Budget 3–5 repetitions per sign per wearer** for the demo vocabulary.

---

## 5. Recognition options for a 20-hour build, ranked

| Rank | Approach | Effort | Expected accuracy **[est.]** | Breaks down at **[est.]** |
|---|---|---|---|---|
| **1 (build first)** | **1-NN DTW on the team's own glove templates** (3–5 per sign, same wearer), using the hand-state features and per-group weights | 3–4 h | 90–95% at 20–30 signs; about 80% at 50. Comparable: 96.6% kNN-DTW vs 98% CNN on 10 Italian signs ([PMC7411686](https://pmc.ncbi.nlm.nih.gov/articles/PMC7411686/)). | About 50–100 signs: template variance plus location-only collisions. Wearer-dependent. |
| **2 (stretch)** | **GRU / 1D-CNN trained on the converted dataset** with `simulate_glove` augmentation, **fine-tuned on real glove data** (and/or the DU glove data), outputting top-5, with **Claude choosing the sentence** | 6–10 h | Top-1 80–90% at 50–100 signs; top-1 60–75% / top-5 about 90% at 250 signs (PopSign scale) | Top-1 falls below about 50% past roughly 500 signs. That limit comes from what the glove can physically sense (§6), not from the model. |
| 3 | DTW against **dataset-derived templates only** (no real glove data) | 2 h after calibration | Highly uncertain: 40–70% at 20 signs. Every other paper in §2.1 saw a large gap here. | Immediately, as a primary recognizer. Use it as a candidate prefilter instead. |
| 4 | Contrastive glove ↔ keypoint embedding | Days | – | Needs far more paired data than a hackathon can collect |

We found no controlled study of DTW accuracy against vocabulary size. The breakdown points above are reasoned estimates, anchored on Athitsos 2010 (921 classes: top-20 recall 66.8% on video trajectories) and on the glove baselines in §2.4.

**Making DTW scale (rank 1 → about 100+ signs):**
1. **Prefilter by handshape:**
   - Cluster signs on `bend` (+ contacts) at the hold frame, the frame with minimum |ω|.
   - At runtime, classify the handshape from flex, then search only that cluster and its 1–2 nearest neighbours. This cuts candidates by about 5–20× **[est.]**.
2. **Prefilter by coarse motion and orientation:** gravity at sign start and end, plus total rotation ∫|ω|.
3. **Prune with LB_Keogh** ([Keogh 2002](https://www.cs.ucr.edu/~eamonn/vldb2002.pdf)) under a **Sakoe–Chiba band** of about 20% (`dtw(band=0.2)`).
4. **Multiple templates per sign** (3–5), taking the minimum distance.
5. Use the group weights in `feature_weights()`. Tune them on held-out repetitions.

**GRU recipe (rank 2):**
- Input: 32 frames × (12 per hand × 2 + contacts).
- Encoder: 2 × Conv1d(64, k=5), then a BiGRU(128), then mean-pool, then a linear layer.
- Train on converted dataset clips with augmentation:
  - `simulate_glove`;
  - time-warp ±20%;
  - random start/end trim of ±3 frames;
  - mixup.
- Fine-tune on real glove data mixed **1:1** with converted data (SignRing's best ratio), or freeze the encoder and use **nearest-centroid on enrolled glove samples**. That second option adds signs without retraining.
- Output top-5 with probabilities. Claude (this repo's `/api/interpret`) can take the sequence of top-5 lists and return the most plausible sentence, which is the role sentence context played in FinGTrAC.

**Segmentation (needed by both):** detect a sign as a burst of `|ω| + |d bend/dt|` above a threshold, framed by at least 300 ms of rest, with a hysteresis threshold. Mirror the web app's pause logic: a pause commits the sentence.

---

## 6. What 5 flex sensors + 1 IMU per hand cannot see, and what to add

ASL signs vary on handshape, location, movement, palm orientation and non-manual markers ([overview](https://en.wikipedia.org/wiki/American_Sign_Language_phonology)). The glove senses some of these well and some not at all:

| Parameter | Seen by flex + IMU? | Examples that collapse |
|---|---|---|
| Handshape: flexion of each finger | Yes (MCP+PIP sum) | Coarse only. Curved vs bent (MCP-only vs PIP-only) can give equal sums, e.g. C vs flat-O vs bent-B **[est.]**. |
| Handshape: **spread** | **No** | U vs V, B vs 4, 3 vs 3-closed |
| Handshape: **thumb position** / opposition | **Weak** (thumb CMC unseen) | A / S / T / M / N / E (thumb over or under fingers), O vs flat-O |
| Finger **crossing** | **No** | R vs U |
| Palm orientation | **Yes** (tilt) | Heading (facing left vs right) only *relative* to the sign start |
| Movement | **Yes** (rotation); weakly for translation | Pure translations with no wrist rotation are faint in the gyro and only appear in accelerometer residuals |
| **Location** | **No** | MOTHER/FATHER (5-hand: chin vs forehead), SUMMER/UGLY/DRY, APPLE/ONION ([Lifeprint minimal pairs](https://www.lifeprint.com/asl101/topics/minimalpairs.htm)) |
| Contact with body / other hand | Only via pads | Signs that touch the face vs signs held in neutral space |
| **Non-manual markers** (face, mouth, brows) | **No** | Questions vs statements, intensifiers, mouthing |

We found no published estimate of what fraction of ASL signs become indistinguishable for a glove. Section 6.1 shows how to compute it from your own data.

### 6.1 Ranking signs by how distinct they are

Run these in order. They are complementary.

1. **Phonological pass (from ASL-LEX 2.0, 2,723 signs; [asl-lex.org](https://asl-lex.org/), [OSF](https://osf.io/zpha4/)).**
   - Group signs by *glove-visible* codes only: Selected Fingers, Flexion, Flexion Change, Path Movement, Repeated Movement, Ulnar Rotation, Sign Type (number of hands).
   - Leave out the invisible codes: Major/Minor Location, Spread, Thumb Position, and Contact unless you have pads.
   - Every group with more than one member is a set of signs the glove cannot separate *in principle*.
   - Re-run the grouping with Spread / Thumb Contact / Location added back to see how many groups each extra sensor splits.
2. **Data-driven pass (from your dataset).**
   - Convert every clip to hand states and apply `simulate_glove` noise.
   - Run `confusability(templates)`. For each sign it returns the nearest other sign and a **margin** = (distance to the nearest other sign) / (median within-sign distance). A margin below about 1 means the pair is not reliably separable **[est. threshold]**.
   - For a more model-faithful version, train the GRU on converted data and read the confusion matrix on held-out signers. The Bhattacharyya distance and the silhouette score (Rousseeuw 1987) are standard alternatives.
3. **Pick the demo vocabulary greedily.**
   - Start from the most-needed signs (ASL-LEX frequency ratings help).
   - Repeatedly add the sign whose *minimum* margin to the already-chosen set is largest. This is farthest-first selection.
   - This gives the largest vocabulary that stays above a chosen margin.
4. **Value of each extra sensor, before you build it.**
   - The keypoints already contain what each candidate sensor would measure:
     - spread: `spread_angles`;
     - contacts: `contacts_from_keypoints`;
     - wrist position relative to the face: Holistic face and pose landmarks.
   - Add one candidate channel to the hand state, recompute the margins, and count how many pairs cross the threshold.
   - Choose the addition that rescues the most pairs per hour of build time.

### 6.2 Cheap sensor additions, by expected recovery per hour of build time **[est.]**

1. **Webcam fusion (free, 2–3 h): biggest win.**
   - Run MediaPipe on a laptop webcam pointed at the signer. Use only *wrist position relative to the face and shoulders* (Holistic pose), and let the glove handle handshape, which it does better than video under occlusion.
   - This resolves location pairs (MOTHER/FATHER) and is the same signal that let Gao et al. reach 5,177 signs with 3D trackers.
2. **More contact pads (cheap, 1 h; you already have copper tape).**
   - Thumb tip ↔ each fingertip covers O, F, 9, T, M/N, and flat-O vs C.
   - **Side of index ↔ side of middle** is a one-bit spread sensor: U/R (touching) vs V (apart). Adding pressure/contact sensing was the 65.7% → 98.2% jump in Lee & Lee 2018.
   - Wire each pad to a digital pin with `INPUT_PULLUP` and tie the thumb pad to GND; the scan is trivial.
3. **Forearm IMU (about $3 MPU-6050/BMI160 on I²C, 2 h).**
   - Forearm elevation plus hand orientation gives a coarse hand height. That separates forehead from chin from chest **[est.]** without a camera.
   - Check which I²C bus the onboard IMU uses before wiring **[unverified]**.
4. **Spread flex sensor:** a short flex sensor across the index–middle web. It gives a continuous spread reading but is fiddly to mount; the side contact pad gets most of the value.
5. **Thumb-CMC sensor:** a flex sensor along the thumb's base, from the thenar eminence toward the index. It helps the A/S/T/E family.

---

## 7. Open datasets and code

| Resource | License | Modality / size | How it helps |
|---|---|---|---|
| **DU ASL Sensor Dataglove** (Faisal et al. 2022), [Figshare](https://figshare.com/articles/dataset/ASL-Sensor-Dataglove-Dataset_zip/20031017) | **CC BY 4.0** | **5 × 2.2" flex + MPU-6050**, 100 Hz: 5 flex + 3 accel + 3 gyro + 4 quaternion. 40 signs, 25 subjects, ~10k samples. | **Your exact hardware class.** Use it for realistic noise and range in `simulate_glove`, to pretrain or fine-tune the glove recognizer, and as a sanity check on the keypoint→flex mapping for overlapping signs. |
| **SignSpeak** (Makkar et al.), [arXiv 2407.12020](https://arxiv.org/abs/2407.12020), [code](https://github.com/adityamakkar000/ASL-Sign-Research), [data](https://doi.org/10.7910/DVN/ODY7GH) | Code **MIT**; data **CC0 1.0** | 5 flex, **no IMU**, 36 Hz, 36 classes, 7,200 samples | Flex-only time-series baselines (LSTM/GRU/Transformer). A second source of real flex noise. *Unrelated to this repo despite the name.* |
| **UCI Auslan High Quality** (Kadous), [UCI](https://archive.ics.uci.edu/dataset/115/australian+sign+language+signs+high+quality) | **CC BY 4.0** | 2 × 5DT gloves: 5 bends + 6-DoF position/orientation per hand. ~100 Hz, 95 signs, 2,565 samples, 1 signer. | The only classic set pairing **flex + orientation + position**. Use it to measure how much location adds (§6) and to prototype DTW. It is Auslan, not ASL. |
| **PopSign ASL v1.0**, [site](https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/) | **CC BY 4.0** | 250 signs, 47 Deaf signers, ~200k selfie videos (1.1 TB) | Many signers per sign, for within-sign spread and confusability margins |
| **Kaggle "Isolated Sign Language Recognition"** (asl-signs), [Kaggle](https://www.kaggle.com/competitions/asl-signs) | **CC BY 4.0**. ⚠ The data card says the data is meant for educational ASL games and "not appropriate for other uses"; read it before relying on it. | MediaPipe Holistic landmarks (parquet), 250 signs, ~100k sequences, 21 signers. Exact counts and 543-landmark layout **[unverified]**. | Probably closest to your dataset's format. Has body pose for camera-roll correction and for location features. Frame size is unknown, so use `fix_aspect`. |
| **FSboard** (Google, fingerspelling), [Kaggle](https://www.kaggle.com/datasets/googleai/fsboard) | **CC BY 4.0** | >3M characters, 147 signers, MediaPipe Holistic at 30 Hz | Fingerspelling is pure handshape, which is the glove's strength. Use it for handshape clusters and letter templates. The 2023 Kaggle competition data license is **[unverified]**. |
| **WLASL**, [GitHub](https://github.com/dxli94/WLASL) | **C-UDA** | 2,000 glosses, YouTube video (some links dead) | Broad vocabulary to run MediaPipe on |
| **ASL Citizen** (Microsoft), [project](https://www.microsoft.com/en-us/research/project/asl-citizen/) | **Microsoft Research License**: non-commercial, no redistribution | 2,731 signs, 83k webcam videos, 52 signers | Large isolated vocabulary whose glosses map to ASL-LEX |
| **How2Sign**, [site](https://how2sign.github.io/) | **CC BY-NC 4.0** | 80 h continuous ASL. 2D OpenPose keypoints, plus a 3 h Panoptic-studio subset with **3D keypoints**. | Panoptic 3D hands give clean ground truth to test the bend math against monocular MediaPipe. Continuous signing helps with segmentation. |
| **ASL-LEX 2.0**, [site](https://asl-lex.org/), [OSF](https://osf.io/zpha4/) | Conflicting: CC BY 4.0 (OSF) vs **CC BY-NC 4.0** (site). Follow the stricter. | 2,723 signs with phonological codes (handshape features, location, movement, sign type), neighbourhood density, frequency | Drives the phonological confusability pass (§6.1) and frequency-weighted vocabulary choice |
| **MediaPipe HandLandmarker**, [docs](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | Apache 2.0 | 21 landmarks: normalized image coordinates + world coordinates in meters | Already used in `js/hands.js`. For calibration, record `worldLandmarks`. |
| **Arduino_BMI270_BMM150**, [GitHub](https://github.com/arduino-libraries/Arduino_BMI270_BMM150) | LGPL-2.1 **[unverified]** | Defaults: accel 100 Hz ±4 g, gyro 100 Hz ±2000 °/s | Matches the 100 Hz plan. Lower the gyro range to ±500 °/s for better resolution if the API allows **[unverified]**. |

---

## 8. Suggested 20-hour split (5 people)

| Hours | Firmware + hardware | Calibration / data | Recognition | App / demo |
|---|---|---|---|---|
| 0–4 | Flex, IMU, pads at 100 Hz over BLE with `micros()`; gyro bias; Madgwick 6-axis | Capture page: webcam `worldLandmarks` + BLE glove → one JSON | Convert the dataset with `keypoints_to_state`; run `confusability`; choose 30 demo signs | BLE → browser plumbing |
| 4–8 | Second glove; strain relief | Calibration flow (§4) and the overlay plot | DTW 1-NN + segmentation on recorded templates | Live UI, speech |
| 8–14 | Extra pads (index–middle side) | Record 5 reps × 30 signs × 2 wearers | GRU on converted dataset + DU data, fine-tune | Top-k → Claude sentence |
| 14–20 | Fixes | Re-calibration UX (10 s flat/fist) | Pick the better of DTW and GRU per vocabulary size | Demo script around signs that rank as separable |

---

## 9. Reference code

[`research/hand_state.py`](../../research/hand_state.py) (numpy only) implements everything above:
- `keypoints_to_state`;
- `glove_to_state`;
- `flex_resistance` and `fit_flex_calibration`;
- `solve_rotation` (Wahba);
- `estimate_lag`;
- `fix_aspect`;
- `mirror_state`;
- `time_normalize`, `dtw` and `confusability`;
- `simulate_glove`;
- the add-on channels `spread_angles` and `contacts_from_keypoints`.

[`research/test_hand_state.py`](../../research/test_hand_state.py) builds a synthetic hand with known joint angles and a known rotation rate, then generates a fake glove stream from the same motion with a random R_mount. It checks that:
- joint angles and bends are recovered exactly;
- gravity, ω and yaw match between the keypoint and glove paths, for right *and* left (mirrored) hands;
- R_mount is recovered by Wahba;
- the time lag and aspect ratio are recovered;
- the confusability ranking puts a near-duplicate pair first.

To run it:

```bash
python3 research/test_hand_state.py
```

These are synthetic checks. The code has **not** yet been run on real glove data or on the team's dataset.

---

## Sources

**Cross-modal / virtual sensor data**
- Kwon et al., "IMUTube," IMWUT 4(3) 2020. https://arxiv.org/abs/2006.05675
- Kwon, Wang, Abowd, Plötz, "Approaching the Real-World: Supporting Activity Recognition Training with Virtual IMU Data," IMWUT 5, 2021. DOI **[unverified]**.
- Kwon, Abowd, Plötz, "Complex Deep Neural Networks from Large Scale Virtual IMU Data…," Sensors 21(24):8337, 2021. https://doi.org/10.3390/s21248337
- Fortes Rey et al., "Let there be IMU data…," UbiComp/ISWC '19 Adjunct, 699–708. DOI 10.1145/3341162.3345590 **[DOI unverified]**.
- Santhalingam et al., "Synthetic Smartwatch IMU Data Generation from In-the-wild ASL Videos" (Vi2IMU), IMWUT 7(2) 2023. https://doi.org/10.1145/3596261
- Li et al., "SignRing," IMWUT 7(3) 2023. https://doi.org/10.1145/3610881
- Leite et al., "Simulation-driven design of smart gloves for gesture recognition," Sci. Rep. 14:14873, 2024. https://doi.org/10.1038/s41598-024-65069-2
- Leng, Kwon, Plötz, IMUGPT. https://arxiv.org/abs/2305.03187 ; IMUGPT 2.0 https://arxiv.org/abs/2402.01049
- Takeda et al., UbiComp/ISWC 2018 Adjunct. https://doi.org/10.1145/3267305.3267509
- Xia & Sugiura, ISWC 2022 (spring-joint virtual IMU).
- Moon et al., IMU2CLIP. https://arxiv.org/abs/2210.14395
- Girdhar et al., ImageBind, CVPR 2023. https://arxiv.org/abs/2305.05665 ; C3T https://arxiv.org/abs/2407.16803
- Deldari et al., COCOA https://arxiv.org/abs/2208.00467 ; CroSSL https://arxiv.org/abs/2307.16847
- Tong, Ge, Lane, zero-shot IMU HAR with video embeddings, IMWUT 2021. https://doi.org/10.1145/3494995

**Gloves and sign recognition**
- Glauser et al., stretch-sensing glove, ACM TOG 38(4) 2019. https://doi.org/10.1145/3306346.3322957
- Wang & Neff, data-driven glove calibration, SCA 2013. Kahlesz et al., CGI 2004. Kessler et al. 1995 (Georgia Tech TR).
- Gao et al., IJPRAI 14(5) 2000. https://doi.org/10.1142/S0218001400000386
- Zhou et al., Nature Electronics 3:571, 2020. https://doi.org/10.1038/s41928-020-0428-6
- Lee & Lee, IEEE Sensors J. 18(3) 2018. https://doi.org/10.1109/JSEN.2017.2779466
- Faisal et al., Sci. Rep. 12:21446, 2022. https://doi.org/10.1038/s41598-022-25108-2
- Makkar et al., SignSpeak. https://arxiv.org/abs/2407.12020
- Liu, Jiang, Gowda, FinGTrAC, IMWUT 4(3) 2020. https://doi.org/10.1145/3414117
- Athitsos et al. 2010, large-vocabulary DTW. https://www.cs.bu.edu/groups/ivc/pubs/Athitsos_cslt2010_a4.pdf
- Italian SL glove kNN-DTW vs CNN. https://pmc.ncbi.nlm.nih.gov/articles/PMC7411686/

**Sensors, pose, math**
- SparkFun Flex Sensor 2.2" and datasheet. https://www.sparkfun.com/flex-sensor-2-2.html ; https://cdn.sparkfun.com/assets/9/5/b/f/7/FLEX_SENSOR_-_SPECIAL_EDITION_DATA_SHEET_v2019__Rev_A_.pdf ; hookup guide https://learn.sparkfun.com/tutorials/flex-sensor-hookup-guide/all
- Saggio et al., "Resistive flex sensors: a survey," Smart Mater. Struct. 25 013001, 2016. https://doi.org/10.1088/0964-1726/25/1/013001
- Simone & Kamper, J. NeuroEng. Rehabil. 2:5, 2005. https://doi.org/10.1186/1743-0003-2-5
- Pratap et al. 2024 (multi-joint flex response). https://arxiv.org/abs/2405.19430
- Zhang et al., MediaPipe Hands. https://arxiv.org/abs/2006.10214 ; HandLandmarker docs https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Sprague et al., MediaPipe vs Vicon finger angles, Sensors 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12473350/
- Arduino Nano 33 BLE Rev2. https://docs.arduino.cc/hardware/nano-33-ble-rev2/ ; BMI270 https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi270/
- Madgwick, Harrison, Vaidyanathan, ICORR 2011. https://pure.ul.ie/en/publications/estimation-of-imu-and-marg-orientation-using-a-gradient-descent-a/
- Wahba 1965 / Markley 1988. https://en.wikipedia.org/wiki/Wahba%27s_problem ; Kabsch 1976 https://doi.org/10.1107/S0567739476001873
- Sakoe & Chiba 1978; Keogh 2002 LB_Keogh https://www.cs.ucr.edu/~eamonn/vldb2002.pdf
- Furgale, Rehder, Siegwart, Kalibr temporal calibration, IROS 2013. https://doi.org/10.1109/IROS.2013.6696514
- Rousseeuw 1987, silhouettes. https://doi.org/10.1016/0377-0427(87)90125-7

**Linguistics**
- ASL phonology overview. https://en.wikipedia.org/wiki/American_Sign_Language_phonology ; minimal pairs https://www.lifeprint.com/asl101/topics/minimalpairs.htm
- Sehyr et al., ASL-LEX 2.0, JDSE 26(2) 2021. https://doi.org/10.1093/deafed/enaa038

### Not verified

- Which Gao et al. paper reports the 5,177-sign result. The IJPRAI 2000 paper is confirmed; the 94.8% figure comes from the companion FG 2000 paper.
- The exact counts and 543-landmark schema of Kaggle asl-signs (the page needs JavaScript), and the license of the 2023 fingerspelling competition data.
- The DOIs of Kwon et al. 2021 (IMWUT) and Fortes Rey et al. 2019.
- The "sum of spanned joint angles" flex model as a formally validated result. It is physically motivated and consistent with Pratap et al. 2024.
- All accuracy and breakdown estimates marked **[est.]** in §5 and §6. No controlled study of DTW accuracy against vocabulary size was found.
- The Arduino_BMI270_BMM150 license, and whether its API exposes gyro range changes.
- The I²C bus layout for adding a forearm IMU to the Nano 33 BLE Rev2.
