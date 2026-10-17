// Synthetic hand and mock glove for testing calibrate.html without hardware.
//
// One motion model (SimHand) drives both sides, so a capture made with the
// synthetic camera and a mock glove has a known ground truth: the glove's
// R_mount, its clock offset and drift, and the camera latency are written to
// meta.mock in the export, and research/load_capture.py checks its estimates
// against them. Conventions follow research/hand_state.py:
//   camera frame  +x right, +y down, +z away from the camera (un-mirrored frames)
//   hand frame    +y wrist -> middle MCP, +x toward the thumb (right hand),
//                 +z out of the palm
//   Madgwick      quaternion maps sensor -> earth, earth +z up; accel reads +1 g up at rest

// --- 3x3 linear algebra (matrices are arrays of rows) ---------------------

export const mat = {
  I: () => [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  mul: (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j])),
  T: (A) => [0, 1, 2].map((j) => [A[0][j], A[1][j], A[2][j]]),
  vec: (A, v) => A.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]),
  cols: (x, y, z) => [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]],
  col: (A, j) => [A[0][j], A[1][j], A[2][j]],
};

const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (v) => Math.hypot(v[0], v[1], v[2]);
const scale = (v, s) => [v[0] * s, v[1] * s, v[2] * s];

/** Rodrigues: rotation vector -> rotation matrix. */
export function expm(w) {
  const th = norm(w);
  if (th < 1e-12) return mat.I();
  const [x, y, z] = scale(w, 1 / th);
  const K = [[0, -z, y], [z, 0, -x], [-y, x, 0]];
  const KK = mat.mul(K, K);
  const s = Math.sin(th), c = 1 - Math.cos(th);
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => (i === j ? 1 : 0) + s * K[i][j] + c * KK[i][j]));
}

/** Rotation matrix -> rotation vector (inverse of expm, angle < pi). */
export function logm(R) {
  const c = Math.min(1, Math.max(-1, (R[0][0] + R[1][1] + R[2][2] - 1) / 2));
  const th = Math.acos(c);
  const v = [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]];
  if (th < 1e-9) return scale(v, 0.5);
  return scale(v, th / (2 * Math.sin(th)));
}

/** Rotation matrix -> unit quaternion (w, x, y, z), Shepperd's method. */
export function quatFromMatrix(R) {
  const t = R[0][0] + R[1][1] + R[2][2];
  let q;
  if (t > 0) {
    const s = Math.sqrt(t + 1) * 2;
    q = [s / 4, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s];
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    const s = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2;
    q = [(R[2][1] - R[1][2]) / s, s / 4, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s];
  } else if (R[1][1] > R[2][2]) {
    const s = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2;
    q = [(R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, s / 4, (R[1][2] + R[2][1]) / s];
  } else {
    const s = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2;
    q = [(R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, s / 4];
  }
  return q[0] < 0 ? q.map((x) => -x) : q;
}

// --- Seeded randomness ------------------------------------------------------

export function rng(seed = 1) {
  let s = seed >>> 0 || 1;
  const uniform = () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(uniform() + 1e-12)) * Math.cos(2 * Math.PI * uniform());
  return { uniform, gauss };
}

function hashString(s) {
  let h = 2166136261;
  for (const ch of s) h = Math.imul(h ^ ch.codePointAt(0), 16777619);
  return h >>> 0;
}

// --- Hand geometry ----------------------------------------------------------

const PALM_M = 0.085; // wrist -> middle MCP, metres

/**
 * 21 MediaPipe-ordered points of a RIGHT hand in its own frame, metres.
 * bends: thumb..pinky, radians, as hand_state.flex_bend measures them
 * (finger MCP + PIP, thumb MCP + IP). A port of test_hand_state.synthetic_hand.
 */
export function handPoints(bends) {
  const p = Array.from({ length: 21 }, () => [0, 0, 0]);
  const xs = [0.35, 0.0, -0.2, -0.4];
  [5, 9, 13, 17].forEach((base, f) => {
    const b = bends[f + 1];
    p[base] = [xs[f], 1.0, 0.0];
    let ang = 0;
    [[0.5 * b, 0.45], [0.5 * b, 0.28], [0.35 * b, 0.2]].forEach(([theta, len], k) => {
      ang += theta; // flexion rotates +y toward +z about +x
      const prev = p[base + k];
      p[base + k + 1] = [prev[0], prev[1] + len * Math.cos(ang), prev[2] + len * Math.sin(ang)];
    });
  });
  // Thumb roughly in the x-y plane, flexing about z toward the palm.
  p[1] = [0.3, 0.25, 0.05];
  const d0 = [0.7 / Math.hypot(0.7, 0.5), 0.5 / Math.hypot(0.7, 0.5), 0];
  p[2] = [p[1][0] + 0.35 * d0[0], p[1][1] + 0.35 * d0[1], p[1][2]];
  let ang = 0;
  [[0.45 * bends[0], 0.3], [0.55 * bends[0], 0.25]].forEach(([theta, len], k) => {
    ang += theta;
    const d = mat.vec(expm([0, 0, ang]), d0);
    const prev = p[2 + k];
    p[3 + k] = [prev[0] + len * d[0], prev[1] + len * d[1], prev[2] + len * d[2]];
  });
  return p.map((v) => scale(v, PALM_M));
}

// camera <- hand rotations (right hand) for the poses the calibration asks for.
const R_FACE = mat.cols([1, 0, 0], [0, -1, 0], [0, 0, -1]);       // palm to camera, fingers up
const R_PALM_DOWN = mat.cols([1, 0, 0], [0, 0, -1], [0, 1, 0]);   // palm to floor, fingers to camera
const R_FINGERS_DOWN = mat.cols([0, 0, -1], [0, 1, 0], [1, 0, 0]); // arm hanging, palm to body
const R_THUMB_UP = mat.cols([0, -1, 0], [0, 0, -1], [1, 0, 0]);    // thumb up, palm facing signer's left

const OPEN = [0.1, 0.05, 0.05, 0.05, 0.05];
const FIST = [1.2, 2.5, 2.6, 2.6, 2.5];
const lerp = (a, b, s) => a.map((x, i) => x + (b[i] - x) * s);

function blendR(A, B, s) {
  return mat.mul(A, expm(scale(logm(mat.mul(mat.T(A), B)), s)));
}

/** Pose of a right hand doing `step` for `tau` seconds. */
function rawPose(step, tau, sign) {
  const t = Math.max(0, tau);
  switch (step) {
    case 'still':
      return { R: R_PALM_DOWN, bends: OPEN };
    case 'flex_fist': {
      const s = (1 - Math.cos(Math.PI * t)) / 2; // open at 0 s, fist at 1 s, 5 cycles in 10 s
      return { R: R_FACE, bends: lerp(OPEN, FIST, s) };
    }
    case 'pose_palm_down':
      return { R: R_PALM_DOWN, bends: OPEN };
    case 'pose_fingers_down':
      return { R: R_FINGERS_DOWN, bends: [0.2, 0.15, 0.15, 0.15, 0.15] };
    case 'pose_thumb_up':
      return { R: R_THUMB_UP, bends: [0.1, 2.5, 2.6, 2.6, 2.5] };
    case 'twists': {
      // Sharp out-and-back twists about the forearm (hand y) at the cue times.
      const a = TWIST_TIMES_S.reduce((sum, c) => sum + 1.3 * Math.exp(-(((t - c) / 0.15) ** 2)), 0);
      return { R: mat.mul(R_FACE, expm([0, a, 0])), bends: [0.3, 0.2, 0.2, 0.2, 0.2] };
    }
    case 'free': {
      const w = [0.6 * Math.sin(0.9 * t) + 0.3 * Math.sin(2.1 * t + 1), 0.8 * Math.sin(0.6 * t + 0.5), 0.5 * Math.sin(1.3 * t + 2)];
      const bends = [0.6 + 0.5 * Math.sin(1.1 * t), 1.2 + 1.1 * Math.sin(1.7 * t + 0.3), 1.2 + 1.1 * Math.sin(1.3 * t + 1.1),
        1.2 + 1.1 * Math.sin(0.9 * t + 2.0), 1.2 + 1.1 * Math.sin(2.3 * t + 2.7)];
      return { R: mat.mul(R_FACE, expm(w)), bends };
    }
    case 'sign': {
      // A deterministic, label-specific movement, so repetitions of one sign look alike.
      const r = rng(hashString(sign || 'sign'));
      const amp = [0, 1, 2].map(() => 0.9 * (r.uniform() - 0.5));
      const ph = [0, 1, 2].map(() => 6.28 * r.uniform());
      const target = FIST.map((f) => f * r.uniform());
      const s = Math.sin(Math.PI * Math.min(t / 2.5, 1)); // up and back within the repetition
      const w = amp.map((a, i) => a * s * Math.sin(2.5 * t + ph[i]) + a);
      return { R: mat.mul(R_FACE, expm(w)), bends: lerp(OPEN, target, s) };
    }
    default: {
      // Idle and between steps: palm toward the camera, gentle sway.
      const w = [0.05 * Math.sin(0.7 * t), 0.08 * Math.sin(0.5 * t), 0.04 * Math.sin(0.9 * t)];
      return { R: mat.mul(R_FACE, expm(w)), bends: OPEN.map((b) => b + 0.25 + 0.1 * Math.sin(0.8 * t)) };
    }
  }
}

export const TWIST_TIMES_S = [1.0, 2.5, 4.0];

/**
 * Time-driven motion of a synthetic right hand. The page tells it which step is
 * running (setStep); anything can then ask for the pose at any time, including
 * past times, which the mock glove needs.
 */
export class SimHand {
  constructor() {
    this.timeline = [{ t: -Infinity, step: 'idle', sign: null }];
  }

  /** From time tMs on, the hand does `step` ('ready' holds the start pose of `next`). */
  setStep(step, tMs, { sign = null, next = null } = {}) {
    this.timeline.push({ t: tMs, step, sign, next });
  }

  _raw(entry, tMs) {
    const tau = entry.t === -Infinity ? tMs / 1000 : (tMs - entry.t) / 1000;
    if (entry.step === 'ready') return rawPose(entry.next || 'idle', 0, entry.sign);
    return rawPose(entry.step, tau, entry.sign);
  }

  /** {R: camera <- right-hand rotation, bends[5]} at time tMs (performance.now() clock). */
  poseAt(tMs) {
    let i = this.timeline.length - 1;
    while (i > 0 && this.timeline[i].t > tMs) i--;
    const cur = this._raw(this.timeline[i], tMs);
    if (i === 0) return cur;
    const s = (tMs - this.timeline[i].t) / 600; // blend over 0.6 s from the previous step
    if (s >= 1) return cur;
    const prev = this._raw(this.timeline[i - 1], this.timeline[i].t);
    const e = s * s * (3 - 2 * s);
    return { R: blendR(prev.R, cur.R, e), bends: lerp(prev.bends, cur.bends, e) };
  }

  /**
   * Pose for a physical hand. A left hand does the mirror image of the right
   * hand's motion; its frame is y = toward the fingers, z = out of the palm,
   * x = y cross z (the pinky side), as in hand_state.glove_to_state.
   * Returns {R: camera <- that hand's frame, points: 21 x 3 camera-frame metres}.
   */
  handAt(tMs, hand) {
    const { R, bends } = this.poseAt(tMs);
    const local = handPoints(bends);
    const right = local.map((v) => mat.vec(R, v));
    if (hand !== 'Left') return { R, points: right, bends };
    const P = (v) => [-v[0], v[1], v[2]];
    const y = P(mat.col(R, 1)), z = P(mat.col(R, 2));
    return { R: mat.cols(cross(y, z), y, z), points: right.map(P), bends };
  }
}

// --- Synthetic camera -------------------------------------------------------

/**
 * Produces MediaPipe-shaped results from the SimHand: world landmarks (metres,
 * origin at the hand's centre) and normalized image landmarks for a 640 x 480
 * un-mirrored frame. The handedness label follows MediaPipe's convention, which
 * assumes a mirrored selfie image: on un-mirrored frames the signer's right hand
 * is labelled 'Left'.
 */
export class SimCamera {
  constructor(sim, { hands = ['Right'], latencyMs = 70, noiseM = 0.0015, seed = 7 } = {}) {
    this.sim = sim;
    this.hands = hands;
    this.latencyMs = latencyMs; // the frame shows the hand as it was this long ago
    this.noiseM = noiseM;
    this.rand = rng(seed);
    this.width = 640;
    this.height = 480;
  }

  /** Detection result for a frame processed at tMs. */
  detect(tMs) {
    const truth = tMs - this.latencyMs;
    const out = { landmarks: [], worldLandmarks: [], handedness: [] };
    for (const hand of this.hands) {
      const { points } = this.sim.handAt(truth, hand);
      const n = () => this.rand.gauss() * this.noiseM;
      const noisy = points.map((v) => [v[0] + n(), v[1] + n(), v[2] + n()]);
      const c = noisy.reduce((a, v) => [a[0] + v[0] / 21, a[1] + v[1] / 21, a[2] + v[2] / 21], [0, 0, 0]);
      const world = noisy.map((v) => ({ x: v[0] - c[0], y: v[1] - c[1], z: v[2] - c[2] }));
      // Physical right hand appears on the image's left in an un-mirrored frame.
      const cx = (hand === 'Left' ? 0.68 : 0.32) + 0.03 * Math.sin(truth / 3000);
      const cy = 0.55;
      const span = 0.6; // metres across the frame width at the hand's depth
      const aspect = this.width / this.height;
      const image = noisy.map((v) => ({
        x: cx + (v[0] - c[0]) / span,
        y: cy + ((v[1] - c[1]) / span) * aspect,
        z: (v[2] - noisy[0][2]) / span,
      }));
      out.worldLandmarks.push(world);
      out.landmarks.push(image);
      out.handedness.push([{ categoryName: hand === 'Left' ? 'Right' : 'Left', score: 0.97, index: 0 }]);
    }
    return out;
  }
}

// --- Mock glove -------------------------------------------------------------

/**
 * Synthesizes the glove's BLE notifications from the SimHand and hands them to
 * `onNotify(DataView)` exactly as a real characteristic would, after encoding
 * them with the page's packet layout (so the decoder is exercised too).
 *
 * Imitates: a 100 Hz sampler on the device clock (micros(), uint32, starting
 * close to its wrap-around and drifting by `driftPpm`), a fixed but unknown
 * IMU mounting rotation, gyro bias and noise, Madgwick yaw drift, a flex
 * sensor in a 47k divider, and BLE delivery in connection events with jitter.
 */
export class MockGlove {
  constructor(sim, { hand = 'Right', encode, samplesPerNotify = 3, adcMax = 4095, seed = 11 } = {}) {
    this.sim = sim;
    this.hand = hand;
    this.encode = encode;
    this.samplesPerNotify = samplesPerNotify;
    this.adcMax = adcMax;
    const r = (this.rand = rng(seed + (hand === 'Left' ? 100 : 0)));
    const randRot = () => expm([0, 1, 2].map(() => (r.uniform() - 0.5) * 3));
    this.rMount = randRot(); // sensor -> hand
    const yaw0 = r.uniform() * 2 * Math.PI;
    // earth <- camera: camera +y (down) is earth -z; heading is arbitrary.
    this.rEarthCam = mat.mul(expm([0, 0, yaw0]), [[1, 0, 0], [0, 0, 1], [0, -1, 0]]);
    this.gyroBias = [0, 1, 2].map(() => (r.uniform() - 0.5) * 0.03); // rad/s
    this.yawDriftRadS = 0.004;
    this.driftPpm = 40;
    this.dev0 = 2 ** 32 - 20e6 + Math.floor(r.uniform() * 1e6); // wraps after ~20 s
    this.t0 = null;
    this.periodMs = 10;
    this.latencyMs = [6, 30];
    this.timer = null;
    this.queue = [];
  }

  /** Ground truth written to the capture's meta.mock. */
  truth() {
    return {
      hand: this.hand,
      r_mount: this.rMount,
      gyro_bias: this.gyroBias,
      device_clock: { t0_ms: this.t0, dev0_us: this.dev0, drift_ppm: this.driftPpm },
      ble_latency_ms: this.latencyMs,
    };
  }

  start(onNotify) {
    this.stop();
    this.onNotify = onNotify;
    this.t0 = performance.now();
    this.k = 0;
    this.lastDelivery = 0;
    this.queue = [];
    this.timer = setInterval(() => this._tick(), 15); // one BLE connection event per 15 ms
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
  }

  /** True (performance.now()) time of sample k: the device clock runs fast by driftPpm. */
  _truthMs(k) {
    return this.t0 + (k * this.periodMs) / (1 + this.driftPpm * 1e-6);
  }

  _sample(k) {
    const truth = this._truthMs(k);
    const devUs = (this.dev0 + k * this.periodMs * 1000) % 2 ** 32;
    const h = 5;
    const a = this.sim.handAt(truth - h, this.hand).R;
    const cur = this.sim.handAt(truth, this.hand);
    const b = this.sim.handAt(truth + h, this.hand).R;
    const omegaHand = scale(logm(mat.mul(mat.T(a), b)), 1000 / (2 * h));
    const rEarthSensor = mat.mul(mat.mul(expm([0, 0, this.yawDriftRadS * (truth - this.t0) / 1000]), this.rEarthCam),
      mat.mul(cur.R, this.rMount));
    const r = this.rand;
    const gyro = mat.vec(mat.T(this.rMount), omegaHand).map((w, i) => w + this.gyroBias[i] + 0.02 * r.gauss());
    const accel = mat.vec(mat.T(rEarthSensor), [0, 0, 1]).map((g) => g + 0.01 * r.gauss());
    const flex = cur.bends.map((bend) => {
      const ohms = 25_000 + 30_000 * Math.max(bend, 0) + 300 * r.gauss(); // flex on top, 47k to ground
      return Math.round((this.adcMax * 47_000) / (47_000 + ohms));
    });
    return { t_us: devUs, flex, accel, gyro, quat: quatFromMatrix(rEarthSensor) };
  }

  /** One connection event: sample up to now, deliver whatever is due, in order. */
  _tick() {
    const now = performance.now();
    const [lo, hi] = this.latencyMs;
    for (let t = this._truthMs(this.k); t <= now; t = this._truthMs(this.k)) {
      // BLE keeps order, so a sample is never due before the one taken earlier.
      this.lastDelivery = Math.max(t + lo + (hi - lo) * this.rand.uniform(), this.lastDelivery);
      this.queue.push({ due: this.lastDelivery, sample: this._sample(this.k++) });
    }
    let n = 0;
    while (n < this.queue.length && this.queue[n].due <= now) n++;
    const due = this.queue.splice(0, n).map((q) => q.sample);
    for (let i = 0; i < due.length; i += this.samplesPerNotify) {
      this.onNotify?.(this.encode(due.slice(i, i + this.samplesPerNotify)));
    }
  }
}
