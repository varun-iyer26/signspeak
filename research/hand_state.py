"""Shared "hand-state" representation for ASL keypoint data and a flex+IMU glove.

Reference implementation for docs/research/glove-dataset-alignment.md.
numpy only. Every function works on one hand; run it twice for two hands.

The hand state for one frame is a 12-vector (see ``STATE_LAYOUT``)::

    bend[5]     thumb..pinky, radians. bend_f = theta_MCP + theta_PIP (thumb: MCP + IP).
                This is what a 2.2" flex sensor on the back of the finger sees.
    gravity[3]  unit vector pointing DOWN, expressed in the hand frame.
    omega[3]    angular velocity of the hand, hand frame, rad/s.
    yaw[1]      rotation about the vertical axis since the start of the sign, radians.

Both sides produce it:

    keypoints (21 x 3, MediaPipe order)  -> keypoints_to_state(...)
    glove (5 flex ADC + accel + gyro + quaternion) -> glove_to_state(...)

Hand frame (right hand; the left hand is mirrored into a right hand first):

    origin  wrist (landmark 0)
    +y      wrist -> middle-finger MCP (landmark 9), "toward the fingers"
    +x      across the palm, pinky MCP (17) -> index MCP (5), "toward the thumb side"
    +z      x cross y = out of the PALM (the side the fingers curl toward)

Camera frame (MediaPipe / OpenCV): +x right, +y down, +z away from the camera.
With an upright camera, gravity in camera coordinates is (0, 1, 0).

MediaPipe landmark indices::

    0 wrist
    thumb  1 CMC  2 MCP  3 IP   4 tip
    index  5 MCP  6 PIP  7 DIP  8 tip
    middle 9      10     11     12
    ring   13     14     15     16
    pinky  17     18     19     20
"""

from __future__ import annotations

import numpy as np

STATE_LAYOUT = {"bend": slice(0, 5), "gravity": slice(5, 8), "omega": slice(8, 11), "yaw": slice(11, 12)}
STATE_DIM = 12

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# (base, j1, j2, j3, tip) chains. For fingers the base is the wrist so the first
# angle is the MCP; for the thumb the chain starts at the CMC so the first angle is the MCP.
_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
# Bones used to sanity-check scale / aspect ratio (parent, child).
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
         (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
         (5, 9), (9, 13), (13, 17)]

GRAVITY_CAMERA = np.array([0.0, 1.0, 0.0])  # upright camera, image y points down
MIRROR = np.diag([-1.0, 1.0, 1.0])          # reflect across the hand's y-z plane


# --------------------------------------------------------------------------- #
# Small rotation helpers
# --------------------------------------------------------------------------- #

def _unit(v, axis=-1, eps=1e-9):
    v = np.asarray(v, dtype=float)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), eps)


def hat(w):
    """3-vector -> skew-symmetric matrix, so hat(w) @ v == cross(w, v)."""
    x, y, z = w
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_log(R):
    """Rotation matrix -> rotation vector (axis * angle, radians)."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-6:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if np.pi - angle < 1e-4:
        # Near 180 degrees: axis from the largest diagonal entry of (R + I) / 2.
        M = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(M)))
        axis = _unit(M[:, k])
        return axis * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * angle / (2.0 * np.sin(angle))


def quat_to_matrix(q):
    """Unit quaternion (w, x, y, z) -> rotation matrix. Madgwick's convention:
    the matrix maps sensor-frame vectors into the earth frame."""
    w, x, y, z = _unit(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def twist_angle(R_delta, axis):
    """Signed rotation of ``R_delta`` about ``axis`` (swing-twist decomposition).

    Used for yaw: ``axis`` is the vertical, so tilting the hand does not leak into yaw.
    """
    rv = so3_log(R_delta)
    angle = np.linalg.norm(rv)
    if angle < 1e-9:
        return 0.0
    w = np.cos(angle / 2.0)
    v = np.sin(angle / 2.0) * rv / angle
    return float(2.0 * np.arctan2(v @ _unit(axis), w))


# --------------------------------------------------------------------------- #
# Keypoints -> hand state
# --------------------------------------------------------------------------- #

def canonicalize(points, handedness="Right", mirrored_input=False):
    """Return landmarks that look like a RIGHT hand in an un-mirrored camera.

    points: (21, 3) or (T, 21, 3).
    handedness: the label of the PHYSICAL hand ('Right' or 'Left'). MediaPipe's label
        assumes a mirrored selfie image; if your video is not mirrored, MediaPipe's
        'Left' is the signer's right hand. Decide this once per dataset.
    mirrored_input: True if the frames were horizontally flipped (selfie view).

    A left hand, or a right hand seen in a mirror, is reflected in x. Reflection
    flips the sign of cross products, so do it BEFORE building the hand frame.
    """
    p = np.array(points, dtype=float)
    flip = (handedness.lower().startswith("l")) != bool(mirrored_input)
    if flip:
        p[..., 0] *= -1.0
    return p


def hand_frame(points):
    """Rotation camera<-hand for one frame of (21, 3) canonical right-hand landmarks.

    Columns are the hand axes (x, y, z) written in camera coordinates, so
    ``v_cam = R @ v_hand`` and ``v_hand = R.T @ v_cam``.
    """
    p = np.asarray(points, dtype=float)
    y = _unit(p[9] - p[0])
    x = p[5] - p[17]
    x = _unit(x - (x @ y) * y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def joint_angles(points):
    """Flexion angles (radians) for one frame of canonical landmarks.

    Returns dict finger -> array of joint angles:
        fingers: [MCP, PIP, DIP]; thumb: [MCP, IP].
    The finger MCP angle is SIGNED flexion measured in the finger's flexion plane
    (the hand's y-z plane), so side-to-side spread does not count as bend and
    hyperextension is negative. PIP/DIP/thumb angles are unsigned (those joints are
    close to hinges and rarely hyperextend much).
    """
    p = np.asarray(points, dtype=float)
    R = hand_frame(p)
    x_axis = R[:, 0]
    out = {}
    for name, chain in _CHAINS.items():
        bones = [p[b] - p[a] for a, b in zip(chain[:-1], chain[1:])]
        angles = []
        for k, (u, v) in enumerate(zip(bones[:-1], bones[1:])):
            if name != "thumb" and k == 0:
                up = u - (u @ x_axis) * x_axis
                vp = v - (v @ x_axis) * x_axis
                angles.append(np.arctan2(np.cross(up, vp) @ x_axis, up @ vp))
            else:
                angles.append(np.arccos(np.clip(_unit(u) @ _unit(v), -1.0, 1.0)))
        out[name] = np.array(angles)
    return out


def flex_bend(angles, weights=None):
    """Collapse joint angles to the one number a back-of-finger flex sensor sees.

    A bend sensor's resistance tracks the curvature integrated along its length,
    which equals the total change in direction between its two ends, i.e. the SUM
    of the joint angles it spans. A 2.2" sensor starting over the metacarpal head
    spans MCP and PIP; the DIP is usually past its end (and moves with the PIP anyway).

    weights: dict finger -> per-joint weights. Default: fingers (1, 1, 0), thumb (1, 1).
    Returns (5,) radians in FINGERS order. Negative values (hyperextension) are kept;
    the sensor is nearly blind to them, so clip at 0 when simulating glove readings.
    """
    if weights is None:
        weights = {f: (np.array([1.0, 1.0]) if f == "thumb" else np.array([1.0, 1.0, 0.0])) for f in FINGERS}
    return np.array([float(np.asarray(weights[f]) @ angles[f]) for f in FINGERS])


def spread_angles(points):
    """Abduction between neighbouring fingers (index-middle, middle-ring, ring-pinky), radians.

    Measured between proximal phalanges projected onto the palm plane. The glove cannot
    see this unless you add a spread sensor; use it to find signs that differ only here.
    """
    p = np.asarray(points, dtype=float)
    z = hand_frame(p)[:, 2]
    prox = [p[6] - p[5], p[10] - p[9], p[14] - p[13], p[18] - p[17]]
    prox = [_unit(v - (v @ z) * z) for v in prox]
    return np.array([np.arccos(np.clip(a @ b, -1.0, 1.0)) for a, b in zip(prox[:-1], prox[1:])])


def contacts_from_keypoints(points, pairs=((4, 8), (4, 12), (4, 16), (4, 20)), threshold=0.25):
    """Binary contacts that mimic copper-tape pads: 1 when two landmarks are within
    ``threshold`` x palm length (wrist -> middle MCP). Default pairs: thumb tip to each fingertip."""
    p = np.asarray(points, dtype=float)
    scale = np.linalg.norm(p[9] - p[0])
    return np.array([float(np.linalg.norm(p[a] - p[b]) < threshold * scale) for a, b in pairs])


def angular_velocity(rotations, t):
    """Body-frame angular velocity from a sequence of camera<-hand rotations.

    omega_k = log(R_k^T R_{k+1}) / dt  (central difference, hand frame, rad/s).
    This is what a gyro rigidly attached to the hand measures (after R_mount).
    Differentiating amplifies keypoint jitter: smooth landmarks first (see ``smooth``).
    """
    Rs = np.asarray(rotations)
    t = np.asarray(t, dtype=float)
    n = len(Rs)
    w = np.zeros((n, 3))
    for k in range(n):
        a, b = max(k - 1, 0), min(k + 1, n - 1)
        if a == b:
            continue
        # Midpoint frame k: (R_a^T R_b) is the body-frame rotation from a to b.
        w[k] = so3_log(Rs[a].T @ Rs[b]) / (t[b] - t[a])
    return w


def smooth(x, window=5):
    """Centered moving average along axis 0 (edges use a shrinking window)."""
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < 2:
        return x.copy()
    k = window // 2
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), x]), axis=0)
    idx = np.arange(len(x))
    lo, hi = np.maximum(idx - k, 0), np.minimum(idx + k + 1, len(x))
    return (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


def keypoints_to_state(seq, t, handedness="Right", mirrored_input=False,
                       gravity_camera=GRAVITY_CAMERA, smooth_window=5, weights=None):
    """(T, 21, 3) landmarks + timestamps (s) -> (T, 12) hand states.

    Use MediaPipe WORLD landmarks (metres, isotropic) when you can. Normalised image
    landmarks also work if you first undo the aspect ratio (``fix_aspect``); their scale
    does not matter because only angles and directions are used.

    gravity_camera: down direction in camera coordinates. (0, 1, 0) assumes an upright,
        level camera; replace it with the phone's accelerometer reading if you have it.
    """
    p = smooth(canonicalize(seq, handedness, mirrored_input), smooth_window)
    t = np.asarray(t, dtype=float)
    Rs = np.stack([hand_frame(f) for f in p])
    g_cam = _unit(np.asarray(gravity_camera, dtype=float))
    if handedness.lower().startswith("l") != bool(mirrored_input):
        g_cam = MIRROR @ g_cam  # gravity lives in the (flipped) canonical space too
    states = np.zeros((len(p), STATE_DIM))
    for k, f in enumerate(p):
        states[k, STATE_LAYOUT["bend"]] = flex_bend(joint_angles(f), weights)
        states[k, STATE_LAYOUT["gravity"]] = Rs[k].T @ g_cam
        states[k, 11] = twist_angle(Rs[k] @ Rs[0].T, g_cam)
    states[:, STATE_LAYOUT["omega"]] = angular_velocity(Rs, t)
    return states


def fix_aspect(seq, aspect=None):
    """Make normalised image landmarks isotropic: y *= height / width.

    If the frame size is unknown (e.g. Kaggle parquet files), estimate it: the right
    aspect is the one that keeps bone lengths most constant while the hand rotates.
    Returns (fixed_seq, aspect).
    """
    seq = np.array(seq, dtype=float)
    if aspect is None:
        candidates = np.linspace(0.4, 2.5, 85)
        cost = [_bone_length_cv(seq * np.array([1.0, a, 1.0])) for a in candidates]
        aspect = float(candidates[int(np.argmin(cost))])
    seq[..., 1] *= aspect
    return seq, aspect


def _bone_length_cv(seq):
    seq = seq.reshape(-1, 21, 3)
    lengths = np.stack([np.linalg.norm(seq[:, b] - seq[:, a], axis=-1) for a, b in BONES], axis=1)
    lengths = lengths / np.maximum(lengths.mean(axis=1, keepdims=True), 1e-9)
    return float(np.mean(lengths.std(axis=0) / np.maximum(lengths.mean(axis=0), 1e-9)))


def finger_straightness_2d(points_2d):
    """Fallback for 2D-only keypoints: chord / path length per finger, in (0, 1].

    1.0 = straight finger, smaller = curled. Not an angle and view-dependent, but
    monotonic in curl for most views. Train a regressor from it to ``bend`` on data
    where you have both (re-run MediaPipe on a few videos) rather than using it raw.
    """
    p = np.asarray(points_2d, dtype=float)[..., :2]
    chains = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
              "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
    out = []
    for f in FINGERS:
        c = chains[f]
        path = sum(np.linalg.norm(p[..., b, :] - p[..., a, :], axis=-1) for a, b in zip(c[:-1], c[1:]))
        chord = np.linalg.norm(p[..., c[-1], :] - p[..., c[0], :], axis=-1)
        out.append(chord / np.maximum(path, 1e-9))
    return np.stack(out, axis=-1)


# --------------------------------------------------------------------------- #
# Glove -> hand state
# --------------------------------------------------------------------------- #

def flex_resistance(adc, r_fixed=47_000.0, adc_max=1023, flex_on_top=True):
    """Undo the voltage divider: ADC counts -> flex sensor resistance (ohms).

    flex_on_top=True: Vcc -- flex -- A_n -- 47k -- GND, so V = Vcc * R_fixed / (R_fixed + R_flex).
    flex_on_top=False: Vcc -- 47k -- A_n -- flex -- GND.
    Resistance is close to linear in bend angle; the divider output is not, so fit in
    resistance (or conductance) space, not raw ADC counts.
    """
    v = np.clip(np.asarray(adc, dtype=float) / adc_max, 1e-4, 1 - 1e-4)
    return r_fixed * (1.0 / v - 1.0) if flex_on_top else r_fixed * v / (1.0 - v)


def fit_flex_calibration(resistance, bend, degree=2):
    """Least-squares polynomial bend = f(R) for one finger, from paired recordings.

    resistance: (N,) ohms from the glove. bend: (N,) radians from ``flex_bend`` on
    time-aligned webcam keypoints. Returns polynomial coefficients for np.polyval.
    Scaling R to kilo-ohms keeps the fit well conditioned.
    """
    r = np.asarray(resistance, dtype=float) / 1000.0
    return np.polyfit(r, np.asarray(bend, dtype=float), degree)


def apply_flex_calibration(resistance, coeffs):
    return np.polyval(coeffs, np.asarray(resistance, dtype=float) / 1000.0)


def solve_rotation(a, b, weights=None):
    """Wahba / Kabsch: the rotation R minimising sum w_i |b_i - R a_i|^2.

    Use it for R_mount (sensor -> hand): a = IMU vectors in the sensor frame, b = the
    same physical vectors in the hand frame. Good pairs:
      * accelerometer at rest (negated) vs the known down direction in a known pose
        (right hand palm-down flat on a table: down = +z_hand; hanging with fingers
        pointing at the floor: down = +y_hand), or vs keypoint gravity;
      * gyro vs keypoint angular velocity during a paired recording.
    Needs at least two non-parallel directions.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    B = (b * w[:, None]).T @ a
    U, _, Vt = np.linalg.svd(B)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def glove_to_state(t, flex_res, accel, gyro, quat, R_mount, flex_coeffs, handedness="Right",
                   gyro_bias=None, accel_lowpass=10):
    """Glove samples -> (T, 12) hand states in the same space as ``keypoints_to_state``.

    t: (T,) seconds. flex_res: (T, 5) ohms (thumb..pinky). accel: (T, 3) in g or m/s^2.
    gyro: (T, 3) rad/s. quat: (T, 4) Madgwick (w, x, y, z), sensor -> earth, or None.
    R_mount: (3, 3) sensor -> hand rotation. flex_coeffs: list of 5 polynomial fits.
    handedness: physical hand. A left glove is mirrored into a right hand like the keypoints.
        For a LEFT glove, R_mount must map into the left hand's frame defined as
        y = toward the fingers, z = out of the palm, x = y cross z (the PINKY side).
        To solve it from a paired recording, run ``solve_rotation`` against
        ``mirror_state(keypoint_states)`` rather than the canonical states.

    Gravity comes from the fused quaternion's tilt, which the accelerometer keeps
    drift-free and which, unlike raw accel, is not thrown off by the hand's own
    acceleration. Pass quat=None to fall back to the low-passed accelerometer.
    Yaw comes from the quaternion relative to the first sample (drift over a 1-3 s sign
    is a few degrees with a calibrated gyro). Sanity-check the quaternion convention
    once: with the glove still, quat_to_matrix(q) @ accel should be about (0, 0, +1 g).
    """
    t = np.asarray(t, dtype=float)
    gyro = np.asarray(gyro, dtype=float) - (0.0 if gyro_bias is None else np.asarray(gyro_bias))
    n = len(t)
    states = np.zeros((n, STATE_DIM))
    for f in range(5):
        states[:, f] = apply_flex_calibration(flex_res[:, f], flex_coeffs[f])
    states[:, 8:11] = gyro @ R_mount.T
    down = np.array([0.0, 0.0, -1.0])  # Madgwick's earth z points up
    if quat is None:
        # At rest an accelerometer reads +1 g UP, so down = -accel.
        acc = smooth(np.asarray(accel, dtype=float), accel_lowpass)
        states[:, 5:8] = _unit(-(acc @ R_mount.T))
        return mirror_state(states) if handedness.lower().startswith("l") else states
    R0 = quat_to_matrix(quat[0]) @ R_mount.T  # earth <- hand
    for k in range(n):
        Rk = quat_to_matrix(quat[k]) @ R_mount.T
        states[k, 5:8] = Rk.T @ down
        states[k, 11] = twist_angle(Rk @ R0.T, down)
    if handedness.lower().startswith("l"):
        states = mirror_state(states)
    return states


def mirror_state(states):
    """Reflect hand states left <-> right (hand-frame x -> -x).

    Directions (gravity) transform as M v; angular velocity is a pseudovector and
    transforms as det(M) M w = -M w; yaw changes sign; bends are unchanged.
    """
    s = np.array(states, dtype=float)
    s[..., 5:8] = s[..., 5:8] @ MIRROR.T
    s[..., 8:11] = -(s[..., 8:11] @ MIRROR.T)
    s[..., 11] *= -1.0
    return s


# --------------------------------------------------------------------------- #
# Time alignment and resampling
# --------------------------------------------------------------------------- #

def resample(t_src, x_src, t_dst, lowpass_window=None):
    """Linear-interpolate a (T, D) signal onto new timestamps.

    Going from 100 Hz glove to 30 fps video, low-pass first (lowpass_window ~ 100/30 ~ 3
    samples) so the gyro is band-limited like the differentiated keypoints.
    Re-normalise unit vectors (gravity) after interpolating.
    """
    x = np.asarray(x_src, dtype=float)
    if lowpass_window:
        x = smooth(x, lowpass_window)
    x2 = x.reshape(len(x), -1)
    out = np.stack([np.interp(t_dst, t_src, x2[:, d]) for d in range(x2.shape[1])], axis=1)
    return out.reshape((len(t_dst),) + x.shape[1:])


def estimate_lag(sig_a, sig_b, rate, max_lag_s=1.0):
    """Lag (seconds) to ADD to b's timestamps so it lines up with a.

    Both signals must already be on the same uniform grid at ``rate`` Hz. Feed it
    rotation-invariant signals so R_mount is not needed yet: |gyro| vs |omega_keypoints|,
    or sum of flex bends vs sum of keypoint bends. A positive result means an event shows
    up later in a than in b.
    """
    a = np.asarray(sig_a, dtype=float)
    b = np.asarray(sig_b, dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    max_lag = int(max_lag_s * rate)
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag]
        else:
            x, y = a[:lag], b[-lag:]
        m = min(len(x), len(y))
        if m < 10:
            continue
        c = float(np.dot(x[:m], y[:m]) / m)
        if c > best:
            best, best_lag = c, lag
    return best_lag / rate


# --------------------------------------------------------------------------- #
# Recognition helpers: time normalisation, DTW, confusability ranking
# --------------------------------------------------------------------------- #

def time_normalize(states, n=32):
    """Resample a (T, 12) sign to exactly n frames and re-normalise gravity."""
    s = np.asarray(states, dtype=float)
    u = np.linspace(0.0, 1.0, len(s))
    out = resample(u, s, np.linspace(0.0, 1.0, n))
    out[:, 5:8] = _unit(out[:, 5:8])
    return out


def feature_weights(bend=1.0, gravity=1.0, omega=0.3, yaw=0.5):
    """Per-dimension weights so the groups contribute comparably (bend is radians,
    gravity is unit-length, omega is rad/s and noisier). Tune on validation data."""
    return np.array([bend] * 5 + [gravity] * 3 + [omega] * 3 + [yaw])


def dtw(a, b, band=0.2, w=None):
    """Dynamic time warping distance between (Ta, D) and (Tb, D), Sakoe-Chiba band.

    band: fraction of the longer length. Returns the path cost / (Ta + Tb).
    Plain Python inner loop: fine for 32-frame signs and a few hundred templates;
    prefilter by handshape before running it against a large vocabulary.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if w is not None:
        a, b = a * w, b * w
    n, m = len(a), len(b)
    r = max(int(band * max(n, m)), abs(n - m))
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - r), min(m, i + r) + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    return float(D[n, m] / (n + m))


def confusability(templates, w=None, n=32):
    """Rank signs by how distinct they are in glove space.

    templates: dict sign -> list of (T, 12) hand-state sequences (converted from the
    dataset, so they contain only what the glove can measure). For every sign returns
    (nearest_other_sign, margin) where

        margin = d(sign, nearest other sign) / median within-sign distance

    using fixed-length Euclidean distance on time-normalised sequences (fast; re-score
    the top candidates with ``dtw`` if needed). margin < ~1 means the glove cannot
    reliably separate the pair. Sorted most-confusable first.
    """
    w = feature_weights() if w is None else w
    names = list(templates)
    seqs = {k: [time_normalize(s, n) * w for s in v] for k, v in templates.items()}
    means = np.stack([np.mean(seqs[k], axis=0).ravel() for k in names])
    spread = {}
    for k in names:
        X = np.stack([s.ravel() for s in seqs[k]])
        d = np.linalg.norm(X[:, None] - X[None], axis=-1)
        spread[k] = float(np.median(d[np.triu_indices(len(X), 1)])) if len(X) > 1 else np.nan
    fallback = np.nanmedian(list(spread.values())) if any(np.isfinite(list(spread.values()))) else 1.0
    D = np.linalg.norm(means[:, None] - means[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    out = []
    for i, k in enumerate(names):
        j = int(np.argmin(D[i]))
        s = spread[k] if np.isfinite(spread[k]) and spread[k] > 0 else fallback
        out.append((k, names[j], float(D[i, j] / s)))
    return sorted(out, key=lambda r: r[2])


def simulate_glove(states, rng=None, flex_noise=0.05, gyro_noise=0.05, gravity_noise=0.03,
                   bend_gain=(0.85, 1.15), yaw_drift=0.05):
    """Augment converted dataset states so they look like real glove output.

    Clips negative bend (flex sensors barely respond to backward bending), applies a
    random per-finger gain (sensor placement and fit vary), and adds noise and a small
    yaw drift (rad/s). Use it while training on the converted dataset.
    """
    rng = np.random.default_rng() if rng is None else rng
    s = np.array(states, dtype=float)
    s[:, :5] = np.maximum(s[:, :5], 0.0) * rng.uniform(*bend_gain, size=5)
    s[:, :5] += rng.normal(0, flex_noise, s[:, :5].shape)
    s[:, 5:8] = _unit(s[:, 5:8] + rng.normal(0, gravity_noise, s[:, 5:8].shape))
    s[:, 8:11] += rng.normal(0, gyro_noise, s[:, 8:11].shape)
    s[:, 11] += np.linspace(0, rng.normal(0, yaw_drift), len(s))
    return s
