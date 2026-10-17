"""Self-checks for hand_state.py on synthetic data. Run: python research/test_hand_state.py
(also works under pytest). A synthetic hand is posed from known joint angles and
rotated at a known rate; a fake glove stream is generated from the same motion.
Both sides must land on the same hand state."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import hand_state as hs  # noqa: E402


def expm(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = hs.hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def matrix_to_quat(R):
    w = np.sqrt(max(1.0 + np.trace(R), 1e-12)) / 2.0
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def synthetic_hand(mcp=(0.4, 0.6, 0.8, 1.0), pip=(0.5, 0.7, 0.9, 1.1), dip=(0.2, 0.3, 0.4, 0.5),
                   thumb=(0.3, 0.4)):
    """Right hand in its own frame (x thumb side, y fingers, z out of palm)."""
    p = np.zeros((21, 3))
    xs = [0.35, 0.0, -0.2, -0.4]
    for f, (base, x0) in enumerate(zip((5, 9, 13, 17), xs)):
        p[base] = [x0, 1.0, 0.0]
        d, ang = np.array([0.0, 1.0, 0.0]), 0.0
        for k, (theta, length) in enumerate(zip((mcp[f], pip[f], dip[f]), (0.45, 0.28, 0.2))):
            ang += theta  # flexion rotates y toward +z about +x
            d = np.array([0.0, np.cos(ang), np.sin(ang)])
            p[base + k + 1] = p[base + k] + length * d
    # Thumb roughly in the x-y plane, flexing about z toward the palm.
    p[1] = [0.3, 0.25, 0.05]
    d = hs._unit(np.array([0.7, 0.5, 0.0]))
    p[2] = p[1] + 0.35 * d
    ang = 0.0
    for k, (theta, length) in enumerate(zip(thumb, (0.3, 0.25))):
        ang += theta
        dk = expm(np.array([0, 0, 1.0]) * ang) @ d
        p[3 + k] = p[2 + k] + length * dk
    return p


def motion(T=90, fps=30.0, omega=(0.8, -0.5, 1.2)):
    """Hand pose rotated at a constant body-frame rate; returns t, camera<-hand rotations."""
    t = np.arange(T) / fps
    R0 = expm(np.array([0.3, 0.2, -0.4]))
    Rs = np.stack([R0 @ expm(np.array(omega) * ti) for ti in t])
    return t, Rs


def test_joint_angles_recovered():
    p = synthetic_hand()
    a = hs.joint_angles(p)
    np.testing.assert_allclose(a["index"], [0.4, 0.5, 0.2], atol=1e-6)
    np.testing.assert_allclose(a["pinky"], [1.0, 1.1, 0.5], atol=1e-6)
    np.testing.assert_allclose(a["thumb"], [0.3, 0.4], atol=1e-6)
    np.testing.assert_allclose(hs.flex_bend(a), [0.7, 0.9, 1.3, 1.7, 2.1], atol=1e-6)


def test_hand_frame_is_identity_in_own_frame():
    np.testing.assert_allclose(hs.hand_frame(synthetic_hand()), np.eye(3), atol=1e-9)


def test_keypoints_state_and_mirror():
    p = synthetic_hand()
    t, Rs = motion()
    seq = np.stack([p @ R.T for R in Rs]) + np.array([0.1, -0.2, 0.5])
    s = hs.keypoints_to_state(seq, t, smooth_window=1)
    g_true = np.stack([R.T @ hs.GRAVITY_CAMERA for R in Rs])
    np.testing.assert_allclose(s[:, 5:8], g_true, atol=1e-9)
    np.testing.assert_allclose(s[1:-1, 8:11], np.tile([0.8, -0.5, 1.2], (len(t) - 2, 1)), atol=2e-3)
    # A mirrored (left) hand doing the mirrored motion gives the same canonical state.
    seq_left = seq.copy()
    seq_left[..., 0] *= -1
    s_left = hs.keypoints_to_state(seq_left, t, handedness="Left", smooth_window=1)
    np.testing.assert_allclose(s_left, s, atol=1e-9)


def test_glove_matches_keypoints():
    p = synthetic_hand()
    t, Rs = motion(T=300, fps=100.0)
    seq = np.stack([p @ R.T for R in Rs])
    s_kp = hs.keypoints_to_state(seq, t, smooth_window=1)

    R_mount = expm(np.array([0.2, -1.1, 0.4]))           # sensor -> hand
    R_earth_cam = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # camera y (down) -> earth -z
    quats, acc, gyro = [], [], []
    for R in Rs:
        R_es = R_earth_cam @ R @ R_mount                   # earth <- sensor
        quats.append(matrix_to_quat(R_es))
        acc.append(R_es.T @ np.array([0, 0, 1.0]))         # at rest: +1 g up, sensor frame
        gyro.append(R_mount.T @ np.array([0.8, -0.5, 1.2]))
    # Flex: pretend R = 25k + 30k * bend (ohms), then fit it back from the keypoint bends.
    sweep = np.linspace(0.0, 2.5, 50)                     # calibration session: fist <-> open
    coeffs = [hs.fit_flex_calibration(25_000 + 30_000 * sweep, sweep, degree=2) for _ in range(5)]
    res = 25_000 + 30_000 * s_kp[:, :5]
    s_gl = hs.glove_to_state(t, res, np.array(acc), np.array(gyro), np.array(quats), R_mount, coeffs,
                             accel_lowpass=1)
    np.testing.assert_allclose(s_gl[:, :8], s_kp[:, :8], atol=1e-6)
    np.testing.assert_allclose(s_gl[1:-1, 8:11], s_kp[1:-1, 8:11], atol=2e-3)
    np.testing.assert_allclose(s_gl[:, 11], s_kp[:, 11], atol=1e-6)
    s_acc = hs.glove_to_state(t, res, np.array(acc), np.array(gyro), None, R_mount, coeffs, accel_lowpass=1)
    np.testing.assert_allclose(s_acc[:, 5:8], s_kp[:, 5:8], atol=1e-6)

    # R_mount recovered from paired gravity + angular-velocity vectors.
    a = np.concatenate([-np.array(acc), np.array(gyro)])
    b = np.concatenate([s_kp[:, 5:8], s_kp[:, 8:11]])
    np.testing.assert_allclose(hs.solve_rotation(a, b), R_mount, atol=1e-3)


def test_left_glove_matches_left_keypoints():
    p = synthetic_hand()
    t, Rs = motion(T=200, fps=100.0)
    P = np.diag([-1.0, 1.0, 1.0])                        # camera-space reflection
    seq_left = np.stack([p @ R.T for R in Rs]) @ P        # a real left hand doing the mirrored motion
    s_kp = hs.keypoints_to_state(seq_left, t, handedness="Left", smooth_window=1)

    R_mount = expm(np.array([-0.7, 0.3, 0.9]))
    R_earth_cam = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
    quats, acc, gyro = [], [], []
    prev = None
    for k, R in enumerate(Rs):
        y, z = P @ R[:, 1], P @ R[:, 2]                    # left hand: fingers, out of palm
        R_left = np.stack([np.cross(y, z), y, z], axis=1)  # camera <- left-hand frame
        R_es = R_earth_cam @ R_left @ R_mount
        quats.append(matrix_to_quat(R_es))
        acc.append(R_es.T @ np.array([0, 0, 1.0]))
        if prev is not None:
            gyro.append(R_mount.T @ (hs.so3_log(prev.T @ R_left) * 100.0))
        prev = R_left
    gyro = np.array([gyro[0]] + gyro)
    sweep = np.linspace(0.0, 2.5, 50)
    coeffs = [hs.fit_flex_calibration(25_000 + 30_000 * sweep, sweep) for _ in range(5)]
    s_gl = hs.glove_to_state(t, 25_000 + 30_000 * s_kp[:, :5], np.array(acc), gyro, np.array(quats),
                             R_mount, coeffs, handedness="Left", accel_lowpass=1)
    np.testing.assert_allclose(s_gl[:, :8], s_kp[:, :8], atol=1e-6)
    np.testing.assert_allclose(s_gl[2:-1, 8:11], s_kp[2:-1, 8:11], atol=5e-3)
    np.testing.assert_allclose(s_gl[:, 11], s_kp[:, 11], atol=1e-6)


def test_lag_and_aspect():
    rate = 100.0
    rng = np.random.default_rng(0)
    x = hs.smooth(rng.normal(size=1000), 15)
    shift = 23
    a, b = np.r_[np.zeros(shift), x][:len(x)], x    # every event shows up 23 samples later in a
    lag = hs.estimate_lag(a, b, rate)
    assert abs(lag - shift / rate) < 1e-9, lag

    p = synthetic_hand()
    _, Rs = motion(T=40)
    seq = np.stack([p @ R.T for R in Rs])
    squashed = seq * np.array([1.0, 1 / 1.6, 1.0])
    _, aspect = hs.fix_aspect(squashed)
    assert abs(aspect - 1.6) < 0.05, aspect


def test_confusability_orders_pairs():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 30)
    base = np.zeros((30, 12))
    base[:, 5:8] = [0, 0, 1]
    a = base.copy(); a[:, :5] = 1.0
    b = base.copy(); b[:, :5] = 1.02           # nearly identical to a (MOTHER/FATHER-like)
    c = base.copy(); c[:, 6] = np.sin(t * 6)   # different motion
    templates = {k: [v + rng.normal(0, 0.02, v.shape) for _ in range(4)] for k, v in
                 {"A": a, "B": b, "C": c}.items()}
    ranking = hs.confusability(templates)
    assert {ranking[0][0], ranking[0][1]} == {"A", "B"}
    assert ranking[-1][0] == "C"
    assert hs.dtw(a, a) == 0.0 and hs.dtw(a, c) > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
