"""Load a calibrate.html capture into numpy arrays and run the first calibration steps.

numpy only. Usage::

    python3 research/load_capture.py signspeak-calib-....json

prints a summary per glove: sample rate, clock mapping, gyro bias (step 0), the
glove -> webcam time lag (step 3, via ``hand_state.estimate_lag``) and a static
R_mount (step 2, via ``hand_state.solve_rotation``). Captures made with the
mock glove carry their ground truth, which is printed alongside.

Capture schema (``signspeak.calibration-capture``, version 1). All times are
``performance.now()`` milliseconds in the browser unless noted::

    clock      {source, unit, time_origin_unix_ms, recording_start, recording_stop}
    wearer     {id, dominant_hand}
    video      {source: 'webcam'|'synthetic', width, height, mirrored_input: false,
                handedness_label: MediaPipe's, which assumes a mirrored image, mediapipe: {...}}
    glove_config  the page's GLOVE_CONFIG (UUIDs, packet fields and scales, adcMax, divider)
    frames     [{t, capture_t (camera capture time or null),
                 hands: [{label, score, world: 21x[x,y,z] metres, image: 21x[x,y,z] normalized}]}]
               one entry per processed video frame, including frames with no hands.
    gloves     [{id, name, hand: physical 'Right'|'Left', source: 'ble'|'mock',
                 samples: {t_device_us: uint32 micros() (wraps), t_recv_ms,
                           flex: Nx5 ADC, accel: Nx3 g, gyro: Nx3 rad/s, quat: Nx4 (w,x,y,z), ...}}]
    segments   [{step, title, t_start, t_end, planned_s, completed, sign, rep}]
               step in still, flex_fist, pose_palm_down, pose_fingers_down, pose_thumb_up,
               twists, free, sign.
    mock       null, or {camera_latency_ms, gloves: [{id, r_mount, gyro_bias, device_clock, ...}]}

Handedness: the browser never flips the frames it gives MediaPipe, but MediaPipe
labels hands as if the image were a mirrored selfie. So on these captures the
label is the opposite of the physical hand; ``hand_tracks`` undoes that and then
calls ``keypoints_to_state(..., handedness=physical, mirrored_input=False)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_state as hs  # noqa: E402

SCHEMA = "signspeak.calibration-capture"
STATIC_POSES = ("pose_palm_down", "pose_fingers_down", "pose_thumb_up")


def _other(hand):
    return "Left" if hand == "Right" else "Right"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_capture(path):
    """Read the JSON and convert every stream to numpy arrays (times in seconds)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file")
    if raw.get("version") != 1:
        raise ValueError(f"{path}: unsupported version {raw.get('version')}")

    cap = {
        "raw": raw,
        "video": raw["video"],
        "glove_config": raw["glove_config"],
        "mock": raw.get("mock"),
        "frame_t": np.array([fr["t"] for fr in raw["frames"]], dtype=float) / 1000.0,
        "segments": [
            {**s, "t_start": s["t_start"] / 1000.0, "t_end": s["t_end"] / 1000.0} for s in raw["segments"]
        ],
        "hands": hand_tracks(raw),
        "gloves": [load_glove(g, raw["glove_config"]) for g in raw["gloves"]],
    }
    return cap


def hand_tracks(raw):
    """Per PHYSICAL hand: the frames where it was detected.

    Returns {'Right': {...}, 'Left': {...}} (a hand that never appeared is left out),
    each with t (N,) s, world (N,21,3) m, image (N,21,3), score (N,), frame (N,) index.
    If MediaPipe reports two hands with the same label in a frame, the higher score wins.
    """
    mirrored = bool(raw["video"].get("mirrored_input", False))
    rows = {"Right": [], "Left": []}
    for k, fr in enumerate(raw["frames"]):
        best = {}
        for h in fr["hands"]:
            if h.get("label") not in ("Left", "Right") or len(h.get("world") or []) != 21:
                continue
            physical = h["label"] if mirrored else _other(h["label"])
            if physical not in best or h["score"] > best[physical]["score"]:
                best[physical] = h
        for physical, h in best.items():
            rows[physical].append((fr["t"] / 1000.0, h["world"], h["image"], h["score"], k))
    out = {}
    for hand, r in rows.items():
        if not r:
            continue
        t, world, image, score, frame = zip(*r)
        out[hand] = {
            "t": np.array(t), "world": np.array(world, dtype=float), "image": np.array(image, dtype=float),
            "score": np.array(score), "frame": np.array(frame),
        }
    return out


def unwrap_micros(t_us):
    """uint32 micros() -> monotonic microseconds (int64), undoing the 2^32 wrap."""
    t = np.asarray(t_us, dtype=np.int64)
    if len(t) == 0:
        return t
    steps = np.diff(t)
    wraps = np.concatenate([[0], np.cumsum(steps < -(2**31))])
    return t + wraps * 2**32


def device_to_browser(t_dev_s, t_recv_s):
    """Map device time to the browser clock: t_browser ~= a + b * t_dev.

    BLE delivery only ever adds delay, so the receive times sit ABOVE the true
    sample times. Fit the slope (clock rate) by least squares, then lower the line
    onto the fastest deliveries (the lower envelope). The remaining offset is the
    minimum BLE latency, a constant that the lag estimate absorbs.
    Returns (a, b, residual_ms) where residual is each sample's delivery delay above the envelope.
    """
    x = t_dev_s - t_dev_s[0]
    b, a = np.polyfit(x, t_recv_s, 1)
    resid = t_recv_s - (a + b * x)
    a += resid.min()
    a -= b * t_dev_s[0]
    return a, b, (t_recv_s - (a + b * t_dev_s)) * 1000.0


def load_glove(g, config):
    s = g["samples"]
    n = len(s["t_device_us"])
    t_dev = unwrap_micros(s["t_device_us"]) / 1e6
    t_recv = np.array(s["t_recv_ms"], dtype=float) / 1000.0
    out = {
        "id": g["id"], "name": g["name"], "hand": g["hand"], "source": g["source"], "n": n,
        "t_device": t_dev, "t_recv": t_recv,
        "flex_adc": np.array(s["flex"], dtype=float).reshape(n, 5),
        "accel": np.array(s["accel"], dtype=float).reshape(n, 3),
        "gyro": np.array(s["gyro"], dtype=float).reshape(n, 3),
        "quat": np.array(s["quat"], dtype=float).reshape(n, 4),
    }
    out["flex_res"] = hs.flex_resistance(out["flex_adc"], r_fixed=config.get("dividerOhms", 47_000.0),
                                         adc_max=config.get("adcMax", 4095), flex_on_top=config.get("flexOnTop", True))
    if n >= 2:
        a, b, delay_ms = device_to_browser(t_dev, t_recv)
        out["t"] = a + b * t_dev            # device time on the browser clock, seconds
        out["clock"] = {"offset_s": a, "rate": b, "delivery_delay_ms": delay_ms}
    else:
        out["t"] = t_recv.copy()
        out["clock"] = None
    return out


# --------------------------------------------------------------------------- #
# Calibration helpers
# --------------------------------------------------------------------------- #

def segments(cap, step):
    return [s for s in cap["segments"] if s["step"] == step]


def last_segment(cap, step):
    seg = segments(cap, step)
    return seg[-1] if seg else None


def window(t, seg, trim=0.0):
    """Boolean mask of t inside a segment, dropping `trim` (fraction) from each end."""
    d = (seg["t_end"] - seg["t_start"]) * trim
    return (t >= seg["t_start"] + d) & (t <= seg["t_end"] - d)


def keypoint_states(track, max_gap_s=0.25, smooth_window=5, hand="Right"):
    """``keypoints_to_state`` on each contiguous run of detections (gaps reset yaw)."""
    t = track["t"]
    states = np.full((len(t), hs.STATE_DIM), np.nan)
    breaks = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    for idx in np.split(np.arange(len(t)), breaks):
        if len(idx) < 3:
            continue
        states[idx] = hs.keypoints_to_state(track["world"][idx], t[idx], handedness=hand,
                                            mirrored_input=False, smooth_window=smooth_window)
    return states


def gyro_bias(glove, cap):
    seg = last_segment(cap, "still")
    if seg is None:
        return None
    m = window(glove["t"], seg, trim=0.15)
    return glove["gyro"][m].mean(axis=0) if m.sum() >= 10 else None


def estimate_glove_lag(glove, track, kp_states, cap, bias=None, rate=50.0, pad_s=1.5, max_lag_s=1.0):
    """Lag (s) to ADD to glove times so they line up with the webcam keypoints.

    Uses the rotation-invariant |gyro| vs |omega_keypoints| (so R_mount is not needed),
    over the last 'twists' segment if there is one, else over the whole overlap.
    """
    ok = ~np.isnan(kp_states[:, 0])
    t_kp, w_kp = track["t"][ok], np.linalg.norm(kp_states[ok, 8:11], axis=1)
    seg = last_segment(cap, "twists")
    if seg is not None:
        lo, hi = seg["t_start"] - pad_s, seg["t_end"] + pad_s
    else:
        lo, hi = max(t_kp[0], glove["t"][0]), min(t_kp[-1], glove["t"][-1])
    lo, hi = max(lo, t_kp[0], glove["t"][0]), min(hi, t_kp[-1], glove["t"][-1])
    if hi - lo < 2 * max_lag_s + 1.0:
        return None, None
    grid = np.arange(lo, hi, 1.0 / rate)
    gyro = glove["gyro"] - (0.0 if bias is None else bias)
    # Band-limit the 100 Hz gyro roughly like the smoothed 30 fps keypoints.
    w_gl = np.linalg.norm(hs.smooth(gyro, 5), axis=1)
    a = np.interp(grid, t_kp, w_kp)
    b = np.interp(grid, glove["t"], w_gl)
    lag = hs.estimate_lag(a, b, rate, max_lag_s=max_lag_s)
    return lag, (lo, hi)


def static_r_mount(glove, cap):
    """Wahba from the static poses: -accel (sensor frame) vs the known down direction (hand frame).

    Right hand frame (x thumb side): palm down -> down = +z, fingers down -> +y, thumb up -> -x.
    Left glove frame (x = pinky side, see hand_state.glove_to_state): thumb up -> +x.
    Returns (R_mount, n_poses) or (None, n).
    """
    down = {
        "pose_palm_down": [0.0, 0.0, 1.0],
        "pose_fingers_down": [0.0, 1.0, 0.0],
        "pose_thumb_up": [-1.0, 0.0, 0.0] if glove["hand"] == "Right" else [1.0, 0.0, 0.0],
    }
    a, b = [], []
    for step in STATIC_POSES:
        seg = last_segment(cap, step)
        if seg is None:
            continue
        m = window(glove["t"], seg, trim=0.2)
        if m.sum() < 10:
            continue
        a.append(hs._unit(-glove["accel"][m].mean(axis=0)))
        b.append(down[step])
    if len(a) < 2:
        return None, len(a)
    return hs.solve_rotation(np.array(a), np.array(b)), len(a)


def rotation_angle_deg(R):
    return float(np.degrees(np.linalg.norm(hs.so3_log(R))))


def _mock_truth(cap, glove):
    mock = cap.get("mock") or {}
    for g in mock.get("gloves", []):
        if g["id"] == glove["id"]:
            return g, mock.get("camera_latency_ms")
    return None, None


def expected_mock_lag(cap, glove):
    """For a synthetic-camera + mock-glove capture: the lag estimate_glove_lag should find."""
    truth, cam_ms = _mock_truth(cap, glove)
    if truth is None or cam_ms is None:
        return None
    clk = truth["device_clock"]
    dev = glove["t_device"] * 1e6  # unwrapped from the first recorded sample
    k = (((dev[0] - clk["dev0_us"]) % 2**32) + (dev - dev[0])) / 10_000.0  # mock sample index (10 ms period)
    t_true = (clk["t0_ms"] + k * 10.0 / (1 + clk["drift_ppm"] * 1e-6)) / 1000.0
    # Keypoints are stamped cam_ms after the motion; glove times are stamped (mapped - true) after it.
    return cam_ms / 1000.0 - float(np.mean(glove["t"] - t_true))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def summarize(cap):
    raw = cap["raw"]
    ft = cap["frame_t"]
    dur = ft[-1] - ft[0] if len(ft) > 1 else 0.0
    print(f"video: {raw['video']['source']} {raw['video']['width']}x{raw['video']['height']}, "
          f"{len(ft)} frames over {dur:.1f} s ({len(ft) / dur if dur else 0:.1f} fps)")
    for hand, tr in cap["hands"].items():
        print(f"  {hand} hand: detected in {len(tr['t'])} frames, mean score {tr['score'].mean():.2f}")
    steps = {}
    for s in cap["segments"]:
        steps.setdefault(s["step"], []).append(s)
    print("segments: " + ", ".join(f"{k} x{len(v)}" for k, v in steps.items()))
    signs = sorted({s["sign"] for s in cap["segments"] if s["step"] == "sign"})
    if signs:
        print("signs: " + ", ".join(f"{s} x{sum(1 for x in cap['segments'] if x.get('sign') == s)}" for s in signs))

    for glove in cap["gloves"]:
        print(f"\nglove {glove['name']} ({glove['hand']}, {glove['source']}): {glove['n']} samples")
        if glove["n"] < 20:
            print("  too few samples")
            continue
        span = glove["t_device"][-1] - glove["t_device"][0]
        print(f"  device rate {(glove['n'] - 1) / span:.1f} Hz over {span:.1f} s")
        clk = glove["clock"]
        d = clk["delivery_delay_ms"]
        print(f"  clock: rate {clk['rate']:.6f} ({(clk['rate'] - 1) * 1e6:+.0f} ppm), "
              f"BLE delay above envelope median {np.median(d):.1f} ms, p95 {np.percentile(d, 95):.1f} ms")
        truth, _ = _mock_truth(cap, glove)

        bias = gyro_bias(glove, cap)
        if bias is not None:
            line = f"  gyro bias (step 0): {np.round(bias, 4)} rad/s"
            if truth:
                line += f"   [mock truth {np.round(truth['gyro_bias'], 4)}]"
            print(line)

        track = cap["hands"].get(glove["hand"])
        if track is None or len(track["t"]) < 10:
            print(f"  no {glove['hand']}-hand keypoints; skipping lag")
        else:
            kp = keypoint_states(track, hand=glove["hand"])
            print(f"  keypoints_to_state: {np.sum(~np.isnan(kp[:, 0]))} states, "
                  f"bend range {np.nanmin(kp[:, :5]):.2f}..{np.nanmax(kp[:, :5]):.2f} rad")
            lag, span_s = estimate_glove_lag(glove, track, kp, cap, bias=bias)
            if lag is None:
                print("  lag: not enough overlapping data")
            else:
                where = "twists segment" if last_segment(cap, "twists") else "whole overlap"
                line = f"  lag (step 3, {where}): {lag * 1000:+.0f} ms to add to glove times"
                exp = expected_mock_lag(cap, glove)
                if exp is not None:
                    line += f"   [mock expected {exp * 1000:+.0f} ms]"
                print(line)

        R, n = static_r_mount(glove, cap)
        if R is None:
            print(f"  R_mount: needs at least 2 static poses (have {n})")
        else:
            line = f"  R_mount (step 2, {n} poses):\n" + "\n".join("    " + " ".join(f"{v:+.3f}" for v in row) for row in R)
            if truth:
                err = rotation_angle_deg(np.array(truth["r_mount"]).T @ R)
                line += f"\n    [mock truth differs by {err:.2f} deg]"
            print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("capture", help="JSON file exported by calibrate.html")
    args = ap.parse_args(argv)
    summarize(load_capture(args.capture))


if __name__ == "__main__":
    main()
