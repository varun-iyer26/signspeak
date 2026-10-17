// Rule-based ASL fingerspelling classifier (static letters A-Y, no J; Z also needs motion).
//
// Approach: turn the 21 landmarks into a handful of rotation-invariant measurements,
// then give every letter a fuzzy score in [0, 1] built from weighted conditions.
// The best-scoring letter wins; confidence also reflects the margin to the runner-up.
//
// Measurements (all scaled by the palm size S = wrist -> middle-MCP distance):
//   - per finger: PIP and DIP joint angles, tip-to-MCP reach, tip-to-wrist lift,
//     combined into one extension value `ext` (0 = fully curled, 1 = straight)
//   - thumb: IP/MCP angles, distance to the palm centre, position in the palm frame
//   - palm frame: `u` runs across the palm (0 at the index MCP, 1 at the pinky MCP),
//     which makes "thumb between index and middle" etc. independent of hand rotation
//   - angles between neighbouring fingers (together vs spread) and tip-to-tip distances
//   - index direction in the image (sideways for G/H, downward for P/Q)

import { toMetric } from './features.js';

export const LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('');

// ---- small vector / fuzzy helpers -----------------------------------------

const clamp01 = (x) => Math.min(1, Math.max(0, x));
/** 0 at a, 1 at b, linear in between. Works in both directions (a > b inverts). */
const ramp = (x, a, b) => clamp01((x - a) / (b - a));
/** Bell: 0 at lo, 1 at peak, 0 at hi. */
const tri = (x, lo, peak, hi) => Math.min(ramp(x, lo, peak), ramp(x, hi, peak));
const avg = (...xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
/** Mean pulled towards the weakest value, so one wrong finger cannot be hidden by three right ones. */
const strict = (...xs) => 0.5 * avg(...xs) + 0.5 * Math.min(...xs);

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const len = (a) => Math.hypot(a.x, a.y, a.z);
const dist = (a, b) => len(sub(a, b));
const scale = (a, s) => ({ x: a.x * s, y: a.y * s, z: a.z * s });
const unit = (a) => {
  const l = len(a) || 1;
  return scale(a, 1 / l);
};
const cosBetween = (a, b) => dot(unit(a), unit(b));
const angleDeg = (a, b, c) => (Math.acos(Math.max(-1, Math.min(1, cosBetween(sub(a, b), sub(c, b))))) * 180) / Math.PI;

/**
 * Weighted mean of fuzzy conditions [value, weight]. Conditions with weight >= 2 are
 * "essential": the result is also scaled down when any of them is poorly met, so a
 * letter cannot win by satisfying only its minor details.
 */
function blend(parts) {
  let sum = 0;
  let weights = 0;
  let worst = 1;
  for (const [value, weight] of parts) {
    sum += value * weight;
    weights += weight;
    if (weight >= 2 && value < worst) worst = value;
  }
  return (sum / weights) * (0.5 + 0.5 * worst);
}

// ---- measurements -----------------------------------------------------------

// MCP, PIP, DIP, TIP landmark indices and typical finger length relative to palm size.
const FINGER_DEFS = [
  { ids: [5, 6, 7, 8], ref: 0.78 }, // index
  { ids: [9, 10, 11, 12], ref: 0.9 }, // middle
  { ids: [13, 14, 15, 16], ref: 0.83 }, // ring
  { ids: [17, 18, 19, 20], ref: 0.65 }, // pinky
];

function measureFinger(p, S, { ids: [mcp, pip, dip, tip], ref }) {
  const pipAngle = angleDeg(p[mcp], p[pip], p[dip]); // 180 = straight
  const dipAngle = angleDeg(p[pip], p[dip], p[tip]);
  const reach = dist(p[tip], p[mcp]) / (S * ref); // tip-vs-MCP distance, ~1 when straight
  const lift = dist(p[tip], p[0]) / dist(p[mcp], p[0]); // tip farther from wrist than the knuckle?
  const ext = 0.3 * ramp(pipAngle, 105, 165) + 0.2 * ramp(dipAngle, 115, 165) + 0.25 * ramp(reach, 0.5, 0.9) + 0.25 * ramp(lift, 1.1, 1.6);
  return { ext, dir: unit(sub(p[tip], p[mcp])) };
}

function analyze(landmarks, size) {
  const p = toMetric(landmarks, size);
  const S = dist(p[0], p[9]);
  if (S < 1e-6) return null;

  // Palm frame: `up` along the hand, `side` across the knuckles (index -> pinky).
  const up = unit(sub(p[9], p[0]));
  const across = sub(p[17], p[5]);
  const side = unit(sub(across, scale(up, dot(across, up))));
  const palmWidth = Math.max(dist(p[5], p[17]), 1e-6);
  const u = (i) => dot(sub(p[i], p[5]), side) / palmWidth;
  const v = (i) => dot(sub(p[i], p[0]), up) / S;
  const d = (i, j) => dist(p[i], p[j]) / S;

  const [I, M, R, P] = FINGER_DEFS.map((def) => measureFinger(p, S, def));
  const angleBetween = (a, b) => (Math.acos(Math.max(-1, Math.min(1, dot(a.dir, b.dir)))) * 180) / Math.PI;

  // Thumb: straightness, direction, and how far it sits from the palm centre.
  const thumbExt = 0.5 * ramp(angleDeg(p[2], p[3], p[4]), 125, 170) + 0.5 * ramp(angleDeg(p[1], p[2], p[3]), 135, 170);
  const thumbDir = unit(sub(p[4], p[2]));
  const palmCentre = scale(
    [0, 5, 9, 13, 17].reduce((s, i) => ({ x: s.x + p[i].x, y: s.y + p[i].y, z: s.z + p[i].z }), { x: 0, y: 0, z: 0 }),
    1 / 5,
  );
  const thumbPalm = dist(p[4], palmCentre) / S; // ~0.5 folded across the palm, >1.1 spread out

  // Index-middle crossing: tip separation relative to knuckle separation.
  // >1 spread (V), ~0.5-1 parallel (U), <=0 tips swapped sides (R).
  const knuckleGap = u(9) - u(5);
  const crossRatio = Math.abs(knuckleGap) > 0.08 ? (u(12) - u(8)) / knuckleGap : 1;

  // Index direction in the image plane, for orientation-dependent letters.
  const dx = p[8].x - p[5].x;
  const dy = p[8].y - p[5].y;
  const planar = Math.abs(dx) + Math.abs(dy) || 1;
  const sideways = ramp(Math.abs(dx) / planar, 0.45, 0.65); // G, H: pointing left/right
  const down = dy > 0 ? ramp(dy / planar, 0.5, 0.7) : 0; // P, Q: pointing down

  // Thumb in front of the curled fingers (S) vs tucked underneath (E, M, N, T).
  // z is noisy, so this is only ever a minor condition.
  const thumbInFront = ramp((p[10].z - p[4].z) / S, -0.1, 0.2);

  return {
    I, M, R, P,
    thumbExt, thumbDir, thumbPalm,
    thumbAcross: ramp(thumbPalm, 0.95, 0.6),
    thumbOut: ramp(thumbPalm, 0.7, 1.05),
    thumbInFront,
    u4: u(4), v4: v(4),
    d,
    spreadIM: angleBetween(I, M),
    spreadMR: angleBetween(M, R),
    maxSpread: Math.max(angleBetween(I, M), angleBetween(M, R), angleBetween(R, P)),
    crossRatio, sideways, down,
  };
}

// ---- letter rules -----------------------------------------------------------

const curled = (f) => ramp(f.ext, 0.5, 0.2); // tucked into the palm
const extended = (f) => ramp(f.ext, 0.5, 0.85); // straight
const hooked = (f) => Math.min(ramp(f.ext, 0.04, 0.18), ramp(f.ext, 0.65, 0.4)); // claw / partly bent
const curved = (f) => tri(f.ext, 0.2, 0.55, 0.9); // C-shaped

function scoreLetters(m) {
  const { I, M, R, P, d } = m;
  const fist = strict(curled(I), curled(M), curled(R), curled(P)); // all four fingers tucked
  const handsTogether = ramp(m.maxSpread, 22, 9); // fingers pressed together
  const upright = (1 - m.sideways) * (1 - m.down);

  // G/Q share a handshape: index out, thumb parallel to it, other fingers curled.
  // Orientation tells them apart: sideways = G, downward = Q.
  const gShape = blend([
    [extended(I), 2],
    [avg(curled(M), curled(R), curled(P)), 2],
    [ramp(m.thumbExt, 0.3, 0.6), 1],
    [ramp(cosBetween(m.thumbDir, I.dir), 0.4, 0.85), 1.5],
    [tri(d(4, 8), 0.1, 0.45, 1.1), 1],
  ]);

  // K/P share a handshape: index and middle spread, thumb touching the middle finger's
  // middle joint, ring and pinky curled. Pointing up/sideways = K, downward = P.
  const kShape = blend([
    [extended(I), 2],
    [extended(M), 2],
    [avg(curled(R), curled(P)), 2],
    [ramp(m.spreadIM, 8, 22), 1.5],
    [ramp(d(4, 10), 0.6, 0.3), 2.5],
  ]);

  const s = {};

  // Fist family. They differ almost only in where the thumb sits (u4 = position across the palm).
  s.A = blend([
    [fist, 3],
    [ramp(m.thumbExt, 0.35, 0.7), 1.5], // thumb straight, resting on the side of the index
    [ramp(m.u4, 0.3, 0.05), 2], // thumb beside the index, not across the fingers
    [ramp(m.thumbPalm, 1.1, 0.85), 1.5], // ...and not stuck out sideways (that is L or Y)
    [ramp(d(4, 6), 0.6, 0.35), 1],
    [ramp(m.v4, 0.45, 0.8), 1],
  ]);
  s.S = blend([
    [fist, 3],
    [m.thumbAcross, 2], // thumb wraps across the front of the fingers
    [tri(m.u4, 0.05, 0.45, 1.0), 1],
    [m.thumbInFront, 1],
  ]);
  s.T = blend([
    [fist, 3],
    [tri(m.u4, 0.05, 0.25, 0.5), 2], // thumb poking out between index and middle
    [ramp(Math.min(d(4, 6), d(4, 10)), 0.5, 0.3), 1.5],
    [m.thumbAcross, 1],
    [1 - m.thumbInFront, 0.5],
  ]);
  s.N = blend([
    [fist, 3],
    [tri(m.u4, 0.3, 0.55, 0.85), 2], // thumb under index + middle
    [m.thumbAcross, 1],
    [1 - m.thumbInFront, 0.5],
  ]);
  s.M = blend([
    [fist, 3],
    [tri(m.u4, 0.55, 0.85, 1.3), 2], // thumb under index, middle and ring
    [m.thumbAcross, 1],
    [1 - m.thumbInFront, 0.5],
  ]);
  s.E = blend([
    [avg(hooked(I), hooked(M), hooked(R), hooked(P)), 2.5], // fingertips bent down onto the thumb
    [ramp(avg(d(4, 8), d(4, 12), d(4, 16)), 0.8, 0.4), 2],
    [m.thumbAcross, 1],
    [1 - m.thumbInFront, 0.5],
  ]);

  // Open / flat hands.
  s.B = blend([
    [strict(extended(I), extended(M), extended(R), extended(P)), 3],
    [handsTogether, 1.5],
    [m.thumbAcross, 2], // thumb folded over the palm
  ]);

  // Curved shapes: C leaves a gap between thumb and fingers, O closes it.
  s.C = blend([
    [avg(curved(I), curved(M), curved(R), curved(P)), 3],
    [tri(d(4, 8), 0.3, 0.75, 1.4), 2],
    [tri(m.thumbPalm, 0.6, 1.0, 1.6), 1],
    [ramp(m.maxSpread, 30, 12), 0.5],
  ]);
  s.O = blend([
    [avg(...[I, M, R, P].map((f) => tri(f.ext, 0.1, 0.45, 0.8))), 2],
    [ramp(d(4, 8), 0.45, 0.2), 3], // thumb tip touches index tip
    [ramp(d(4, 12), 0.6, 0.3), 1.5],
    [ramp(m.maxSpread, 30, 12), 0.5],
  ]);

  // Pinch shapes: D (index up, thumb on middle fingertip), F (OK sign).
  s.D = blend([
    [extended(I), 3],
    [avg(ramp(M.ext, 0.7, 0.35), ramp(R.ext, 0.7, 0.35), ramp(P.ext, 0.7, 0.35)), 2],
    [ramp(d(4, 12), 0.55, 0.25), 3],
    [ramp(d(4, 8), 0.25, 0.5), 0.5],
  ]);
  s.F = blend([
    [avg(extended(M), extended(R), extended(P)), 2.5],
    [ramp(d(4, 8), 0.4, 0.18), 3], // thumb and index tips touch
  ]);

  // Index-pointing family.
  s.G = gShape * (0.15 + 0.85 * m.sideways);
  s.Q = gShape * m.down;
  s.L = blend([
    [extended(I), 2.5],
    [avg(curled(M), curled(R), curled(P)), 2],
    [m.thumbOut, 2],
    [ramp(m.thumbExt, 0.4, 0.7), 1],
    [ramp(Math.abs(cosBetween(m.thumbDir, I.dir)), 0.85, 0.45), 2], // thumb ~perpendicular to index
  ]);
  s.X = blend([
    [tri(I.ext, 0.08, 0.35, 0.7), 3], // index hooked
    [avg(curled(M), curled(R), curled(P)), 2],
    [ramp(I.ext - avg(M.ext, R.ext, P.ext), 0.05, 0.22), 2],
    [m.thumbAcross, 1],
  ]);

  // Pinky family.
  s.I = blend([
    [extended(P), 3],
    [avg(curled(I), curled(M), curled(R)), 2],
    [ramp(m.thumbPalm, 1.0, 0.65), 1.5],
  ]);
  s.Y = blend([
    [extended(P), 2.5],
    [avg(curled(I), curled(M), curled(R)), 2],
    [m.thumbOut, 2.5],
    [ramp(m.thumbExt, 0.4, 0.7), 1],
  ]);

  // Two-finger family.
  s.K = kShape * (1 - m.down);
  s.P = kShape * m.down;
  s.H = blend([
    [extended(I), 2],
    [extended(M), 2],
    [avg(curled(R), curled(P)), 2],
    [ramp(m.spreadIM, 22, 9), 1.5], // index and middle together
    [m.sideways, 3], // held horizontally
  ]);
  s.U = blend([
    [extended(I), 2],
    [extended(M), 2],
    [avg(curled(R), curled(P)), 2],
    [ramp(m.spreadIM, 20, 9), 2], // together, parallel
    [ramp(m.crossRatio, 0.3, 0.6), 1],
    [upright, 1],
  ]);
  s.V = blend([
    [extended(I), 2],
    [extended(M), 2],
    [avg(curled(R), curled(P)), 2],
    [ramp(m.spreadIM, 12, 26), 3], // clearly spread
    [ramp(d(4, 10), 0.3, 0.55), 1], // thumb not touching the middle finger (that would be K)
    [ramp(m.thumbPalm, 1.0, 0.7), 0.5],
    [upright, 0.5],
  ]);
  s.R = blend([
    [extended(I), 2],
    [extended(M), 2],
    [avg(curled(R), curled(P)), 2],
    [ramp(m.crossRatio, 0.5, 0.0), 4], // fingertips swapped sides: index crossed over middle
    [ramp(d(8, 12), 0.35, 0.15), 1.5],
    [m.thumbAcross, 0.5],
  ]);
  s.W = blend([
    [extended(I), 2.5],
    [extended(M), 2.5],
    [extended(R), 2.5],
    [ramp(P.ext, 0.6, 0.3), 2], // pinky held down by the thumb
    [avg(ramp(m.spreadIM, 8, 18), ramp(m.spreadMR, 8, 18)), 1.5],
    [ramp(d(4, 20), 0.8, 0.4), 1.5],
  ]);

  return s;
}

/**
 * Classify one hand.
 * @param {Array<{x:number,y:number,z:number}>} landmarks 21 normalised MediaPipe landmarks
 * @param {{width:number,height:number}} size video size in pixels (fixes the aspect ratio)
 * @returns {{letter:string, confidence:number}|null}
 */
export function classifyLetter(landmarks, size) {
  if (!landmarks || landmarks.length < 21) return null;
  const measurements = analyze(landmarks, size);
  if (!measurements) return null;

  const scores = scoreLetters(measurements);
  let best = null;
  let bestScore = 0;
  let second = 0;
  for (const letter of LETTERS) {
    const score = scores[letter];
    if (score > bestScore) {
      second = bestScore;
      bestScore = score;
      best = letter;
    } else if (score > second) {
      second = score;
    }
  }
  if (!best || bestScore < 0.5) return null;

  // Ambiguous shapes (e.g. M vs N, U vs V) lose up to 25% confidence.
  const margin = clamp01((bestScore - second) / 0.2);
  return { letter: best, confidence: Math.min(1, bestScore * (0.75 + 0.25 * margin)) };
}
