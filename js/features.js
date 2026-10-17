// Landmark helpers shared by the fingerspelling rules and the teachable k-NN.

export const FEATURE_LENGTH = 126; // 2 hands x 21 landmarks x (x, y, z)

/**
 * Convert MediaPipe's normalised landmarks into one isotropic space.
 * x is normalised by the frame width and y by the frame height, so without this
 * a hand would look squashed in a non-square frame. z shares x's scale.
 * Units: frame widths.
 */
export function toMetric(landmarks, size) {
  const w = size?.width || 1;
  const h = size?.height || 1;
  const yScale = h / w;
  return landmarks.map((l) => ({ x: l.x, y: l.y * yScale, z: l.z }));
}

/** One hand: 21 landmarks relative to the wrist, scaled by wrist -> middle-MCP distance. 63 numbers. */
function normalizeHand(landmarks, size, mirrored) {
  const pts = toMetric(landmarks, size);
  const wrist = pts[0];
  const mcp = pts[9];
  const scale = Math.hypot(mcp.x - wrist.x, mcp.y - wrist.y, mcp.z - wrist.z);
  if (scale < 1e-6) return null;
  const out = new Array(63);
  // When the video is mirrored (front camera) flip x so a sign recorded on
  // either camera produces the same vector.
  const flip = mirrored ? -1 : 1;
  for (let i = 0; i < 21; i++) {
    out[i * 3] = (flip * (pts[i].x - wrist.x)) / scale;
    out[i * 3 + 1] = (pts[i].y - wrist.y) / scale;
    out[i * 3 + 2] = (pts[i].z - wrist.z) / scale;
  }
  return out;
}

/**
 * Build the 126-value feature vector for the current frame.
 * Hands are ordered by handedness (Right first, then Left); a missing hand is
 * zero-filled. Returns null when no usable hand is visible.
 * @param {Array<{landmarks:Array, handedness:string}>} hands
 * @param {boolean} mirrored
 * @param {{width:number,height:number}} size video size
 */
export function buildFeatureVector(hands, mirrored, size) {
  const vec = new Float32Array(FEATURE_LENGTH);
  const slots = [false, false];
  // Right hands first so they take slot 0 when the model reports two of a kind.
  const ordered = [...hands].sort((a, b) => (a.handedness === 'Right' ? 0 : 1) - (b.handedness === 'Right' ? 0 : 1));
  let used = 0;
  for (const hand of ordered) {
    let slot = hand.handedness === 'Right' ? 0 : 1;
    if (slots[slot]) slot = 1 - slot;
    if (slots[slot]) continue;
    const values = normalizeHand(hand.landmarks, size, mirrored);
    if (!values) continue;
    vec.set(values, slot * 63);
    slots[slot] = true;
    used++;
  }
  return used ? vec : null;
}
