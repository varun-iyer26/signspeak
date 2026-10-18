// Glove calibration capture page (calibrate.html).
//
// Records, on one clock (performance.now(), ms):
//   * MediaPipe HandLandmarker results per video frame: worldLandmarks (metres),
//     normalized landmarks, handedness and score;
//   * every BLE sample from each glove, with the glove's own micros() timestamp
//     and the browser receive time;
//   * labelled segments for the calibration steps in
//     docs/research/glove-dataset-alignment.md §4.
// Export is one JSON file; research/load_capture.py reads it.

// ===========================================================================
// GLOVE BLE CONFIG. The firmware is not written yet: change this to match it.
// ===========================================================================
//
// One GATT service per glove with one NOTIFY characteristic. Each notification
// carries one or more samples packed back to back (byteLength must be a whole
// multiple of the sample size). A sample is the `fields` below, in order, with
// no padding, little-endian (the nRF52840's native order).
//
// Values on the wire are integers. The page multiplies each by `scale`, so the
// firmware sends round(value / scale). With the defaults, one sample is 34 bytes:
//
//   offset  type      field  meaning
//   0       uint32    t_us   micros() when the sample was taken (wraps at 2^32, ~71.6 min)
//   4       uint16 x5 flex   raw ADC counts, thumb..pinky (analogReadResolution(12) -> 0..4095)
//   14      int16 x3  accel  g,      scale 1/4096    (BMI270 at ±8 g: 4096 LSB/g)
//   20      int16 x3  gyro   rad/s,  scale π/180/16.4 (BMI270 at ±2000 °/s: 16.4 LSB per °/s)
//   26      int16 x4  quat   Madgwick w, x, y, z, sensor -> earth, earth z up; scale 1/16384 (Q14)
//
// 34 bytes is more than the 20-byte payload of the default 23-byte ATT MTU, so
// the firmware must accept a larger MTU (desktop Chrome asks for one) or the
// layout must shrink. Batching 2-4 samples per notification keeps 100 Hz
// reliable when the OS picks a 15-30 ms connection interval (macOS often does).
// Extra fields (for example a `contacts` bitmask for the copper pads) can be
// appended here; they are decoded and exported under their own name.
export const GLOVE_CONFIG = {
  namePrefix: 'SignGlove', // advertised name, e.g. 'SignGlove-R' / 'SignGlove-L' (the suffix picks the hand)
  serviceUuid: '6e5a0001-8f3c-4b6a-9d2e-5c1b7a0f3e21',
  sampleCharUuid: '6e5a0002-8f3c-4b6a-9d2e-5c1b7a0f3e21',
  littleEndian: true,
  sampleRateHz: 100,
  adcMax: 4095,
  dividerOhms: 47_000,
  flexOnTop: true, // Vcc -- flex -- A_n -- 47k -- GND
  fields: [
    { name: 't_us', type: 'u32', count: 1, scale: 1, unit: 'us (device micros())' },
    { name: 'flex', type: 'u16', count: 5, scale: 1, unit: 'ADC counts, thumb..pinky' },
    { name: 'accel', type: 'i16', count: 3, scale: 1 / 4096, unit: 'g, sensor frame' },
    { name: 'gyro', type: 'i16', count: 3, scale: Math.PI / 180 / 16.4, unit: 'rad/s, sensor frame' },
    { name: 'quat', type: 'i16', count: 4, scale: 1 / 16384, unit: 'w, x, y, z; sensor -> earth' },
  ],
};

import { HandTracker } from './hands.js';
import { Camera, describeCameraError } from './camera.js';
import { SimHand, SimCamera, MockGlove, TWIST_TIMES_S } from './calibrate-sim.js';

const SCHEMA = 'signspeak.calibration-capture';
const SCHEMA_VERSION = 1;
const MP_VERSION = '0.10.14'; // must match js/hands.js

// --- Packet codec -----------------------------------------------------------

const TYPES = {
  u8: [1, 'getUint8', 'setUint8'], i8: [1, 'getInt8', 'setInt8'],
  u16: [2, 'getUint16', 'setUint16'], i16: [2, 'getInt16', 'setInt16'],
  u32: [4, 'getUint32', 'setUint32'], i32: [4, 'getInt32', 'setInt32'],
  f32: [4, 'getFloat32', 'setFloat32'],
};

function layoutOf(config) {
  let offset = 0;
  const fields = config.fields.map((f) => {
    const [size, get, set] = TYPES[f.type];
    const out = { ...f, offset, size, get, set };
    offset += size * f.count;
    return out;
  });
  return { fields, sampleSize: offset, le: config.littleEndian };
}

const LAYOUT = layoutOf(GLOVE_CONFIG);

/** DataView with one or more packed samples -> [{t_us, flex: [...], ...}]. */
export function decodePacket(view, layout = LAYOUT) {
  if (view.byteLength === 0 || view.byteLength % layout.sampleSize !== 0) {
    throw new Error(`packet is ${view.byteLength} bytes, not a multiple of ${layout.sampleSize}`);
  }
  const out = [];
  for (let base = 0; base < view.byteLength; base += layout.sampleSize) {
    const s = {};
    for (const f of layout.fields) {
      const vals = [];
      for (let i = 0; i < f.count; i++) vals.push(view[f.get](base + f.offset + i * f.size, layout.le) * f.scale);
      s[f.name] = f.count === 1 ? vals[0] : vals;
    }
    out.push(s);
  }
  return out;
}

/** Inverse of decodePacket; the mock glove uses it so the decoder is exercised. */
export function encodePacket(samples, layout = LAYOUT) {
  const view = new DataView(new ArrayBuffer(samples.length * layout.sampleSize));
  samples.forEach((s, k) => {
    const base = k * layout.sampleSize;
    for (const f of layout.fields) {
      const vals = f.count === 1 ? [s[f.name]] : s[f.name];
      for (let i = 0; i < f.count; i++) {
        let v = vals[i] / f.scale;
        if (f.type !== 'f32') {
          v = Math.round(v);
          if (f.type === 'i16') v = Math.max(-32768, Math.min(32767, v));
        }
        view[f.set](base + f.offset + i * f.size, v, layout.le);
      }
    }
  });
  return view;
}

// --- Small helpers ----------------------------------------------------------

const $ = (id) => document.getElementById(id);
const r4 = (x) => Math.round(x * 1e4) / 1e4; // landmarks: 0.1 mm world, sub-pixel image
const r3 = (x) => Math.round(x * 1e3) / 1e3;
const r6 = (x) => Math.round(x * 1e6) / 1e6;
const params = new URLSearchParams(location.search);

let audio = null;
function beep(freq = 880, ms = 90) {
  try {
    audio ||= new AudioContext();
    const o = audio.createOscillator();
    const g = audio.createGain();
    o.frequency.value = freq;
    g.gain.value = 0.08;
    o.connect(g).connect(audio.destination);
    o.start();
    o.stop(audio.currentTime + ms / 1000);
  } catch {
    /* sound is optional */
  }
}

// --- Capture buffers ---------------------------------------------------------

const capture = {
  recording: false,
  startedAt: null,
  stoppedAt: null,
  exported: false,
  frames: [],
  segments: [],
};

// --- Gloves -----------------------------------------------------------------

const gloves = []; // {id, name, hand, source, device?, mock?, rec, live}
let nextGloveId = 1;

function emptyRec() {
  const rec = { t_device_us: [], t_recv_ms: [] };
  for (const f of LAYOUT.fields) if (f.name !== 't_us') rec[f.name] = [];
  return rec;
}

function addGlove({ name, hand, source, device = null, mock = null }) {
  const glove = {
    id: `glove${nextGloveId++}`, name, hand, source, device, mock,
    connected: true, rec: emptyRec(),
    live: { last: null, recvTimes: [], lastDevUs: null, drops: 0, count: 0, errors: 0 },
  };
  gloves.push(glove);
  renderGloves();
  return glove;
}

function onGlovePacket(glove, view) {
  const recv = performance.now();
  let samples;
  try {
    samples = decodePacket(view);
  } catch (err) {
    glove.live.errors++;
    if (glove.live.errors === 1) toast(`${glove.name}: ${err.message}. Check GLOVE_CONFIG.`);
    return;
  }
  const live = glove.live;
  const periodUs = 1e6 / GLOVE_CONFIG.sampleRateHz;
  for (const s of samples) {
    if (live.lastDevUs !== null) {
      const gap = (s.t_us - live.lastDevUs + 2 ** 32) % 2 ** 32;
      if (gap > 1.5 * periodUs && gap < 2 ** 31) live.drops += Math.round(gap / periodUs) - 1;
    }
    live.lastDevUs = s.t_us;
    live.count++;
    live.recvTimes.push(recv);
    if (capture.recording) {
      const rec = glove.rec;
      rec.t_device_us.push(s.t_us);
      rec.t_recv_ms.push(r3(recv));
      for (const f of LAYOUT.fields) {
        if (f.name === 't_us') continue;
        const v = s[f.name];
        rec[f.name].push(f.type === 'f32' || f.scale !== 1 ? (Array.isArray(v) ? v.map(r6) : r6(v)) : v);
      }
    }
  }
  live.last = samples[samples.length - 1];
  while (live.recvTimes.length && live.recvTimes[0] < recv - 1000) live.recvTimes.shift();
}

function handFromName(name) {
  if (/(-|_|\s)L$|left/i.test(name)) return 'Left';
  if (/(-|_|\s)R$|right/i.test(name)) return 'Right';
  return $('gloveHand').value;
}

async function connectBle() {
  if (!navigator.bluetooth) {
    toast('Web Bluetooth is not available here. Use Chrome or Edge on desktop or Android, over HTTPS or localhost.');
    return;
  }
  let device;
  try {
    device = await navigator.bluetooth.requestDevice({
      filters: [{ namePrefix: GLOVE_CONFIG.namePrefix }],
      optionalServices: [GLOVE_CONFIG.serviceUuid],
    });
  } catch (err) {
    if (err.name !== 'NotFoundError') toast(`Bluetooth: ${err.message}`); // NotFoundError = chooser cancelled
    return;
  }
  const name = device.name || 'glove';
  const glove = addGlove({ name, hand: handFromName(name), source: 'ble', device });
  const handler = (e) => onGlovePacket(glove, e.target.value);
  const connect = async () => {
    const server = await device.gatt.connect();
    const service = await server.getPrimaryService(GLOVE_CONFIG.serviceUuid);
    const ch = await service.getCharacteristic(GLOVE_CONFIG.sampleCharUuid);
    ch.addEventListener('characteristicvaluechanged', handler);
    await ch.startNotifications();
    glove.connected = true;
    glove.live.lastDevUs = null;
    renderGloves();
  };
  glove.reconnect = async () => {
    try {
      await connect();
    } catch (err) {
      toast(`${name}: ${err.message}`);
    }
  };
  device.addEventListener('gattserverdisconnected', () => {
    glove.connected = false;
    renderGloves();
    toast(`${name} disconnected${capture.recording ? ' (recording continues; reconnect it)' : ''}.`);
  });
  try {
    await connect();
  } catch (err) {
    glove.connected = false;
    renderGloves();
    toast(`${name}: ${err.message}`);
  }
}

function addMockGlove(hand = $('gloveHand').value) {
  if (gloves.some((g) => g.source === 'mock' && g.hand === hand)) {
    toast(`A mock ${hand.toLowerCase()} glove is already running.`);
    return;
  }
  const mock = new MockGlove(sim, { hand, encode: (batch) => encodePacket(batch), adcMax: GLOVE_CONFIG.adcMax });
  const glove = addGlove({ name: `SignGlove-${hand[0]} (mock)`, hand, source: 'mock', mock });
  mock.start((view) => onGlovePacket(glove, view));
  if (simCamera) simCamera.hands = handsForSim();
}

function removeGlove(glove) {
  glove.mock?.stop();
  try {
    glove.device?.gatt?.disconnect();
  } catch {
    /* already gone */
  }
  gloves.splice(gloves.indexOf(glove), 1);
  if (simCamera) simCamera.hands = handsForSim();
  renderGloves();
}

function renderGloves() {
  const list = $('gloveList');
  list.replaceChildren();
  if (!gloves.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No glove connected.';
    list.append(li);
  }
  for (const g of gloves) {
    const li = document.createElement('li');
    li.className = 'glove';
    li.innerHTML = `
      <div class="glove-head">
        <strong></strong><span class="tag"></span>
        <button class="btn small" type="button" data-act="reconnect" hidden>Reconnect</button>
        <button class="btn small" type="button" data-act="remove">Remove</button>
      </div>
      <div class="flexbars" title="Flex ADC, thumb to pinky">${'<div><i></i></div>'.repeat(5)}</div>
      <div class="glove-stat" data-k="rate">waiting for data</div>
      <div class="glove-stat" data-k="imu"></div>`;
    li.querySelector('strong').textContent = g.name;
    li.querySelector('.tag').textContent = `${g.hand} · ${g.source === 'mock' ? 'mock' : g.connected ? 'BLE' : 'disconnected'}`;
    const rc = li.querySelector('[data-act="reconnect"]');
    rc.hidden = g.source !== 'ble' || g.connected;
    rc.addEventListener('click', () => g.reconnect?.());
    li.querySelector('[data-act="remove"]').addEventListener('click', () => removeGlove(g));
    g.el = li;
    list.append(li);
  }
  updateStatus();
}

function updateGloveLive() {
  for (const g of gloves) {
    if (!g.el) continue;
    const { last, recvTimes, drops, count } = g.live;
    g.el.querySelector('[data-k="rate"]').textContent = `${recvTimes.length} Hz · ${count} samples · ${drops} dropped`;
    if (!last) continue;
    g.el.querySelectorAll('.flexbars i').forEach((bar, i) => {
      bar.style.width = `${Math.max(0, Math.min(100, (last.flex[i] / GLOVE_CONFIG.adcMax) * 100))}%`;
    });
    const w = Math.hypot(...last.gyro);
    const a = Math.hypot(...last.accel);
    g.el.querySelector('[data-k="imu"]').textContent = `|accel| ${a.toFixed(2)} g · |gyro| ${w.toFixed(2)} rad/s · q ${last.quat.map((x) => x.toFixed(2)).join(' ')}`;
  }
}

// --- Camera / hand tracking ---------------------------------------------------

/** HandTracker that keeps worldLandmarks (the app's detect() drops them). */
class CaptureTracker extends HandTracker {
  detectFull(video, nowMs) {
    if (!this.landmarker || video.readyState < 2 || video.videoWidth === 0) return null;
    if (video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = video.currentTime;
    try {
      return this.landmarker.detectForVideo(video, nowMs);
    } catch {
      this._fallbackToCpu();
      return null;
    }
  }
}

const sim = new SimHand();
let simCamera = null;
let tracker = null;
let camera = null;
let videoSource = null; // 'webcam' | 'synthetic'
let videoGen = 0; // bumped on every start/stop so stale frame loops exit
let videoSize = { width: 0, height: 0 };
let lastHands = [];
let fps = { times: [] };

function handsForSim() {
  const hands = [...new Set(gloves.filter((g) => g.source === 'mock').map((g) => g.hand))];
  return hands.length ? hands : ['Right'];
}

async function startVideo() {
  const source = $('videoSource').value;
  stopVideo();
  $('startVideoBtn').disabled = true;
  try {
    if (source === 'synthetic') {
      simCamera = new SimCamera(sim, { hands: handsForSim() });
      videoSize = { width: simCamera.width, height: simCamera.height };
      videoSource = 'synthetic';
      setPill('pillVideo', 'ok', 'Video: synthetic hand');
      nextSynthetic = 0;
      syntheticLoop(videoGen);
    } else {
      setPill('pillVideo', 'warn', 'Video: loading hand model');
      tracker ||= new CaptureTracker();
      if (!tracker.ready) await tracker.load();
      camera ||= new Camera($('video'));
      await camera.start('user');
      const video = $('video');
      videoSize = { width: video.videoWidth, height: video.videoHeight };
      videoSource = 'webcam';
      setPill('pillVideo', 'ok', `Video: webcam ${videoSize.width}×${videoSize.height} (${tracker.delegate})`);
      scheduleWebcamFrame(videoGen);
    }
    $('stage').classList.toggle('synthetic', source === 'synthetic');
    $('stageCard').hidden = true;
  } catch (err) {
    const msg = err?.name ? describeCameraError(err) : null;
    setPill('pillVideo', 'err', 'Video: failed');
    toast(msg ? `${msg.title}. ${msg.message}` : `Could not start video: ${err?.message || err}`);
    videoSource = null;
  } finally {
    $('startVideoBtn').disabled = false;
    updateStatus();
  }
}

function stopVideo() {
  videoGen++;
  camera?.stop();
  simCamera = null;
  videoSource = null;
}

function scheduleWebcamFrame(gen) {
  const video = $('video');
  if (gen !== videoGen) return;
  if (video.requestVideoFrameCallback) {
    video.requestVideoFrameCallback((_now, meta) => {
      if (gen !== videoGen) return;
      onWebcamFrame(meta?.captureTime ?? null);
      scheduleWebcamFrame(gen);
    });
  } else {
    requestAnimationFrame(() => {
      if (gen !== videoGen) return;
      onWebcamFrame(null);
      scheduleWebcamFrame(gen);
    });
  }
}

function onWebcamFrame(captureTime) {
  const t = performance.now();
  const result = tracker.detectFull($('video'), t);
  if (result) onDetection(result, t, captureTime);
}

// Synthetic frames come at a nominal 30 fps. A hidden tab throttles timers and
// animation frames, so catch up on every frame that fell due in the meantime.
let nextSynthetic = 0;
function syntheticLoop(gen) {
  if (gen !== videoGen || !simCamera) return;
  const now = performance.now();
  if (nextSynthetic < now - 2000) nextSynthetic = now;
  while (nextSynthetic <= now) {
    onDetection(simCamera.detect(nextSynthetic), nextSynthetic, null);
    nextSynthetic += 1000 / 30;
  }
  setTimeout(() => syntheticLoop(gen), 10);
}

function onDetection(result, t, captureTime) {
  const handed = result.handedness || result.handednesses || [];
  const hands = (result.landmarks || []).map((lm, i) => ({
    label: handed[i]?.[0]?.categoryName ?? null,
    score: r3(handed[i]?.[0]?.score ?? 0),
    world: (result.worldLandmarks?.[i] || []).map((p) => [r4(p.x), r4(p.y), r4(p.z)]),
    image: lm.map((p) => [r4(p.x), r4(p.y), r4(p.z)]),
  }));
  lastHands = hands;
  fps.times.push(t);
  while (fps.times.length && fps.times[0] < t - 1000) fps.times.shift();
  if (capture.recording) {
    capture.frames.push({ t: r3(t), capture_t: captureTime === null ? null : r3(captureTime), hands });
  }
  draw(hands);
}

const BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

function draw(hands) {
  const canvas = $('overlay');
  if (canvas.width !== videoSize.width || canvas.height !== videoSize.height) {
    canvas.width = videoSize.width;
    canvas.height = videoSize.height;
  }
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (videoSource === 'synthetic') {
    ctx.fillStyle = '#10172b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }
  const W = canvas.width, H = canvas.height;
  for (const h of hands) {
    ctx.strokeStyle = '#5eead4';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (const [a, b] of BONES) {
      ctx.moveTo(h.image[a][0] * W, h.image[a][1] * H);
      ctx.lineTo(h.image[b][0] * W, h.image[b][1] * H);
    }
    ctx.stroke();
    ctx.fillStyle = '#f472b6';
    for (const p of h.image) {
      ctx.beginPath();
      ctx.arc(p[0] * W, p[1] * H, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

// --- Calibration steps (docs/research/glove-dataset-alignment.md §4) ----------

const FREE_TIPS = [
  'Fingerspell slowly: A, B, C, ... facing the camera.',
  'Finger waves: curl and uncurl each finger in turn.',
  'Turn the palm: toward the camera, down, up, sideways. Keep the fingers moving.',
  'Pinch thumb to each fingertip, then open wide.',
  'Sign anything you like, slowly, at different palm angles.',
];

const STEPS = [
  {
    id: 'still', group: 0, title: 'Hold still', seconds: 3,
    prompt: 'Rest the gloved hand flat on the table, palm down, and keep it completely still.',
    solves: 'gyro bias',
  },
  {
    id: 'flex_fist', group: 1, title: 'Flat ↔ fist ×5', seconds: 10,
    prompt: 'Palm toward the camera. Open the hand flat, then squeeze a tight fist, following the cue.',
    cue: (t) => (Math.floor(t) % 2 ? 'FIST' : 'OPEN'),
    solves: 'per-finger flex min/max',
  },
  {
    id: 'pose_palm_down', group: 2, title: 'Pose: palm down', seconds: 6,
    prompt: 'Hand flat in the air, palm to the floor, fingers pointing at the camera. Hold it.',
    solves: 'R_mount (static)',
  },
  {
    id: 'pose_fingers_down', group: 2, title: 'Pose: fingers down', seconds: 6,
    prompt: 'Let the arm hang. Fingers point at the floor, palm toward your leg. Hold it.',
    solves: 'R_mount (static)',
  },
  {
    id: 'pose_thumb_up', group: 2, title: 'Pose: thumb up', seconds: 6,
    prompt: 'Fist with the thumb pointing at the ceiling, palm facing in (toward your other hand). Hold it.',
    solves: 'R_mount (static)',
  },
  {
    id: 'twists', group: 3, title: 'Three sharp twists', seconds: 5,
    prompt: 'Palm toward the camera. On each TWIST cue, turn the wrist sharply (like a doorknob) and back.',
    cue: (t) => (TWIST_TIMES_S.some((c) => t >= c - 0.15 && t < c + 0.35) ? 'TWIST!' : ''),
    beeps: TWIST_TIMES_S,
    solves: 'time lag (|gyro| vs keypoint |ω|)',
  },
  {
    id: 'free', group: 4, title: 'Free motion', seconds: null, // from the duration field
    prompt: 'Move slowly, facing the camera, at several palm orientations.',
    cue: (t) => FREE_TIPS[Math.floor(t / 15) % FREE_TIPS.length],
    solves: 'quadratic flex fit, R_mount refinement, validation',
  },
];

const runner = { busy: false, skip: false, cancel: false };

function freeSeconds() {
  const v = Number($('freeSeconds').value);
  return Math.max(30, Math.min(180, Number.isFinite(v) ? v : 90));
}

/** Wait for `ms`, calling onTick(elapsedSeconds) every animation frame. Returns false if skipped/cancelled. */
function wait(ms, onTick) {
  return new Promise((resolve) => {
    const t0 = performance.now();
    const loop = () => {
      const el = performance.now() - t0;
      if (runner.skip || runner.cancel) return resolve(false);
      onTick?.(el / 1000);
      if (el >= ms) return resolve(true);
      requestAnimationFrame(loop);
    };
    loop();
  });
}

function showPrompt({ kicker = '', title = '', text = '', count = '', cue = '', live = false }) {
  $('prompt').hidden = false;
  $('prompt').classList.toggle('live', live);
  $('promptKicker').textContent = kicker;
  $('promptTitle').textContent = title;
  $('promptText').textContent = text;
  $('promptCount').textContent = count;
  $('promptCue').textContent = cue;
}

function hidePrompt() {
  $('prompt').hidden = true;
}

/** Get-ready countdown, then the recorded segment. Returns the segment (or null if cancelled). */
async function runSegment({ step, title, prompt, seconds, cue, beeps = [], sign = null, rep = null, readySeconds = 3, kicker }) {
  sim.setStep('ready', performance.now(), { next: step, sign });
  showPrompt({ kicker: 'Get ready', title, text: prompt, count: readySeconds });
  const ready = await wait(readySeconds * 1000, (t) => {
    $('promptCount').textContent = Math.ceil(readySeconds - t) || '';
  });
  if (runner.cancel) return null;
  if (!ready) runner.skip = false; // skipping the countdown starts the segment now

  const t_start = performance.now();
  sim.setStep(step, t_start, { sign });
  beep(880, 120);
  let beeped = 0;
  const done = await wait(seconds * 1000, (t) => {
    showPrompt({ kicker, title, text: prompt, count: Math.ceil(seconds - t), cue: cue ? cue(t) : '', live: true });
    if (beeped < beeps.length && t >= beeps[beeped]) {
      beep(1320, 60);
      beeped++;
    }
  });
  const t_end = performance.now();
  sim.setStep('idle', t_end);
  beep(440, 120);
  runner.skip = false;
  const seg = {
    step, title, t_start: r3(t_start), t_end: r3(t_end),
    planned_s: seconds, completed: done, sign, rep,
  };
  if (capture.recording) capture.segments.push(seg);
  renderSegments();
  return runner.cancel ? null : seg;
}

async function runSteps(list) {
  if (runner.busy) return;
  if (!capture.recording && !startRecording()) return;
  runner.busy = true;
  runner.cancel = false;
  runner.skip = false;
  updateStatus();
  try {
    for (const s of list) {
      const seconds = s.seconds ?? freeSeconds();
      const seg = await runSegment({
        step: s.id, title: s.title, prompt: s.prompt, seconds, cue: s.cue, beeps: s.beeps,
        kicker: `Step ${s.group} · ${s.solves}`,
      });
      if (!seg) break;
    }
  } finally {
    finishRun();
  }
}

async function recordSign() {
  const label = $('signLabel').value.trim();
  if (!label) {
    toast('Type the sign label first.');
    $('signLabel').focus();
    return;
  }
  if (runner.busy) return;
  if (!capture.recording && !startRecording()) return;
  const reps = Math.max(1, Math.min(10, Number($('signReps').value) || 5));
  const seconds = Math.max(1, Math.min(10, Number($('signSeconds').value) || 3));
  runner.busy = true;
  runner.cancel = false;
  runner.skip = false;
  updateStatus();
  try {
    const already = capture.segments.filter((s) => s.step === 'sign' && s.sign === label).length;
    for (let i = 0; i < reps; i++) {
      const seg = await runSegment({
        step: 'sign', title: `${label} (${i + 1}/${reps})`, sign: label, rep: already + i + 1,
        prompt: 'Start from rest, sign once at a natural pace, return to rest.',
        seconds, readySeconds: i === 0 ? 3 : 1.5, kicker: 'Step 5 · sign repetitions',
      });
      if (!seg) break;
    }
  } finally {
    finishRun();
  }
}

function finishRun() {
  runner.busy = false;
  runner.skip = false;
  runner.cancel = false;
  hidePrompt();
  updateStatus();
}

function renderSteps() {
  const list = $('stepList');
  list.replaceChildren();
  for (const s of STEPS) {
    const li = document.createElement('li');
    const secs = s.seconds ?? `${freeSeconds()}`;
    li.innerHTML = `<span class="step-n"></span><span class="step-body"><strong></strong><small></small></span><span class="step-done"></span><button class="btn small" type="button">Run</button>`;
    li.querySelector('.step-n').textContent = s.group;
    li.querySelector('strong').textContent = s.title;
    li.querySelector('small').textContent = `${secs} s · ${s.solves}`;
    li.querySelector('button').addEventListener('click', () => runSteps([s]));
    li.dataset.step = s.id;
    list.append(li);
  }
  renderSegments();
}

function renderSegments() {
  for (const li of $('stepList').children) {
    const n = capture.segments.filter((s) => s.step === li.dataset.step).length;
    li.querySelector('.step-done').textContent = n ? `✓${n > 1 ? ` ×${n}` : ''}` : '';
  }
  const counts = new Map();
  for (const s of capture.segments) if (s.step === 'sign') counts.set(s.sign, (counts.get(s.sign) || 0) + 1);
  const ul = $('signDone');
  ul.replaceChildren();
  for (const [label, n] of counts) {
    const li = document.createElement('li');
    li.textContent = `${label} · ${n} rep${n === 1 ? '' : 's'}`;
    ul.append(li);
  }
}

// --- Recording and export ------------------------------------------------------

function startRecording() {
  if (capture.frames.length && !capture.exported && !confirm('Discard the current recording? It has not been exported.')) return false;
  capture.recording = true;
  capture.exported = false;
  capture.startedAt = performance.now();
  capture.stoppedAt = null;
  capture.frames = [];
  capture.segments = [];
  for (const g of gloves) g.rec = emptyRec();
  renderSegments();
  updateStatus();
  return true;
}

function stopRecording() {
  runner.cancel = true;
  capture.recording = false;
  capture.stoppedAt = performance.now();
  updateStatus();
}

/** The export object. Schema documented in research/load_capture.py. */
export function buildCapture() {
  const mockGloves = gloves.filter((g) => g.mock);
  return {
    schema: SCHEMA,
    version: SCHEMA_VERSION,
    created_at: new Date().toISOString(),
    clock: {
      source: 'performance.now()', unit: 'ms',
      time_origin_unix_ms: performance.timeOrigin,
      recording_start: capture.startedAt === null ? null : r3(capture.startedAt),
      recording_stop: capture.stoppedAt === null ? null : r3(capture.stoppedAt),
    },
    wearer: { id: $('wearerId').value.trim() || null, dominant_hand: $('dominantHand').value },
    video: {
      source: videoSource,
      width: videoSize.width,
      height: videoSize.height,
      // Frames given to MediaPipe are never flipped (the preview may be, the pixels are not).
      mirrored_input: false,
      // MediaPipe labels handedness as if the image were a mirrored selfie, so on these
      // un-mirrored frames the label is the OPPOSITE of the physical hand.
      handedness_label: 'mediapipe (assumes mirrored input)',
      mediapipe: { package: '@mediapipe/tasks-vision', version: MP_VERSION, task: 'HandLandmarker', running_mode: 'VIDEO', num_hands: 2, delegate: tracker?.delegate ?? null },
      world_units: 'm, origin at the hand centre, camera axes (+x right, +y down, +z away)',
      image_units: 'normalized [0, 1] x, y; z relative to the wrist, same scale as x',
    },
    glove_config: GLOVE_CONFIG,
    user_agent: navigator.userAgent,
    frames: capture.frames,
    gloves: gloves.map((g) => ({ id: g.id, name: g.name, hand: g.hand, source: g.source, samples: g.rec })),
    segments: capture.segments,
    mock: mockGloves.length || videoSource === 'synthetic'
      ? {
        camera_latency_ms: videoSource === 'synthetic' ? simCamera?.latencyMs ?? null : null,
        gloves: mockGloves.map((g) => ({ id: g.id, ...g.mock.truth() })),
      }
      : null,
  };
}

function exportCapture() {
  const data = buildCapture();
  const nSamples = data.gloves.reduce((n, g) => n + g.samples.t_device_us.length, 0);
  if (!data.frames.length && !nSamples) {
    toast('Nothing recorded yet.');
    return;
  }
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const stamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
  const who = (data.wearer.id || 'wearer').replace(/[^\w-]+/g, '_');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `signspeak-calib-${who}-${stamp}.json`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
  capture.exported = true;
  toast(`Saved ${a.download} (${(blob.size / 1e6).toFixed(1)} MB).`);
}

// --- Status --------------------------------------------------------------------

function setPill(id, state, text) {
  const el = $(id);
  el.dataset.state = state;
  el.querySelector('span').textContent = text;
}

function updateStatus() {
  const rec = capture.recording;
  const live = gloves.filter((g) => g.connected).length;
  setPill('pillGlove', live ? 'ok' : 'idle', `Gloves: ${live}`);
  setPill('pillRec', rec ? 'err' : capture.frames.length ? 'ok' : 'idle',
    rec ? 'Recording' : capture.frames.length ? 'Recorded' : 'Not recording');
  $('recBtn').textContent = rec ? 'Stop recording' : capture.frames.length ? 'New recording' : 'Start recording';
  $('recBtn').classList.toggle('danger', rec);
  $('runAllBtn').disabled = runner.busy;
  $('signBtn').disabled = runner.busy;
  $('skipBtn').disabled = !runner.busy;
  $('cancelBtn').disabled = !runner.busy;
  for (const b of $('stepList').querySelectorAll('button')) b.disabled = runner.busy;
  $('exportBtn').disabled = !capture.frames.length && !gloves.some((g) => g.rec.t_device_us.length);
}

function updateCounters() {
  const secs = capture.startedAt === null ? 0 : ((capture.stoppedAt ?? performance.now()) - capture.startedAt) / 1000;
  const glove = gloves.map((g) => `${g.hand[0]} ${g.rec.t_device_us.length}`).join(', ') || '0';
  $('counters').textContent = `${secs.toFixed(0)} s · ${capture.frames.length} frames · glove samples ${glove} · ${capture.segments.length} segments`;
  const handText = lastHands.length ? lastHands.map((h) => `${h.label} ${(h.score * 100).toFixed(0)}%`).join(', ') : 'no hands';
  $('handBadge').textContent = `${fps.times.length} fps · ${handText}`;
}

let toastTimer = null;
function toast(msg) {
  const el = $('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 5000);
}

// --- Wire up -------------------------------------------------------------------

function init() {
  $('startVideoBtn').addEventListener('click', startVideo);
  $('cardBtn').addEventListener('click', startVideo);
  $('bleBtn').addEventListener('click', connectBle);
  $('mockBtn').addEventListener('click', () => addMockGlove());
  $('recBtn').addEventListener('click', () => (capture.recording ? stopRecording() : startRecording()));
  $('runAllBtn').addEventListener('click', () => runSteps(STEPS));
  $('signBtn').addEventListener('click', recordSign);
  $('skipBtn').addEventListener('click', () => (runner.skip = true));
  $('cancelBtn').addEventListener('click', () => (runner.cancel = true));
  $('exportBtn').addEventListener('click', exportCapture);
  $('freeSeconds').addEventListener('change', renderSteps);
  $('bleBtn').disabled = !navigator.bluetooth;
  if (!navigator.bluetooth) $('bleHint').textContent = 'Web Bluetooth is unavailable in this browser; use Chrome or Edge (desktop or Android). The mock glove works anywhere.';

  if (params.has('mock')) {
    $('videoSource').value = 'synthetic';
    addMockGlove('Right');
  }
  renderSteps();
  renderGloves();
  setInterval(() => {
    updateGloveLive();
    updateCounters();
  }, 200);
  window.addEventListener('beforeunload', (e) => {
    if (capture.frames.length && !capture.exported) e.preventDefault();
  });
  // A debugging hook: window.calib.buildCapture() returns what Export would save.
  window.calib = { buildCapture, decodePacket, encodePacket, GLOVE_CONFIG, capture, gloves, sim };
}

init();
