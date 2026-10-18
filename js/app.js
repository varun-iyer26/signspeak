// SignSpeak: bootstraps the app, wires the UI, and runs the per-frame recognition loop.
import { Camera, describeCameraError } from './camera.js';
import { HandTracker } from './hands.js';
import { buildFeatureVector } from './features.js';
import { classifyLetter } from './fingerspell.js';
import { KnnClassifier } from './knn.js';
import { Speech, DEFAULT_VOICE_ID, DEFAULT_MODEL_ID, DEFAULT_TTS_ENDPOINT, TTS_MODELS } from './speech.js';
import { FrameBuffer, interpretFrames, checkBackend, DEFAULT_ENDPOINT } from './ai.js';

// ---- tuning constants ------------------------------------------------------

const FS_HOLD_FRAMES = 12; // frames a letter must hold to be committed
const FS_MIN_CONF = 0.6;
const CUSTOM_HOLD_FRAMES = 10;
const CUSTOM_MIN_CONF = 0.35;
const SPACE_AFTER_MS = 1200; // no hand: insert a space
const SPEAK_AFTER_MS = 3000; // no hand: speak the pending text
const CUSTOM_PAUSE_MS = 1500; // no hand: speak pending custom words
const AI_PAUSE_MS = 1000; // no hand: send the buffered frames
const AI_MIN_FRAMES = 3;
const DROPOUT_MS = 400; // a hand missing this long counts as "hand changed" for the cooldown
const RECORD_TARGET = 30;
const RECORD_INTERVAL_MS = 66;
const RECORD_MAX_MS = 2500;
const MIN_SAMPLES = 5;

const $ = (id) => document.getElementById(id);

// ---- settings (localStorage, every access wrapped) ---------------------------

const SETTINGS_KEY = 'signspeak.settings.v1';
const DEFAULT_SETTINGS = {
  endpoint: DEFAULT_ENDPOINT,
  apiKey: '',
  model: '',
  engine: 'elevenlabs',
  voiceId: DEFAULT_VOICE_ID,
  ttsModel: DEFAULT_MODEL_ID,
  elevenKey: '',
  ttsEndpoint: DEFAULT_TTS_ENDPOINT,
  voiceURI: '',
  rate: 1,
  muted: false,
};

function loadSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
    const s = { ...DEFAULT_SETTINGS };
    if (typeof stored.endpoint === 'string') s.endpoint = stored.endpoint;
    if (typeof stored.apiKey === 'string') s.apiKey = stored.apiKey;
    if (typeof stored.model === 'string') s.model = stored.model;
    if (stored.engine === 'elevenlabs' || stored.engine === 'browser') s.engine = stored.engine;
    if (typeof stored.voiceId === 'string' && stored.voiceId) s.voiceId = stored.voiceId;
    if (TTS_MODELS.includes(stored.ttsModel)) s.ttsModel = stored.ttsModel;
    if (typeof stored.elevenKey === 'string') s.elevenKey = stored.elevenKey;
    if (typeof stored.ttsEndpoint === 'string' && stored.ttsEndpoint) s.ttsEndpoint = stored.ttsEndpoint;
    if (typeof stored.voiceURI === 'string') s.voiceURI = stored.voiceURI;
    if (Number.isFinite(stored.rate)) s.rate = Math.min(2, Math.max(0.5, stored.rate));
    s.muted = stored.muted === true;
    return s;
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    /* storage blocked: settings last for this session only */
  }
}

// ---- core objects ------------------------------------------------------------

const settings = loadSettings();
const video = $('video');
const overlay = $('overlay');
const stage = $('stage');
const camera = new Camera(video);
const tracker = new HandTracker();
const speech = new Speech();
const knn = new KnnClassifier({ k: 5 });
const frames = new FrameBuffer({ max: 8, width: 384, intervalMs: 333 });

/** Copy the voice settings onto the speech object and forget any earlier ElevenLabs failure. */
function applySpeechSettings() {
  speech.engine = settings.engine;
  speech.voiceId = settings.voiceId;
  speech.modelId = settings.ttsModel;
  speech.apiKey = settings.elevenKey;
  speech.endpoint = settings.ttsEndpoint;
  speech.reset();
}

applySpeechSettings();
speech.voiceURI = settings.voiceURI;
speech.rate = settings.rate;
speech.muted = settings.muted;
speech.onFallback = (message) => toast(message);
speech.onStatusChange = () => renderVoicePill();
knn.load();

const state = {
  mode: 'fingerspell',
  cam: 'idle', // idle | starting | on | error
  camError: null,
  model: 'loading', // loading | ready | error
  lastHandAt: performance.now(),
  handsNow: false,
  rec: null, // active recording, or null
  aiBusy: false,
  aiSent: false, // frames already sent for the current pause
  spaceDone: false,
  speakDone: false,
};

// ---- small UI helpers --------------------------------------------------------

let toastTimer = 0;
function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2800);
}

function setPill(id, st, text) {
  const pill = $(id);
  pill.dataset.state = st;
  pill.lastElementChild.textContent = text;
}

let lastGuess = '';
function setGuess(label, confidence = 0) {
  const pct = Math.round(Math.max(0, Math.min(1, confidence)) * 100);
  const key = `${label}|${pct}`;
  if (key === lastGuess) return;
  lastGuess = key;
  $('guessLabel').textContent = label || '-';
  $('guessPct').textContent = label && pct ? `${pct}%` : '';
  $('guessBar').style.width = `${label ? pct : 0}%`;
  $('guessMeter').setAttribute('aria-valuenow', String(label ? pct : 0));
}

// ---- transcript ----------------------------------------------------------------

class Transcript {
  constructor(el) {
    this.el = el;
    this.text = '';
    this.spoken = 0; // characters already spoken
  }
  render() {
    this.el.textContent = this.text;
    this.el.scrollTop = this.el.scrollHeight;
  }
  appendLetter(ch) {
    this.text += ch;
    this.render();
  }
  appendWord(word) {
    if (this.text && !/\s$/.test(this.text)) this.text += ' ';
    this.text += word;
    this.render();
  }
  addSpace() {
    if (!this.text || /\s$/.test(this.text)) return false;
    this.text += ' ';
    this.render();
    return true;
  }
  backspace() {
    this.text = this.text.slice(0, -1);
    this.spoken = Math.min(this.spoken, this.text.length);
    this.render();
  }
  clear() {
    this.text = '';
    this.spoken = 0;
    this.render();
  }
  pending() {
    return this.text.slice(Math.min(this.spoken, this.text.length)).trim();
  }
  markSpoken() {
    this.spoken = this.text.length;
  }
  context() {
    return this.text.slice(-200);
  }
}
const transcript = new Transcript($('transcript'));

/** Speak the not-yet-spoken part of the transcript. */
function speakPending() {
  const text = transcript.pending();
  if (!text) return;
  speech.speak(text);
  transcript.markSpoken();
}

// ---- stabiliser: "same prediction for N frames, then not again until it changes" ------

class Stabilizer {
  constructor(hold) {
    this.hold = hold;
    this.release();
  }
  /** Feed one frame's prediction (or null). Returns the value when it is newly committed. */
  push(value) {
    if (value !== null && value === this.cand) this.count++;
    else {
      this.cand = value;
      this.count = value !== null ? 1 : 0;
    }
    // Cooldown: the committed value may repeat only after the prediction changed for a few frames.
    if (value !== null && value === this.last) this.breaks = 0;
    else if (this.last !== null && ++this.breaks >= 3) this.last = null;

    if (value !== null && value !== this.last && this.count >= this.hold) {
      this.last = value;
      this.count = 0;
      this.breaks = 0;
      return value;
    }
    return null;
  }
  /** The hand dropped out: forget everything so the same value can be committed again. */
  release() {
    this.cand = null;
    this.count = 0;
    this.last = null;
    this.breaks = 0;
  }
}
const fsStable = new Stabilizer(FS_HOLD_FRAMES);
const customStable = new Stabilizer(CUSTOM_HOLD_FRAMES);

function resetRecognition() {
  fsStable.release();
  customStable.release();
  frames.clear();
  state.aiSent = false;
  state.spaceDone = false;
  state.speakDone = false;
  state.lastHandAt = performance.now();
  setGuess('');
  renderFrameCount();
}

// ---- status pills and stage card ----------------------------------------------------

function renderModelPill() {
  if (state.model === 'ready') setPill('pillModel', 'ok', `Hand model: ready (${tracker.delegate})`);
  else if (state.model === 'error') setPill('pillModel', 'err', 'Hand model: failed to load');
  else setPill('pillModel', 'warn', 'Hand model: loading');
}

function renderCameraPill() {
  if (state.cam === 'on') setPill('pillCamera', 'ok', `Camera: ${camera.facing === 'user' ? 'front' : 'rear'}`);
  else if (state.cam === 'starting') setPill('pillCamera', 'warn', 'Camera: starting');
  else if (state.cam === 'error') setPill('pillCamera', 'err', 'Camera: unavailable');
  else setPill('pillCamera', 'idle', 'Camera: off');
}

function renderVoicePill() {
  const voice = speech.activeVoice;
  if (voice === 'elevenlabs') setPill('pillVoice', 'ok', 'Voice: ElevenLabs');
  else if (voice === 'fallback') setPill('pillVoice', 'warn', 'Voice: browser fallback');
  else setPill('pillVoice', 'idle', 'Voice: browser');
}

let backendTimer = 0;
function refreshBackend() {
  clearTimeout(backendTimer);
  backendTimer = setTimeout(async () => {
    if (settings.apiKey) {
      setPill('pillBackend', 'ok', 'AI: direct mode (your key)');
      return;
    }
    setPill('pillBackend', 'warn', 'AI backend: checking');
    const endpointAtStart = settings.endpoint;
    const ok = await checkBackend(endpointAtStart);
    if (endpointAtStart !== settings.endpoint || settings.apiKey) return; // settings changed meanwhile
    setPill('pillBackend', ok ? 'ok' : 'idle', ok ? 'AI backend: ready' : 'AI backend: not available');
  }, 400);
}

let cardAction = null;
/** Show the stage card (start / error / retry) or hide it, depending on the app state. */
function renderCard() {
  const card = $('stageCard');
  const btn = $('cardBtn');
  const show = (title, text, label, action) => {
    $('cardTitle').textContent = title;
    $('cardText').textContent = text;
    btn.textContent = label;
    btn.hidden = !label;
    btn.disabled = false;
    cardAction = action;
    card.hidden = false;
  };

  if (state.cam === 'starting') {
    show('Starting camera', 'Waiting for the camera...', '', null);
  } else if (state.cam === 'error') {
    show(state.camError.title, state.camError.message, 'Try again', () => startCamera(camera.facing));
  } else if (state.cam === 'idle') {
    show('Point the camera at someone signing', 'Everything runs in your browser. The rear camera is used by default on phones.', 'Start camera', () => startCamera('environment'));
  } else if (state.model === 'error') {
    show('Hand model did not load', 'Check your connection (the model comes from a CDN), then retry.', 'Retry', loadModel);
  } else {
    card.hidden = true;
  }
}

$('cardBtn').addEventListener('click', () => cardAction?.());

// ---- camera ----------------------------------------------------------------------------

async function startCamera(facing) {
  state.cam = 'starting';
  renderCard();
  renderCameraPill();
  speech.unlock(); // we are inside a user gesture, which iOS requires for speech
  try {
    await camera.start(facing);
    state.cam = 'on';
    stage.classList.toggle('mirrored', camera.mirrored);
    stage.style.aspectRatio = `${video.videoWidth || 4} / ${video.videoHeight || 3}`;
    $('flipBtn').hidden = false;
    resetRecognition();
  } catch (err) {
    if (err.message === 'Superseded') return;
    state.cam = 'error';
    state.camError = describeCameraError(err);
  }
  renderCameraPill();
  renderCard();
}

$('flipBtn').addEventListener('click', async () => {
  const facing = camera.facing === 'user' ? 'environment' : 'user';
  $('flipBtn').disabled = true;
  await startCamera(facing);
  $('flipBtn').disabled = false;
});

async function loadModel() {
  state.model = 'loading';
  renderModelPill();
  renderCard();
  try {
    await tracker.load();
    state.model = 'ready';
  } catch {
    state.model = 'error';
  }
  renderModelPill();
  renderCard();
}

// ---- mode switching ----------------------------------------------------------------------

function setMode(mode, focus = false) {
  if (state.rec) cancelRecording();
  state.mode = mode;
  for (const tab of document.querySelectorAll('.tab')) {
    const on = tab.dataset.mode === mode;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    $(`panel-${tab.dataset.mode}`).hidden = !on;
    if (on && focus) tab.focus();
  }
  resetRecognition();
}

const tabs = [...document.querySelectorAll('.tab')];
tabs.forEach((tab, i) => {
  tab.addEventListener('click', () => setMode(tab.dataset.mode));
  tab.addEventListener('keydown', (e) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[e.key];
    if (!step) return;
    e.preventDefault();
    setMode(tabs[(i + step + tabs.length) % tabs.length].dataset.mode, true);
  });
});

// ---- per-frame loop ------------------------------------------------------------------------

function loop(now) {
  requestAnimationFrame(loop);
  if (state.cam !== 'on' || !tracker.ready) return;
  const hands = tracker.detect(video, now);
  if (!hands) return; // no new video frame yet
  tracker.draw(overlay, video, hands);
  handleFrame(hands, now);
}

function handleFrame(hands, now) {
  const size = { width: video.videoWidth, height: video.videoHeight };
  state.handsNow = hands.length > 0;
  if (state.handsNow) {
    state.lastHandAt = now;
    state.spaceDone = false;
    state.speakDone = false;
  }
  const gap = now - state.lastHandAt; // ms since a hand was last seen
  $('handBadge').textContent = hands.length ? `${hands.length} hand${hands.length > 1 ? 's' : ''}` : 'No hands';

  if (state.mode === 'fingerspell') stepFingerspell(hands, gap, size);
  else if (state.mode === 'custom') stepCustom(hands, gap, size, now);
  else stepAi(hands, gap, now);
}

function stepFingerspell(hands, gap, size) {
  if (!hands.length) {
    setGuess('');
    if (gap > DROPOUT_MS) fsStable.release();
    if (gap > SPACE_AFTER_MS && !state.spaceDone) {
      state.spaceDone = true;
      transcript.addSpace();
    }
    if (gap > SPEAK_AFTER_MS && !state.speakDone) {
      state.speakDone = true;
      speakPending();
    }
    return;
  }
  let best = null;
  for (const hand of hands) {
    const r = classifyLetter(hand.landmarks, size);
    if (r && (!best || r.confidence > best.confidence)) best = r;
  }
  setGuess(best ? best.letter : '', best ? best.confidence : 0);
  const committed = fsStable.push(best && best.confidence >= FS_MIN_CONF ? best.letter : null);
  if (committed) transcript.appendLetter(committed.toLowerCase());
}

function stepCustom(hands, gap, size, now) {
  if (state.rec) {
    recordStep(hands, size, now);
    return;
  }
  if (!hands.length) {
    setGuess('');
    if (gap > DROPOUT_MS) customStable.release();
    if (gap > CUSTOM_PAUSE_MS && !state.speakDone) {
      state.speakDone = true;
      speakPending();
    }
    return;
  }
  const pred = knn.predict(buildFeatureVector(hands, camera.mirrored, size));
  setGuess(pred ? pred.label : '', pred ? pred.confidence : 0);
  const committed = customStable.push(pred && pred.confidence >= CUSTOM_MIN_CONF ? pred.label : null);
  if (committed) transcript.appendWord(committed);
}

function stepAi(hands, gap, now) {
  if (state.aiBusy) return;
  if (hands.length) {
    state.aiSent = false;
    if (frames.capture(video, now)) renderFrameCount();
    setGuess('');
  } else if (gap >= AI_PAUSE_MS && !state.aiSent) {
    if (frames.size >= AI_MIN_FRAMES) {
      state.aiSent = true;
      runInterpret();
    } else if (frames.size > 0 && gap > 4000) {
      frames.clear(); // a brief hand wave, not a sentence
      renderFrameCount();
    }
  }
}

// ---- custom signs: recording, list, import / export ------------------------------------------------

function renderFrameCount() {
  $('aiFrames').textContent = `${frames.size} frame${frames.size === 1 ? '' : 's'} buffered`;
}

function setRecordMsg(text) {
  $('recordMsg').textContent = text;
}

function startRecording() {
  if (state.rec || state.mode !== 'custom') return;
  const label = $('signLabel').value.trim();
  if (!label) {
    setRecordMsg('Type a label first.');
    $('signLabel').focus();
    return;
  }
  if (state.cam !== 'on' || !tracker.ready) {
    setRecordMsg('Start the camera and wait for the hand model to load.');
    return;
  }
  state.rec = { label, vectors: [], start: performance.now(), last: 0 };
  $('recordBtn').classList.add('recording');
  $('recText').textContent = 'Recording...';
  setRecordMsg(`Recording "${label}"... keep your hand in view.`);
}

function recordStep(hands, size, now) {
  const rec = state.rec;
  if (hands.length && now - rec.last >= RECORD_INTERVAL_MS) {
    const vec = buildFeatureVector(hands, camera.mirrored, size);
    if (vec) {
      rec.vectors.push(vec);
      rec.last = now;
    }
  }
  $('recFill').style.width = `${Math.min(100, (rec.vectors.length / RECORD_TARGET) * 100)}%`;
  if (rec.vectors.length >= RECORD_TARGET || now - rec.start >= RECORD_MAX_MS) stopRecording();
}

function endRecordingUi() {
  state.rec = null;
  $('recordBtn').classList.remove('recording');
  $('recText').textContent = 'Hold to record';
  $('recFill').style.width = '0';
  resetRecognition();
}

function stopRecording() {
  const rec = state.rec;
  if (!rec) return;
  endRecordingUi();
  if (rec.vectors.length < MIN_SAMPLES) {
    setRecordMsg('Not enough samples. Keep your hand in view and hold the button for about 2 seconds.');
    return;
  }
  knn.addMany(rec.label, rec.vectors);
  const saved = knn.save();
  renderSignList();
  setRecordMsg(`Saved ${rec.vectors.length} samples for "${rec.label}".${saved ? '' : ' Could not write to browser storage, so they last for this session only.'}`);
}

function cancelRecording() {
  if (state.rec) endRecordingUi();
}

(() => {
  const btn = $('recordBtn');
  btn.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    btn.setPointerCapture?.(e.pointerId);
    startRecording();
  });
  for (const type of ['pointerup', 'pointercancel']) btn.addEventListener(type, stopRecording);
  btn.addEventListener('contextmenu', (e) => e.preventDefault());
  btn.addEventListener('keydown', (e) => {
    if (e.key !== ' ' && e.key !== 'Enter') return;
    e.preventDefault();
    if (!e.repeat) startRecording();
  });
  btn.addEventListener('keyup', (e) => {
    if (e.key !== ' ' && e.key !== 'Enter') return;
    e.preventDefault();
    stopRecording();
  });
})();

function renderSignList() {
  const list = $('signList');
  list.textContent = '';
  const labels = knn.labels();
  if (!labels.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No custom signs yet.';
    list.append(li);
    return;
  }
  for (const { label, count } of labels) {
    const li = document.createElement('li');
    const name = document.createElement('button');
    name.type = 'button';
    name.className = 'name';
    name.textContent = label;
    name.title = 'Use this label to record more samples';
    name.addEventListener('click', () => {
      $('signLabel').value = label;
    });
    const n = document.createElement('span');
    n.className = 'count';
    n.textContent = `${count} samples`;
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'del';
    del.textContent = '×';
    del.setAttribute('aria-label', `Delete ${label}`);
    del.addEventListener('click', () => {
      if (!confirm(`Delete the sign "${label}" and its ${count} samples?`)) return;
      knn.deleteLabel(label);
      knn.save();
      renderSignList();
    });
    li.append(name, n, del);
    list.append(li);
  }
}

$('exportBtn').addEventListener('click', () => {
  if (!knn.size) {
    setRecordMsg('Nothing to export yet.');
    return;
  }
  const url = URL.createObjectURL(new Blob([knn.exportJSON()], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'signspeak-signs.json';
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$('importBtn').addEventListener('click', () => $('importFile').click());
$('importFile').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (!file) return;
  try {
    const { labels, samples } = knn.importJSON(await file.text());
    const saved = knn.save();
    renderSignList();
    setRecordMsg(`Imported ${samples} samples for ${labels} sign${labels === 1 ? '' : 's'}.${saved ? '' : ' Could not write to browser storage.'}`);
  } catch (err) {
    setRecordMsg(err.message);
  }
});

// ---- AI interpret ---------------------------------------------------------------------------------

function setAiStatus(text) {
  $('aiStatus').textContent = text;
}

/** Remove the "(unsure)" marker before speaking. */
const forSpeech = (text) => text.replace(/^\s*\(unsure\)\s*/i, '');

async function runInterpret() {
  if (state.aiBusy) return;
  if (state.handsNow) frames.capture(video, performance.now(), true);
  const batch = frames.take();
  renderFrameCount();
  if (!batch.length) {
    setAiStatus('No frames yet. Sign in front of the camera first.');
    return;
  }
  state.aiBusy = true;
  $('interpretBtn').disabled = true;
  setAiStatus(`Interpreting ${batch.length} frames...`);
  setGuess('...');
  try {
    const text = await interpretFrames({
      frames: batch,
      context: transcript.context(),
      endpoint: settings.endpoint,
      apiKey: settings.apiKey,
      model: settings.model,
    });
    if (!text) {
      setAiStatus('No signing detected.');
    } else {
      transcript.appendWord(text);
      transcript.markSpoken();
      speech.speak(forSpeech(text));
      setAiStatus('Done.');
    }
  } catch (err) {
    setAiStatus(err.message);
    toast(err.message);
  } finally {
    state.aiBusy = false;
    $('interpretBtn').disabled = false;
    setGuess('');
  }
}

$('interpretBtn').addEventListener('click', () => {
  speech.unlock();
  runInterpret();
});

// ---- transcript buttons -------------------------------------------------------------------------------

$('speakBtn').addEventListener('click', () => {
  const text = transcript.text.trim();
  if (!text) return toast('Nothing to speak yet.');
  if (speech.muted) return toast('Speech is muted.');
  speech.unlock();
  speech.speak(forSpeech(text), { interrupt: true });
  transcript.markSpoken();
});
$('spaceBtn').addEventListener('click', () => transcript.addSpace());
$('backBtn').addEventListener('click', () => transcript.backspace());
$('clearBtn').addEventListener('click', () => {
  transcript.clear();
  speech.cancel();
});
$('copyBtn').addEventListener('click', async () => {
  const text = transcript.text.trim();
  if (!text) return toast('Nothing to copy yet.');
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied to clipboard.');
  } catch {
    // Clipboard API unavailable (e.g. insecure context): fall back to a temporary textarea.
    const area = document.createElement('textarea');
    area.value = text;
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.append(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch {
      /* ignore */
    }
    area.remove();
    toast(ok ? 'Copied to clipboard.' : 'Could not copy. Select the text and copy it manually.');
  }
});

// ---- settings dialog -------------------------------------------------------------------------------------------

const dialog = $('settingsDialog');

function renderMute() {
  const btn = $('muteBtn');
  btn.setAttribute('aria-pressed', String(speech.muted));
  btn.title = speech.muted ? 'Unmute speech' : 'Mute speech';
  btn.setAttribute('aria-label', btn.title);
  $('muteIcon').style.opacity = speech.muted ? '0.45' : '1';
  $('setMute').checked = speech.muted;
}

function setMuted(muted) {
  settings.muted = muted;
  speech.setMuted(muted);
  saveSettings();
  renderMute();
}

function renderVoices(voices) {
  const select = $('setVoice');
  const lang = (navigator.language || 'en').slice(0, 2).toLowerCase();
  // Voices matching the browser language first, then the rest alphabetically.
  const sorted = [...voices].sort((a, b) => {
    const am = a.lang.toLowerCase().startsWith(lang) ? 0 : 1;
    const bm = b.lang.toLowerCase().startsWith(lang) ? 0 : 1;
    return am - bm || a.name.localeCompare(b.name);
  });
  select.textContent = '';
  select.append(new Option('Default voice', ''));
  for (const v of sorted) select.append(new Option(`${v.name} (${v.lang})`, v.voiceURI));
  select.value = settings.voiceURI;
}

/** Show the ElevenLabs fields only for that engine; the browser voice is the fallback otherwise. */
function renderEngineFields() {
  $('elevenFields').hidden = settings.engine !== 'elevenlabs';
  $('voiceLabel').textContent = settings.engine === 'elevenlabs' ? 'Fallback voice' : 'Voice';
}

function fillSettingsForm() {
  $('setEngine').value = settings.engine;
  $('setVoiceId').value = settings.voiceId;
  $('setTtsModel').value = settings.ttsModel;
  $('setElevenKey').value = settings.elevenKey;
  $('setTtsEndpoint').value = settings.ttsEndpoint;
  renderEngineFields();
  $('setEndpoint').value = settings.endpoint;
  $('setKey').value = settings.apiKey;
  $('setModel').value = settings.model;
  $('setVoice').value = settings.voiceURI;
  $('setRate').value = String(settings.rate);
  $('rateOut').textContent = settings.rate.toFixed(1);
  renderMute();
}

$('settingsBtn').addEventListener('click', () => {
  fillSettingsForm();
  if (typeof dialog.showModal === 'function') dialog.showModal();
  else dialog.setAttribute('open', '');
});
$('muteBtn').addEventListener('click', () => setMuted(!speech.muted));
$('setMute').addEventListener('change', (e) => setMuted(e.target.checked));

$('setEndpoint').addEventListener('input', (e) => {
  settings.endpoint = e.target.value.trim() || DEFAULT_ENDPOINT;
  saveSettings();
  refreshBackend();
});
$('setKey').addEventListener('input', (e) => {
  settings.apiKey = e.target.value.trim();
  saveSettings();
  refreshBackend();
});
$('setModel').addEventListener('input', (e) => {
  settings.model = e.target.value.trim();
  saveSettings();
});
$('setEngine').addEventListener('change', (e) => {
  settings.engine = e.target.value === 'browser' ? 'browser' : 'elevenlabs';
  speech.cancel();
  applySpeechSettings();
  saveSettings();
  renderEngineFields();
  renderVoicePill();
});
$('setVoiceId').addEventListener('input', (e) => {
  settings.voiceId = e.target.value.trim() || DEFAULT_VOICE_ID;
  applySpeechSettings();
  saveSettings();
});
$('setTtsModel').addEventListener('change', (e) => {
  settings.ttsModel = e.target.value;
  applySpeechSettings();
  saveSettings();
});
$('setElevenKey').addEventListener('input', (e) => {
  settings.elevenKey = e.target.value.trim();
  applySpeechSettings();
  saveSettings();
});
$('setTtsEndpoint').addEventListener('input', (e) => {
  settings.ttsEndpoint = e.target.value.trim() || DEFAULT_TTS_ENDPOINT;
  applySpeechSettings();
  saveSettings();
});
$('setVoice').addEventListener('change', (e) => {
  settings.voiceURI = e.target.value;
  speech.voiceURI = settings.voiceURI;
  saveSettings();
});
$('setRate').addEventListener('input', (e) => {
  settings.rate = Number(e.target.value);
  speech.rate = settings.rate;
  $('rateOut').textContent = settings.rate.toFixed(1);
  saveSettings();
});
$('testVoiceBtn').addEventListener('click', () => {
  speech.reset(); // retry ElevenLabs even if it failed recently
  speech.unlock();
  if (!speech.speak('Hello from SignSpeak.', { interrupt: true, force: true })) toast('Speech is not available in this browser.');
});

// ---- start-up ---------------------------------------------------------------------------------------------------------

renderMute();
renderSignList();
renderFrameCount();
speech.init(renderVoices);
if (!speech.supported) toast('Speech output is not supported in this browser.');
renderCameraPill();
renderModelPill();
renderVoicePill();
renderCard();
refreshBackend();
loadModel();
requestAnimationFrame(loop);
