// Frame capture for AI mode and the client for the interpret endpoint / direct Anthropic calls.

export const DEFAULT_ENDPOINT = '/api/interpret';
export const DEFAULT_MODEL = 'claude-haiku-5-5';
const ANTHROPIC_URL = 'https://api.anthropic.com/v1/messages';
const TIMEOUT_MS = 30000;

export const SYSTEM_PROMPT =
  "You are an expert sign language interpreter (ASL primarily). You receive sequential frames from a camera, in chronological order, of a person signing. Translate what they are signing into one natural English sentence. Use the prior transcript only as context. If the signing is unclear, give your best guess, prefixed with '(unsure)'. If no signing is visible, reply with exactly: [no sign]. Output only the translation.";

/** Ring buffer of downscaled JPEG frames (base64, no data: prefix). */
export class FrameBuffer {
  constructor({ max = 8, width = 384, intervalMs = 333, quality = 0.7 } = {}) {
    this.max = max;
    this.width = width;
    this.intervalMs = intervalMs;
    this.quality = quality;
    this.frames = [];
    this.lastCapture = 0;
    this.canvas = document.createElement('canvas');
  }

  get size() {
    return this.frames.length;
  }

  /** Capture the current video frame if the capture interval has elapsed. Returns true if one was stored. */
  capture(video, now = performance.now(), force = false) {
    if (!force && now - this.lastCapture < this.intervalMs) return false;
    if (!video.videoWidth) return false;
    this.lastCapture = now;
    const w = this.width;
    const h = Math.round((video.videoHeight / video.videoWidth) * w);
    this.canvas.width = w;
    this.canvas.height = h;
    const ctx = this.canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, w, h);
    const data = this.canvas.toDataURL('image/jpeg', this.quality).split(',')[1];
    if (!data) return false;
    this.frames.push(data);
    if (this.frames.length > this.max) this.frames.shift();
    return true;
  }

  /** Return the buffered frames and empty the buffer. */
  take() {
    const out = this.frames;
    this.frames = [];
    return out;
  }

  clear() {
    this.frames = [];
  }
}

/** The text block that follows the images; identical in client and server. */
function userText(context) {
  const ctx = (context || '').trim();
  return `Prior transcript (context only, may be empty): ${ctx || '(none)'}\nTranslate the signing in these frames.`;
}

function normalizeReply(text) {
  const t = (text || '').trim();
  return /^\[no sign\]\.?$/i.test(t) ? '' : t;
}

async function fetchJson(url, init) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    const data = await res.json().catch(() => null);
    return { res, data };
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('The AI request timed out.');
    throw new Error('Could not reach the AI service (network error).');
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Translate frames into an English sentence ('' when no sign was seen).
 * With an API key set this calls Anthropic directly from the browser;
 * otherwise it posts to the configured endpoint (default same-origin /api/interpret).
 */
export async function interpretFrames({ frames, context, endpoint, apiKey, model }) {
  if (apiKey) {
    const { res, data } = await fetchJson(ANTHROPIC_URL, {
      method: 'POST',
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'anthropic-dangerous-direct-browser-access': 'true',
      },
      body: JSON.stringify({
        model: model || DEFAULT_MODEL,
        max_tokens: 200,
        system: SYSTEM_PROMPT,
        messages: [
          {
            role: 'user',
            content: [
              ...frames.map((data64) => ({ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: data64 } })),
              { type: 'text', text: userText(context) },
            ],
          },
        ],
      }),
    });
    if (!res.ok) throw new Error(data?.error?.message || `Anthropic API error (HTTP ${res.status}).`);
    const text = (data?.content || []).filter((b) => b.type === 'text').map((b) => b.text).join('');
    return normalizeReply(text);
  }

  const { res, data } = await fetchJson(endpoint || DEFAULT_ENDPOINT, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ frames, context: (context || '').slice(-200) }),
  });
  if (!res.ok) {
    if (res.status === 404 || res.status === 405) {
      throw new Error('No AI backend at this address. Deploy the Vercel API or add your own API key in Settings.');
    }
    throw new Error(data?.error || `AI backend error (HTTP ${res.status}).`);
  }
  return normalizeReply(data?.text);
}

/** Is something answering at the endpoint? (The server answers OPTIONS for CORS.) */
export async function checkBackend(endpoint) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 4000);
  try {
    const res = await fetch(endpoint || DEFAULT_ENDPOINT, { method: 'OPTIONS', signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}
