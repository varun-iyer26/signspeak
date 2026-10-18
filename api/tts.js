// Vercel serverless function: turns text into speech with ElevenLabs and returns MP3 audio.
// The API key never leaves the server.
//
// Contract:
//   POST /api/tts  { "text": "<1-1000 chars>", "voiceId"?: "<alphanumeric, <=64>", "modelId"?: "<<=64>" }
//   200  audio/mpeg bytes, Cache-Control: no-store
//   4xx/5xx { "error": "<message>" }

const ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1/text-to-speech";
const OUTPUT_FORMAT = "mp3_44100_128";
const DEFAULT_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"; // "George", a premade voice; keep in sync with js/speech.js
const DEFAULT_MODEL_ID = "eleven_flash_v2_5"; // low-latency model
const MAX_TEXT_CHARS = 1000;
const MAX_ID_CHARS = 64;
const VOICE_ID_RE = /^[A-Za-z0-9]{1,64}$/;
const MODEL_ID_RE = /^[A-Za-z0-9_.-]{1,64}$/;

export default async function handler(req, res) {
  setCors(res);

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST, OPTIONS");
    res.status(405).json({ error: "Method not allowed. Use POST." });
    return;
  }

  const parsed = parseBody(req.body);
  if (parsed.error) {
    res.status(400).json({ error: parsed.error });
    return;
  }

  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    res
      .status(500)
      .json({ error: "Server is missing ELEVENLABS_API_KEY. Add it in the Vercel project settings." });
    return;
  }

  const voiceId = parsed.value.voiceId || process.env.ELEVENLABS_VOICE_ID || DEFAULT_VOICE_ID;
  const modelId = parsed.value.modelId || process.env.ELEVENLABS_MODEL_ID || DEFAULT_MODEL_ID;
  const url =
    `${ELEVENLABS_BASE_URL}/${encodeURIComponent(voiceId)}` +
    `?output_format=${OUTPUT_FORMAT}`;

  let upstream;
  try {
    upstream = await fetch(url, {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "content-type": "application/json",
        accept: "audio/mpeg",
      },
      body: JSON.stringify({ text: parsed.value.text, model_id: modelId }),
    });
  } catch {
    res.status(502).json({ error: "Could not reach the ElevenLabs API." });
    return;
  }

  if (!upstream.ok) {
    let data = null;
    try {
      data = await upstream.json();
    } catch {
      // Non-JSON upstream body; fall back to the status message below.
    }
    const message = upstreamMessage(data) || `ElevenLabs API returned HTTP ${upstream.status}.`;
    res.status(502).json({ error: redact(message, apiKey) });
    return;
  }

  let audio;
  try {
    audio = Buffer.from(await upstream.arrayBuffer());
  } catch {
    res.status(502).json({ error: "Could not read audio from the ElevenLabs API." });
    return;
  }

  res.setHeader("Content-Type", "audio/mpeg");
  res.setHeader("Cache-Control", "no-store");
  res.status(200).send(audio);
}

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", process.env.ALLOWED_ORIGIN || "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "86400");
}

// Returns { value: { text, voiceId, modelId } } or { error: message }.
function parseBody(raw) {
  let body = raw;
  if (typeof body === "string") {
    try {
      body = JSON.parse(body);
    } catch {
      return { error: "Request body must be valid JSON." };
    }
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return { error: "Request body must be a JSON object." };
  }

  const text = typeof body.text === "string" ? body.text.trim() : "";
  if (text.length === 0) {
    return { error: "`text` must be a non-empty string." };
  }
  if (text.length > MAX_TEXT_CHARS) {
    return { error: `\`text\` must be at most ${MAX_TEXT_CHARS} characters.` };
  }

  let voiceId;
  if (body.voiceId !== undefined && body.voiceId !== null) {
    if (typeof body.voiceId !== "string" || !VOICE_ID_RE.test(body.voiceId)) {
      return { error: `\`voiceId\` must be 1 to ${MAX_ID_CHARS} letters or digits.` };
    }
    voiceId = body.voiceId;
  }

  let modelId;
  if (body.modelId !== undefined && body.modelId !== null) {
    if (typeof body.modelId !== "string" || !MODEL_ID_RE.test(body.modelId)) {
      return {
        error: `\`modelId\` must be 1 to ${MAX_ID_CHARS} letters, digits, underscores, dots or hyphens.`,
      };
    }
    modelId = body.modelId;
  }

  return { value: { text, voiceId, modelId } };
}

// ElevenLabs errors look like { detail: { message } } or { detail: "..." }.
function upstreamMessage(data) {
  const detail = data?.detail;
  if (typeof detail === "string") return detail;
  if (typeof detail?.message === "string") return detail.message;
  return null;
}

// Belt and braces: make sure the key can never be echoed back in an error.
function redact(message, apiKey) {
  return String(message).split(apiKey).join("[redacted]");
}
