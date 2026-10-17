// Vercel serverless function: turns a short burst of sign-language camera frames
// into one English sentence with Claude vision. The API key never leaves the server.
//
// Contract (same shape the browser client in js/ai.js sends):
//   POST /api/interpret  { "frames": ["<base64 jpeg>", ...], "context": "<recent transcript>" }
//   200  { "text": "<English sentence>" }
//   4xx/5xx { "error": "<message>" }

const ANTHROPIC_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
const DEFAULT_MODEL = "claude-haiku-5-5";
const MAX_TOKENS = 200;
const MAX_FRAMES = 8;
const MAX_FRAME_CHARS = 400 * 1024; // base64 characters per frame
const MAX_CONTEXT_CHARS = 200;
const BASE64_RE = /^[A-Za-z0-9+/]*={0,2}$/;

const SYSTEM_PROMPT =
  "You are an expert sign language interpreter (ASL primarily). You receive sequential frames " +
  "from a camera, in chronological order, of a person signing. Translate what they are signing " +
  "into one natural English sentence. Use the prior transcript only as context. If the signing " +
  "is unclear, give your best guess, prefixed with '(unsure)'. If no signing is visible, reply " +
  "with exactly: [no sign]. Output only the translation.";

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

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    res
      .status(500)
      .json({ error: "Server is missing ANTHROPIC_API_KEY. Add it in the Vercel project settings." });
    return;
  }

  const payload = buildPayload(parsed.value.frames, parsed.value.context);

  let upstream;
  try {
    upstream = await fetch(ANTHROPIC_URL, {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
      },
      body: JSON.stringify(payload),
    });
  } catch {
    res.status(502).json({ error: "Could not reach the Anthropic API." });
    return;
  }

  let data = null;
  try {
    data = await upstream.json();
  } catch {
    // Non-JSON upstream body; handled below by the status check.
  }

  if (!upstream.ok) {
    const message = data?.error?.message || `Anthropic API returned HTTP ${upstream.status}.`;
    res.status(502).json({ error: redact(message, apiKey) });
    return;
  }

  // The reply is the concatenation of its text blocks, trimmed.
  const text = (data?.content ?? [])
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("")
    .trim();
  res.status(200).json({ text });
}

function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", process.env.ALLOWED_ORIGIN || "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "86400");
}

// Returns { value: { frames, context } } or { error: message }.
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

  const { frames } = body;
  const context = body.context ?? "";

  if (!Array.isArray(frames) || frames.length === 0) {
    return { error: "`frames` must be a non-empty array." };
  }
  if (frames.length > MAX_FRAMES) {
    return { error: `At most ${MAX_FRAMES} frames are allowed.` };
  }
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (typeof frame !== "string" || frame.length === 0) {
      return { error: `frames[${i}] must be a non-empty base64 string.` };
    }
    // Check the length before the regex so oversized input is rejected cheaply.
    if (frame.length > MAX_FRAME_CHARS) {
      return { error: `frames[${i}] is larger than 400 KB.` };
    }
    if (!BASE64_RE.test(frame)) {
      return { error: `frames[${i}] is not plain base64 (send it without a data: prefix).` };
    }
  }
  if (typeof context !== "string") {
    return { error: "`context` must be a string." };
  }

  return { value: { frames, context: context.slice(-MAX_CONTEXT_CHARS) } };
}

// One user message: the image blocks in chronological order, then a text block.
function buildPayload(frames, context) {
  const content = frames.map((data) => ({
    type: "image",
    source: { type: "base64", media_type: "image/jpeg", data },
  }));
  content.push({
    type: "text",
    text: `Prior transcript (context only, may be empty): ${context || "(none)"}\nTranslate the signing in these frames.`,
  });

  return {
    model: process.env.ANTHROPIC_MODEL || DEFAULT_MODEL,
    max_tokens: MAX_TOKENS,
    system: SYSTEM_PROMPT,
    messages: [{ role: "user", content }],
  };
}

// Belt and braces: make sure the key can never be echoed back in an error.
function redact(message, apiKey) {
  return String(message).split(apiKey).join("[redacted]");
}
