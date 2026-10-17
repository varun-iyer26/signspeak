# Master Prompt — SignSpeak

> Build **SignSpeak**: a web app (works in desktop and phone browsers, installable as a PWA) that you point a camera at someone signing, and it shows the translation as text and speaks it aloud.

## Goals
- Runs fully in the browser, with no build step: plain HTML, CSS and ES-module JavaScript. Deployable as static files (GitHub Pages) and optionally on Vercel, which adds an AI backend.
- Camera: rear camera by default on phones (you are pointing it at *someone else*), a button to switch to the front camera, and the video mirrored only when the front camera is active.
- Hand tracking on the device with **MediaPipe Tasks Vision HandLandmarker** (up to 2 hands), with the landmarks drawn over the video.
- Speech output through the Web Speech API (`speechSynthesis`): a voice picker, a rate slider and a mute toggle.
- Three recognition modes, picked from tabs:
  1. **Fingerspelling (on-device):** a rule-based ASL alphabet classifier built on landmark geometry (finger extended or curled, thumb position, fingertip distances and angles). It covers the static letters A–Y except J. A letter is committed once the same prediction holds for about 12 consecutive frames with confidence ≥ 0.6, and the same letter is not committed again until the hand changes or drops out. No hand for more than 1.2 s inserts a space (a word break). No hand for more than 3 s with new text speaks the pending sentence.
  2. **Custom signs (on-device, teachable):** the user types a label (a word or phrase), holds a "Record" button for about 2 s to capture about 30 feature vectors, and repeats this for each sign. Recognition is k-NN (k=5) with a distance threshold. A label is committed once it holds for about 10 frames, with the same cooldown rules as fingerspelling, and words are spoken after a pause. The dataset is saved to `localStorage`, can be exported and imported as JSON, and each label can be deleted.
  3. **AI interpret (Claude vision):** while hands are visible, keep a ring buffer of JPEG frames downscaled to 384 px wide, captured about 3 times per second, holding at most 8. When signing pauses (no hands for 1 s), or when the user presses "Interpret now", send the frames to the AI endpoint, show the returned sentence and speak it.
- Transcript panel: committed text, a live "current guess" with a confidence bar, and buttons for Speak, Space, Backspace, Clear and Copy.
- Responsive and mobile-first, with a dark theme by default, large touch targets, `aria-live` on the transcript, and no horizontal scrolling at 375 px.

## File layout and ownership
```
index.html            # Agent A
manifest.webmanifest  # Agent A
icon.svg              # Agent A
css/styles.css        # Agent A
js/app.js             # Agent A: bootstraps, UI wiring, mode switching
js/camera.js          # Agent A: getUserMedia, switching between front and rear
js/hands.js           # Agent A: HandLandmarker wrapper and overlay drawing
js/features.js        # Agent A: landmark normalization into a feature vector
js/fingerspell.js     # Agent A: rule-based ASL letter classifier
js/knn.js             # Agent A: teachable k-NN plus persistence
js/speech.js          # Agent A: TTS, voice list, rate, mute
js/ai.js              # Agent A: frame capture and AI client
api/interpret.js      # Agent B: Vercel serverless function
package.json          # Agent B: {"type":"module"} plus name and scripts; no runtime deps
vercel.json           # Agent B
.github/workflows/pages.yml  # Agent B: deploy the repo root to GitHub Pages
README.md, LICENSE (MIT), .gitignore  # Agent B
```
Agents never edit each other's files.

## Fixed technical contracts
- MediaPipe: `import { FilesetResolver, HandLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14"`; the wasm is at `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm`; the model is at `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`; `runningMode: "VIDEO"`, `numHands: 2`, and `delegate: "GPU"` that falls back to CPU on error.
- Features: for each hand, take the 21 landmarks relative to the wrist, scale by the wrist-to-middle-MCP distance, and flatten x, y, z (63 values). Order the hands by handedness (Right first, then Left) and zero-fill a missing hand, which gives 126 values. If the front camera is mirrored, flip x so a sign recorded on either camera matches.
- **AI endpoint contract:** `POST {endpoint}` with JSON `{ "frames": ["<base64 jpeg, no data: prefix>", ...], "context": "<the last ~200 chars of transcript>" }` returns `200 { "text": "<English sentence>" }` or `4xx/5xx { "error": "..." }`. The default endpoint is the same-origin `/api/interpret`.
- **Direct mode (no backend, e.g. GitHub Pages):** in Settings the user can paste their own Anthropic API key, which is stored only in this browser's `localStorage` and labeled as such. The client then calls `https://api.anthropic.com/v1/messages` directly with the headers `x-api-key`, `anthropic-version: 2023-06-01`, `content-type: application/json` and `anthropic-dangerous-direct-browser-access: true`. The client and the server use the same prompt and payload shape.
- Claude request: model `claude-haiku-5-5` by default (the server can override it with the `ANTHROPIC_MODEL` env var, and Settings can override it in direct mode), `max_tokens: 200`. The single user message holds the image blocks in order, each `{type:"image", source:{type:"base64", media_type:"image/jpeg", data}}`, followed by a text block. System prompt: *"You are an expert sign language interpreter (ASL primarily). You receive sequential frames from a camera, in chronological order, of a person signing. Translate what they are signing into one natural English sentence. Use the prior transcript only as context. If the signing is unclear, give your best guess, prefixed with '(unsure)'. If no signing is visible, reply with exactly: [no sign]. Output only the translation."* The reply is the concatenated text blocks, trimmed. The client ignores `[no sign]`.
- Server function (`api/interpret.js`): a Vercel Node function using `export default async function handler(req, res)` and global `fetch`, with the API key in `ANTHROPIC_API_KEY`. It validates input (at most 8 frames, each at most 400 KB of base64), sets CORS headers (`Access-Control-Allow-Origin` from the `ALLOWED_ORIGIN` env var, defaulting to `*`), handles `OPTIONS`, and never logs the key or frames.

## Quality bar
- Every JS file passes `node --check`. Nothing throws on load in a browser that has no camera: show a friendly error card with a retry button.
- Status indicators for model loading, the camera and the backend.
- Code is commented where the logic isn't obvious (especially the fingerspelling rules), with no dead code.
- The README covers what it is, a demo GIF placeholder, features, how it works (with a diagram in text), running locally (`npx serve .` or `python3 -m http.server`; the camera needs localhost or HTTPS), deploying to GitHub Pages (on-device modes plus direct mode) and to Vercel (all modes, with `ANTHROPIC_API_KEY`), limitations and accuracy notes (fingerspelling is heuristic; custom signs work best when trained on the signer; AI mode is approximate; this is not a replacement for a human interpreter), privacy (on-device modes never send video), and a hackathon pitch section.
