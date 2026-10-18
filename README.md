# SignSpeak

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fvarun-iyer26%2Fsignspeak)

**SignSpeak** turns sign language into text and speech. Point a camera at someone signing, and it shows the translation on screen and speaks it aloud. It runs in desktop and phone browsers and can be installed as a PWA.

Live demo (GitHub Pages): https://varun-iyer26.github.io/signspeak/

> Demo GIF placeholder: a short recording of fingerspelling "HELLO" and the transcript updating will go here.

## Features

- **Works in the browser, no build step.** Plain HTML, CSS and ES-module JavaScript. Deploys as static files to GitHub Pages, and optionally to Vercel for the AI mode.
- **Camera for signing someone else.** Starts on the rear camera on phones. A button switches to the front camera, and the video is mirrored only when the front camera is active.
- **On-device hand tracking.** MediaPipe Tasks Vision HandLandmarker tracks up to two hands and draws their landmarks over the video.
- **Three recognition modes:**
  1. **Fingerspelling (on device):** a rule-based ASL alphabet classifier for the static letters A to Y (J needs motion, so it is not included).
  2. **Custom signs (on device):** teach the app your own signs with a label and a short recording. It recognizes them with k-NN and saves your dataset in the browser.
  3. **AI interpret (Claude vision):** sends a short burst of camera frames to Claude and gets back one English sentence.
- **Speech output.** Uses the Web Speech API with a voice picker, a rate slider and a mute toggle.
- **Transcript panel.** Shows committed text and a live "current guess" with a confidence bar. Buttons for Speak, Space, Backspace, Clear and Copy.
- **Accessible and responsive.** Dark theme by default, large touch targets, `aria-live` on the transcript, and no horizontal scrolling at 375 px.

## How it works

```
 camera (rear or front)
        |
        v
 MediaPipe HandLandmarker (on device, up to 2 hands)
        |                                  |
        v                                  v
 landmarks -> features.js            ring buffer of JPEG frames
 (wrist-relative, scaled,            (384 px wide, ~3 per second,
  Right hand then Left)               max 8)
        |                                  |
        v                                  |  signing pauses (no hands
 +--------------+---------------+          |  for 1 s) or "Interpret now"
 | fingerspell  | custom signs  |          v
 | rule-based   | k-NN, k = 5   |   AI interpret
 +--------------+---------------+     |                    |
        |                             v                    v
        |                     /api/interpret         direct mode:
        |                     (Vercel function)      browser calls
        |                             |              Anthropic API
        |                             v                    |
        |                        Claude vision <-----------+
        |                             |
        v                             v
 committed letters or words  <--  sentence
        |
        v
 transcript panel  -->  speechSynthesis (spoken output)
```

- **Fingerspelling** reads each hand's landmarks and decides which fingers are extended or curled, where the thumb sits, and how the fingertips relate to each other. A letter is committed after about 12 consecutive frames with confidence of at least 0.6, and not again until the hand changes or leaves the frame. A hand that is absent for more than 1.2 s inserts a space. Absent for more than 3 s, the pending sentence is spoken.
- **Custom signs** record about 30 feature vectors per sign. A label is committed after it holds for about 10 frames, and words are spoken after a pause.
- **AI interpret** sends the buffered frames and the last ~200 characters of transcript to Claude, which replies with one sentence. The model is `claude-haiku-5-5` by default.

## Running locally

The camera requires `localhost` or HTTPS, so open the app from one of these:

```sh
npx serve .
# or
python3 -m http.server 8000
```

Then open http://localhost:3000 (for `serve`) or http://localhost:8000 (for Python).

The on-device modes work with any static server. AI mode calls `/api/interpret`, which only exists when you run the Vercel function. To test that locally, use `npx vercel dev` with `ANTHROPIC_API_KEY` set. Otherwise, use direct mode in Settings.

Run the syntax checks with `npm run check`.

## Deploying

### GitHub Pages (on-device modes plus direct mode)

1. In the repository settings, open **Pages** and set **Source** to **GitHub Actions**.
2. Push to `main`. The workflow in `.github/workflows/pages.yml` publishes the repo root. You can also run it by hand from the Actions tab.
3. Open https://varun-iyer26.github.io/signspeak/.

Fingerspelling and custom signs work fully. For AI mode, there is no backend on Pages, so open **Settings**, paste your own Anthropic API key, and choose **direct mode**. The key is stored only in this browser's `localStorage`, and the app labels it that way. The browser then calls the Anthropic API directly.

> **Warning:** in direct mode the key is visible to anyone who can use this browser profile, and to any script that runs on the page. Use a key with a spending limit, and remove it from Settings when you are done.

### Vercel (all modes)

Click the **Deploy with Vercel** button at the top of this page, or import the repository in the Vercel dashboard. Then set these environment variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key. It stays on the server and is never returned to the browser. |
| `ANTHROPIC_MODEL` | No | Overrides the model. Defaults to `claude-haiku-5-5`. |
| `ALLOWED_ORIGIN` | No | The origin allowed to call `/api/interpret` and `/api/tts` through CORS. Defaults to `*`. |
| `ELEVENLABS_API_KEY` | No (natural voice only) | Your ElevenLabs API key, used by `/api/tts`. It stays on the server. |
| `ELEVENLABS_VOICE_ID` | No | Default ElevenLabs voice. Defaults to the premade voice "George" (`JBFqnCBsd6RMkjVDRZzb`). |
| `ELEVENLABS_MODEL_ID` | No | ElevenLabs model. Defaults to `eleven_flash_v2_5`, the low-latency model. |

The function accepts at most 8 frames, each up to 400 KB of base64, and sets `Access-Control-Allow-Origin` from `ALLOWED_ORIGIN`. It never logs the key or the frames.

## Natural voice (ElevenLabs)

Speech output can use an ElevenLabs voice instead of the browser's built-in voice. It is optional. The server function is `api/tts.js`. It accepts `{ "text", "voiceId"?, "modelId"? }` and returns MP3 audio.

- **On Vercel (key stays on the server):** open **Settings → Environment Variables** in your Vercel project and add `ELEVENLABS_API_KEY`. You can also add `ELEVENLABS_VOICE_ID` and `ELEVENLABS_MODEL_ID` (see the table above). For `vercel dev`, put the same variables in `.env.local` in the project root. Do not commit that file.
- **On GitHub Pages:** there is no server, so the key cannot be set for the site. Open **Settings** and paste your own ElevenLabs key. It is stored only in your browser.
- **Fallback:** if ElevenLabs is not configured, or a request fails, the app uses the browser's built-in voice.
- **Picking a voice:** browse the ElevenLabs Voice Library on elevenlabs.io and copy the ID of a voice you like. Or clone your own voice in ElevenLabs and copy the ID of the result. Set the ID as `ELEVENLABS_VOICE_ID`, or choose it in Settings.

## Limitations and accuracy

- **Fingerspelling is heuristic.** It uses hand-tuned geometry rules, not a trained model, so similar-looking letters can be confused. It covers only the static letters A to Y, not J or Z, which need motion. It is sensitive to lighting, camera angle and hand distance.
- **Custom signs work best when trained on the signer.** The k-NN model learns from the examples you record, so a model trained on one person may not transfer well to another.
- **AI mode is approximate.** Claude sees a few low-resolution frames, not continuous video, so it can miss fast movement, facial grammar and context. It may also be slow, and each request costs money.
- **This is not a replacement for a human interpreter.** Use SignSpeak as an aid for casual conversation, learning and exploration. For medical, legal, educational or other high-stakes settings, use a qualified sign language interpreter.

## Privacy

- **On-device modes never send video.** Fingerspelling and custom signs run entirely in the browser. Hand landmarks and the custom-sign dataset stay on the device, in `localStorage`. You can export the dataset as JSON, import it again, or delete individual labels.
- **AI mode sends frames only when you interpret.** When signing pauses or you press "Interpret now", up to 8 JPEG frames at 384 px wide, plus the last ~200 characters of transcript, are sent to the AI endpoint. On Vercel, the function forwards them to Anthropic. In direct mode, your browser sends them to Anthropic.
- **The API key is never sent to this project's servers** in direct mode, and on Vercel it is read only on the server.

## Hackathon pitch

**The problem.** Most hearing people do not know sign language, and most Deaf and hard-of-hearing people cannot get an interpreter for everyday conversations. Apps that teach signs do not help with the conversation itself.

**The idea.** SignSpeak is a lens. Point a phone at someone signing, and it shows the words on screen and speaks them out loud. No app install is needed for the browser version, and the main modes run on the device.

**Why it is different.**
- Three modes that trade off privacy, speed and coverage: on-device fingerspelling, teachable custom signs, and AI sentence translation.
- The on-device modes never upload video.
- It works on the GitHub Pages free tier, with no backend required for the core features.

**What is next.**
- Train the fingerspelling model on real signers, and add J and Z using motion.
- Let communities share custom-sign datasets, with consent.
- Add more sign languages.

**Built with** MediaPipe Tasks Vision, the Web Speech API, Claude vision, Vercel functions and GitHub Pages.

## License

MIT. See [LICENSE](LICENSE).
