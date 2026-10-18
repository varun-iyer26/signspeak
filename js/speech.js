// Text-to-speech: ElevenLabs audio (via a backend or a user-supplied key) with the Web Speech API as fallback.

export const DEFAULT_VOICE_ID = 'JBFqnCBsd6RMkjVDRZzb'; // change this to switch the default ElevenLabs voice
export const DEFAULT_MODEL_ID = 'eleven_flash_v2_5'; // lowest latency
export const DEFAULT_TTS_ENDPOINT = '/api/tts';
export const TTS_MODELS = ['eleven_flash_v2_5', 'eleven_turbo_v2_5', 'eleven_multilingual_v2'];

const ELEVEN_URL = 'https://api.elevenlabs.io/v1/text-to-speech/';
const FETCH_TIMEOUT_MS = 6000; // give up on ElevenLabs and use the browser voice
const COOLDOWN_MS = 20000; // after a failure, go straight to the browser voice for a while
const CACHE_MAX = 30;
// Tiny silent WAV, played from a user gesture to unlock the shared audio element on iOS.
const SILENT_WAV = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=';

export class Speech {
  constructor() {
    this.browserSupported = typeof window !== 'undefined' && 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined';
    this.elevenSupported = typeof Audio !== 'undefined' && typeof fetch === 'function';
    this.supported = this.browserSupported || this.elevenSupported;
    this.voices = [];
    this.voiceURI = '';
    this.rate = 1;
    this.muted = false;
    // ElevenLabs settings
    this.engine = 'elevenlabs'; // 'elevenlabs' | 'browser'
    this.voiceId = DEFAULT_VOICE_ID;
    this.modelId = DEFAULT_MODEL_ID;
    this.apiKey = ''; // when set, call ElevenLabs directly instead of the endpoint
    this.endpoint = DEFAULT_TTS_ENDPOINT;
    this.onFallback = null; // called with a message, once per session, the first time we fall back
    this.onStatusChange = null; // called when activeVoice changes
    this.audio = this.elevenSupported ? new Audio() : null; // one reused element keeps iOS unlocked
    this.queue = []; // pending ElevenLabs utterances
    this.cache = new Map(); // key -> blob URL, oldest first (LRU)
    this.fellBack = false; // last ElevenLabs attempt failed
    this.notified = false;
    this.cooldownUntil = 0;
    this.running = false;
    this.current = null;
    this.endPlay = null;
    this.endBrowser = null;
  }

  /** Which voice is actually being used: 'elevenlabs', 'browser' (chosen) or 'fallback' (ElevenLabs failed). */
  get activeVoice() {
    if (this.engine !== 'elevenlabs') return 'browser';
    return this.fellBack ? 'fallback' : 'elevenlabs';
  }

  /** Load voices (they arrive asynchronously in some browsers) and call onChange(voices) whenever they change. */
  init(onChange) {
    if (!this.browserSupported) return;
    const load = () => {
      this.voices = window.speechSynthesis.getVoices();
      onChange?.(this.voices);
    };
    load();
    window.speechSynthesis.addEventListener?.('voiceschanged', load);
  }

  setMuted(muted) {
    this.muted = !!muted;
    if (this.muted) this.cancel();
  }

  /** Forget a previous ElevenLabs failure (call after the settings changed) so it is tried again. */
  reset() {
    this.cooldownUntil = 0;
    this.#setFellBack(false);
  }

  /** Speak text. Returns true if an utterance was queued. */
  speak(text, { interrupt = false, force = false } = {}) {
    const clean = (text || '').trim();
    if (!this.supported || !clean || (this.muted && !force)) return false;
    if (interrupt) this.#stopAll();
    if (this.engine === 'elevenlabs' && this.elevenSupported) {
      const item = { text: clean, controller: new AbortController(), cancelled: false, timedOut: false };
      item.audio = this.#getAudio(clean, item);
      item.audio.catch(() => {}); // handled in the queue runner
      this.queue.push(item);
      this.#run();
      return true;
    }
    return this.#speakBrowser(clean);
  }

  /** Stop everything: current audio, queued phrases and in-flight fetches. */
  cancel() {
    this.#stopAll();
  }

  /** iOS Safari only allows audio and speech after a user gesture; call this from a click handler. */
  unlock() {
    if (this.audio && !this.running) {
      try {
        this.audio.src = SILENT_WAV;
        this.audio.play()?.catch(() => {});
      } catch {
        /* ignore */
      }
    }
    if (!this.browserSupported) return;
    try {
      const u = new SpeechSynthesisUtterance(' ');
      u.volume = 0;
      window.speechSynthesis.speak(u);
    } catch {
      /* ignore */
    }
  }

  // ---- ElevenLabs ----------------------------------------------------------------

  /** Play queued phrases in order; a phrase that cannot be fetched or played is spoken by the browser voice instead. */
  async #run() {
    if (this.running) return;
    this.running = true;
    try {
      while (this.queue.length) {
        const item = this.queue.shift();
        this.current = item;
        let url = null;
        try {
          url = await item.audio;
          if (!item.cancelled) {
            this.#setFellBack(false);
            await this.#play(url);
          }
        } catch (err) {
          if (!item.cancelled) {
            url = null;
            this.#noteFallback(err);
          }
        }
        if (!url && !item.cancelled) await this.#speakBrowserAsync(item.text);
      }
    } finally {
      this.running = false;
      this.current = null;
    }
  }

  #play(url) {
    return new Promise((resolve, reject) => {
      const audio = this.audio;
      const done = (err) => {
        audio.onended = audio.onerror = null;
        this.endPlay = null;
        if (err) reject(err);
        else resolve();
      };
      this.endPlay = () => done();
      audio.onended = () => done();
      audio.onerror = () => done(new Error('audio playback failed'));
      audio.src = url;
      audio.defaultPlaybackRate = audio.playbackRate = this.rate; // set after src: loading resets it
      audio.play()?.catch(done);
    });
  }

  /** Resolve a blob URL for the phrase, from the cache or from ElevenLabs. */
  async #getAudio(text, item) {
    const key = `${this.voiceId}|${this.modelId}|${text}`;
    const hit = this.cache.get(key);
    if (hit) {
      this.cache.delete(key); // refresh its LRU position
      this.cache.set(key, hit);
      return hit;
    }
    if (Date.now() < this.cooldownUntil) throw new Error('ElevenLabs unavailable');
    const timer = setTimeout(() => {
      item.timedOut = true;
      item.controller.abort();
    }, FETCH_TIMEOUT_MS);
    let url;
    try {
      url = await this.#fetchAudio(text, item.controller.signal);
    } catch (err) {
      if (!item.cancelled) this.cooldownUntil = Date.now() + COOLDOWN_MS;
      throw item.timedOut ? new Error('timed out') : err;
    } finally {
      clearTimeout(timer);
    }
    this.cache.set(key, url);
    while (this.cache.size > CACHE_MAX) {
      const [oldKey, oldUrl] = this.cache.entries().next().value;
      this.cache.delete(oldKey);
      if (this.audio?.src !== oldUrl) URL.revokeObjectURL(oldUrl);
    }
    return url;
  }

  async #fetchAudio(text, signal) {
    const direct = !!this.apiKey;
    const res = direct
      ? await fetch(`${ELEVEN_URL}${encodeURIComponent(this.voiceId || DEFAULT_VOICE_ID)}?output_format=mp3_44100_128`, {
          method: 'POST',
          headers: { 'xi-api-key': this.apiKey, 'Content-Type': 'application/json', Accept: 'audio/mpeg' },
          body: JSON.stringify({ text, model_id: this.modelId || DEFAULT_MODEL_ID }),
          signal,
        })
      : await fetch(this.endpoint || DEFAULT_TTS_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          // Only send a custom voice so the server's ELEVENLABS_VOICE_ID default can apply.
          body: JSON.stringify({ text, voiceId: this.voiceId !== DEFAULT_VOICE_ID ? this.voiceId : undefined, modelId: this.modelId }),
          signal,
        });
    if (!res.ok) {
      let reason = `HTTP ${res.status}`;
      try {
        const data = await res.json();
        const msg = data?.error ?? data?.detail?.message;
        if (typeof msg === 'string' && msg) reason = msg;
      } catch {
        /* not JSON */
      }
      throw new Error(reason);
    }
    const type = res.headers.get('content-type') || '';
    if (type && !/audio|octet-stream/i.test(type)) throw new Error('unexpected response type');
    const blob = await res.blob();
    if (!blob.size) throw new Error('empty audio');
    return URL.createObjectURL(new Blob([blob], { type: 'audio/mpeg' }));
  }

  #setFellBack(value) {
    if (this.fellBack === value) return;
    this.fellBack = value;
    this.onStatusChange?.();
  }

  /** Switch the status to "fallback" and tell the user once per session. */
  #noteFallback(err) {
    this.#setFellBack(true);
    if (this.notified) return;
    this.notified = true;
    const reason = err?.message ? ` (${err.message})` : '';
    this.onFallback?.(`ElevenLabs voice unavailable${reason}. Using the browser voice instead.`);
  }

  // ---- browser voice ---------------------------------------------------------------

  #utterance(text) {
    const utterance = new SpeechSynthesisUtterance(text);
    const voice = this.voices.find((v) => v.voiceURI === this.voiceURI);
    if (voice) {
      utterance.voice = voice;
      utterance.lang = voice.lang;
    }
    utterance.rate = this.rate;
    return utterance;
  }

  #speakBrowser(text) {
    if (!this.browserSupported) return false;
    try {
      window.speechSynthesis.speak(this.#utterance(text));
      return true;
    } catch {
      return false;
    }
  }

  /** Speak with the browser voice and resolve when it finishes (or is cancelled). */
  #speakBrowserAsync(text) {
    if (!this.browserSupported) return Promise.resolve();
    return new Promise((resolve) => {
      let timer = 0;
      const done = () => {
        clearTimeout(timer);
        if (this.endBrowser === done) this.endBrowser = null;
        resolve();
      };
      try {
        const utterance = this.#utterance(text);
        utterance.onend = utterance.onerror = done;
        this.endBrowser = done;
        timer = setTimeout(done, 5000 + text.length * 120 / this.rate); // some browsers never fire onend
        window.speechSynthesis.speak(utterance);
      } catch {
        done();
      }
    });
  }

  #stopAll() {
    for (const item of [...this.queue, this.current]) {
      if (!item) continue;
      item.cancelled = true;
      item.controller.abort();
    }
    this.queue.length = 0;
    this.endPlay?.();
    try {
      this.audio?.pause();
    } catch {
      /* ignore */
    }
    this.endBrowser?.();
    try {
      if (this.browserSupported) window.speechSynthesis.cancel();
    } catch {
      /* ignore */
    }
  }
}
