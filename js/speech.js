// Text-to-speech through the Web Speech API: voice list, rate, mute.

export class Speech {
  constructor() {
    this.supported = typeof window !== 'undefined' && 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined';
    this.voices = [];
    this.voiceURI = '';
    this.rate = 1;
    this.muted = false;
  }

  /** Load voices (they arrive asynchronously in some browsers) and call onChange(voices) whenever they change. */
  init(onChange) {
    if (!this.supported) return;
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

  /** Speak text. Returns true if an utterance was queued. */
  speak(text, { interrupt = false, force = false } = {}) {
    const clean = (text || '').trim();
    if (!this.supported || !clean || (this.muted && !force)) return false;
    try {
      if (interrupt) window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(clean);
      const voice = this.voices.find((v) => v.voiceURI === this.voiceURI);
      if (voice) {
        utterance.voice = voice;
        utterance.lang = voice.lang;
      }
      utterance.rate = this.rate;
      window.speechSynthesis.speak(utterance);
      return true;
    } catch {
      return false;
    }
  }

  cancel() {
    try {
      if (this.supported) window.speechSynthesis.cancel();
    } catch {
      /* ignore */
    }
  }

  /** iOS Safari only allows speech after a user gesture; call this from a click handler. */
  unlock() {
    if (!this.supported) return;
    try {
      const u = new SpeechSynthesisUtterance(' ');
      u.volume = 0;
      window.speechSynthesis.speak(u);
    } catch {
      /* ignore */
    }
  }
}
