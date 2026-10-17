// MediaPipe HandLandmarker wrapper plus overlay drawing.
// The library is imported lazily so a CDN failure cannot break app start-up.

const MP_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

export class HandTracker {
  constructor() {
    this.mp = null;
    this.fileset = null;
    this.landmarker = null;
    this.delegate = null; // 'GPU' | 'CPU'
    this.lastVideoTime = -1;
    this.drawing = null;
    this.drawingCtx = null;
    this._recovering = false;
  }

  get ready() {
    return !!this.landmarker;
  }

  /** Download and initialise the model. GPU first, CPU if that fails. Throws if neither works. */
  async load() {
    if (!this.mp) this.mp = await import(MP_URL);
    if (!this.fileset) this.fileset = await this.mp.FilesetResolver.forVisionTasks(WASM_URL);
    try {
      this.landmarker = await this._create('GPU');
      this.delegate = 'GPU';
    } catch {
      this.landmarker = await this._create('CPU');
      this.delegate = 'CPU';
    }
  }

  _create(delegate) {
    return this.mp.HandLandmarker.createFromOptions(this.fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numHands: 2,
    });
  }

  /**
   * Run detection on the current video frame.
   * Returns null if there is no new frame to process, otherwise an array of
   * {landmarks, handedness: 'Left'|'Right', score}.
   */
  detect(video, nowMs) {
    if (!this.landmarker || video.readyState < 2 || video.videoWidth === 0) return null;
    if (video.currentTime === this.lastVideoTime) return null;
    this.lastVideoTime = video.currentTime;
    let result;
    try {
      result = this.landmarker.detectForVideo(video, nowMs);
    } catch {
      // A GPU delegate can fail at run time; rebuild on CPU once.
      this._fallbackToCpu();
      return null;
    }
    const hands = [];
    (result.landmarks || []).forEach((landmarks, i) => {
      const category = (result.handedness || result.handednesses || [])[i]?.[0];
      hands.push({ landmarks, handedness: category?.categoryName || 'Right', score: category?.score ?? 1 });
    });
    return hands;
  }

  async _fallbackToCpu() {
    if (this._recovering || this.delegate === 'CPU') return;
    this._recovering = true;
    const old = this.landmarker;
    this.landmarker = null;
    try {
      old?.close();
      this.landmarker = await this._create('CPU');
      this.delegate = 'CPU';
    } finally {
      this._recovering = false;
    }
  }

  /** Draw landmarks on the overlay canvas, which shares the video's intrinsic size. */
  draw(canvas, video, hands) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.mp || !hands.length) return;
    if (this.drawingCtx !== ctx) {
      this.drawing = new this.mp.DrawingUtils(ctx);
      this.drawingCtx = ctx;
    }
    for (const hand of hands) {
      this.drawing.drawConnectors(hand.landmarks, this.mp.HandLandmarker.HAND_CONNECTIONS, { color: '#5eead4', lineWidth: 3 });
      this.drawing.drawLandmarks(hand.landmarks, { color: '#f472b6', fillColor: '#0b1020', lineWidth: 2, radius: 4 });
    }
  }
}
