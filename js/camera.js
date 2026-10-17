// getUserMedia wrapper: rear camera by default, switchable to the front camera.

export class Camera {
  constructor(video) {
    this.video = video;
    this.stream = null;
    this.facing = 'environment'; // the facing mode that was actually obtained
    this.mirrored = false; // only the front camera is mirrored
    this._seq = 0; // guards against overlapping start() calls
  }

  get active() {
    return !!this.stream;
  }

  /** Start (or restart) the camera. Resolves once the video has metadata. Throws on failure. */
  async start(facing = this.facing) {
    if (!navigator.mediaDevices?.getUserMedia) {
      const err = new Error('Camera API unavailable');
      err.name = 'NotSupportedError';
      throw err;
    }
    const seq = ++this._seq;
    this.stop();

    const wanted = { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } };
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: wanted });
    } catch (err) {
      // Some devices reject the constraints; retry with anything that works.
      if (err.name === 'OverconstrainedError' || err.name === 'NotFoundError') {
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
      } else {
        throw err;
      }
    }
    if (seq !== this._seq) {
      stream.getTracks().forEach((t) => t.stop());
      throw Object.assign(new Error('Superseded'), { name: 'AbortError' });
    }

    const reported = stream.getVideoTracks()[0]?.getSettings?.().facingMode;
    // Desktop webcams usually report no facing mode and point at the user.
    this.facing = reported || (navigator.maxTouchPoints > 0 ? facing : 'user');
    this.mirrored = this.facing === 'user';
    this.stream = stream;
    this.video.srcObject = stream;
    await new Promise((resolve) => {
      if (this.video.readyState >= 1) resolve();
      else this.video.addEventListener('loadedmetadata', resolve, { once: true });
    });
    try {
      await this.video.play();
    } catch (err) {
      if (err.name !== 'AbortError') throw err;
    }
    return { facing: this.facing, mirrored: this.mirrored };
  }

  /** Switch between the rear and the front camera. */
  flip() {
    return this.start(this.facing === 'user' ? 'environment' : 'user');
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }
}

/** Turn a getUserMedia error into text for the error card. */
export function describeCameraError(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return { title: 'Camera permission needed', message: 'Allow camera access in your browser settings, then try again.' };
    case 'NotFoundError':
    case 'OverconstrainedError':
      return { title: 'No camera found', message: 'Connect a camera or use a device that has one, then try again.' };
    case 'NotReadableError':
    case 'AbortError':
      return { title: 'Camera is busy', message: 'Another app may be using the camera. Close it and try again.' };
    case 'NotSupportedError':
      return { title: 'Camera not available', message: 'This browser cannot access the camera here. The camera needs HTTPS or localhost.' };
    default:
      return { title: 'Camera problem', message: 'Something went wrong starting the camera. Try again.' };
  }
}
