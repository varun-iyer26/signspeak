// Teachable k-NN over the 126-value feature vectors, persisted in localStorage.
import { FEATURE_LENGTH } from './features.js';

const clamp01 = (x) => Math.min(1, Math.max(0, x));

export class KnnClassifier {
  /**
   * @param {object} opts
   * @param {number} opts.k neighbours that vote
   * @param {number} opts.threshold max Euclidean distance (in palm-size units) for a match
   * @param {string} opts.storageKey localStorage key
   * @param {number} opts.maxPerLabel oldest samples are dropped beyond this
   */
  constructor({ k = 5, threshold = 1.5, storageKey = 'signspeak.signs.v1', maxPerLabel = 300 } = {}) {
    this.k = k;
    this.threshold = threshold;
    this.storageKey = storageKey;
    this.maxPerLabel = maxPerLabel;
    this.data = new Map(); // label -> Float32Array[]
  }

  /** [{label, count}] sorted by label. */
  labels() {
    return [...this.data.entries()].map(([label, list]) => ({ label, count: list.length })).sort((a, b) => a.label.localeCompare(b.label));
  }

  get size() {
    let n = 0;
    for (const list of this.data.values()) n += list.length;
    return n;
  }

  addMany(label, vectors) {
    if (!vectors.length) return;
    const list = this.data.get(label) || [];
    for (const v of vectors) list.push(Float32Array.from(v));
    if (list.length > this.maxPerLabel) list.splice(0, list.length - this.maxPerLabel);
    this.data.set(label, list);
  }

  deleteLabel(label) {
    this.data.delete(label);
  }

  /**
   * Classify a vector. Returns {label, confidence, distance} or null when
   * nothing is stored or the nearest sample is farther than the threshold.
   */
  predict(vector) {
    if (!vector || !this.size) return null;
    const k = Math.min(this.k, this.size);
    const best = []; // k nearest, ascending by distance
    for (const [label, list] of this.data) {
      for (const sample of list) {
        let sum = 0;
        for (let i = 0; i < FEATURE_LENGTH; i++) {
          const d = sample[i] - vector[i];
          sum += d * d;
        }
        if (best.length === k && sum >= best[k - 1].sq) continue;
        let pos = best.length;
        while (pos > 0 && best[pos - 1].sq > sum) pos--;
        best.splice(pos, 0, { sq: sum, label });
        if (best.length > k) best.pop();
      }
    }
    if (Math.sqrt(best[0].sq) > this.threshold) return null;

    // Inverse-distance weighted vote among the neighbours.
    const votes = new Map();
    let total = 0;
    for (const n of best) {
      const w = 1 / (Math.sqrt(n.sq) + 0.05);
      votes.set(n.label, (votes.get(n.label) || 0) + w);
      total += w;
    }
    let label = null;
    let top = -1;
    for (const [l, w] of votes) {
      if (w > top) {
        top = w;
        label = l;
      }
    }
    const winners = best.filter((n) => n.label === label);
    const meanDist = winners.reduce((s, n) => s + Math.sqrt(n.sq), 0) / winners.length;
    const closeness = clamp01(1 - meanDist / this.threshold);
    return { label, confidence: (top / total) * (0.4 + 0.6 * closeness), distance: meanDist };
  }

  // ---- persistence -------------------------------------------------------

  toJSON() {
    const samples = {};
    for (const [label, list] of this.data) {
      samples[label] = list.map((v) => Array.from(v, (x) => Math.round(x * 1e4) / 1e4));
    }
    return { app: 'signspeak', version: 1, featureLength: FEATURE_LENGTH, samples };
  }

  exportJSON() {
    return JSON.stringify(this.toJSON(), null, 1);
  }

  /** Merge a dataset from JSON text. Returns {labels, samples}. Throws on invalid data. */
  importJSON(text) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new Error('That file is not valid JSON.');
    }
    const samples = parsed?.samples;
    if (!samples || typeof samples !== 'object' || Array.isArray(samples)) throw new Error('No sign data found in that file.');
    const staged = [];
    for (const [label, list] of Object.entries(samples)) {
      if (!label.trim() || !Array.isArray(list)) throw new Error(`Bad entry for "${label}".`);
      for (const v of list) {
        if (!Array.isArray(v) || v.length !== FEATURE_LENGTH || !v.every(Number.isFinite)) {
          throw new Error(`Bad sample in "${label}".`);
        }
      }
      staged.push([label.trim().slice(0, 40), list]);
    }
    let count = 0;
    for (const [label, list] of staged) {
      this.addMany(label, list);
      count += list.length;
    }
    return { labels: staged.length, samples: count };
  }

  /** Persist to localStorage. Returns false if storage is unavailable or full. */
  save() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.toJSON()));
      return true;
    } catch {
      return false;
    }
  }

  load() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (raw) this.importJSON(raw);
    } catch {
      this.data.clear(); // corrupted or blocked storage: start empty
    }
  }
}
