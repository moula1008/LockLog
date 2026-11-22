/* LockLog: 端末の中で判定するための、特徴量計算とSVMの予測。
   script/features.py と、同じ式で書いてある（結果が一致することを、数値で確認している）。 */
(function (root) {
  "use strict";
  var TARGET_SR = 32000, N_MFCC = 13, N_MELS = 40, N_FFT = 1024, HOP = 256;
  var PRE_SEC = 0.2, POST_SEC = 0.8, HALF_WIDTH = 16;
  var NBINS = N_FFT / 2 + 1, i, j, k;

  var WINDOW = new Float64Array(N_FFT);
  for (i = 0; i < N_FFT; i++) WINDOW[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N_FFT);

  var FREQS = new Float64Array(NBINS);
  for (k = 0; k < NBINS; k++) FREQS[k] = k * TARGET_SR / N_FFT;

  function hzToMel(f) { return 2595 * Math.log10(1 + f / 700); }
  function melToHz(m) { return 700 * (Math.pow(10, m / 2595) - 1); }

  // メルフィルタ（三角形・40本）
  var BANK = [];
  (function () {
    var start = hzToMel(0), stop = hzToMel(TARGET_SR / 2), step = (stop - start) / (N_MELS + 1), hz = [];
    for (i = 0; i < N_MELS + 2; i++) hz.push(melToHz(i === N_MELS + 1 ? stop : start + i * step));
    for (j = 0; j < N_MELS; j++) {
      var row = new Float64Array(NBINS), low = hz[j], mid = hz[j + 1], high = hz[j + 2];
      for (k = 0; k < NBINS; k++) row[k] = Math.max(0, Math.min((FREQS[k] - low) / (mid - low), (high - FREQS[k]) / (high - mid)));
      BANK.push(row);
    }
  })();

  // DCT（13行×40列）
  var DCT = [];
  for (k = 0; k < N_MFCC; k++) {
    var drow = new Float64Array(N_MELS);
    for (i = 0; i < N_MELS; i++) {
      drow[i] = Math.cos(Math.PI * k * (i + 0.5) / N_MELS) * Math.sqrt(2 / N_MELS);
      if (k === 0) drow[i] *= Math.sqrt(0.5);
    }
    DCT.push(drow);
  }

  // FFT（1024点・その場で計算）
  var BITREV = new Uint16Array(N_FFT), COS = new Float64Array(N_FFT / 2), SIN = new Float64Array(N_FFT / 2);
  (function () {
    var bits = Math.round(Math.log2(N_FFT));
    for (i = 0; i < N_FFT; i++) {
      var r = 0;
      for (var b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      BITREV[i] = r;
    }
    for (i = 0; i < N_FFT / 2; i++) { COS[i] = Math.cos(2 * Math.PI * i / N_FFT); SIN[i] = -Math.sin(2 * Math.PI * i / N_FFT); }
  })();
  function fft(re, im) {
    var n = N_FFT, t;
    for (var a = 0; a < n; a++) {
      var r = BITREV[a];
      if (r > a) { t = re[a]; re[a] = re[r]; re[r] = t; t = im[a]; im[a] = im[r]; im[r] = t; }
    }
    for (var size = 2; size <= n; size <<= 1) {
      var half = size >> 1, stride = n / size;
      for (var s = 0; s < n; s += size) {
        for (var q = 0; q < half; q++) {
          var wr = COS[q * stride], wi = SIN[q * stride], p = s + q, o = p + half;
          var tr = re[o] * wr - im[o] * wi, ti = re[o] * wi + im[o] * wr;
          re[o] = re[p] - tr; im[o] = im[p] - ti; re[p] += tr; im[p] += ti;
        }
      }
    }
  }

  function gcd(a, b) { while (b) { var t = a % b; a = b; b = t; } return a; }

  // 窓付きsincによるリサンプリング（重みは、位相ごとに1度だけ計算して使い回す）
  function resample(x, origSr, targetSr) {
    if (origSr === targetSr) return Float64Array.from(x);
    var scale = Math.min(1, targetSr / origSr), half = HALF_WIDTH / scale, reach = Math.ceil(half);
    var nOut = Math.floor(x.length * targetSr / origSr);
    if (!(Number.isInteger(origSr) && Number.isInteger(targetSr))) return resampleSlow(x, origSr, targetSr, nOut);
    var g = gcd(origSr, targetSr), inStep = origSr / g, period = targetSr / g, taps = 2 * reach + 1;
    var table = new Float64Array(period * taps), out = new Float64Array(nOut), r, o, n, q;
    for (r = 0; r < period; r++) {
      var frac = ((r * inStep) % period) / period, wsum = 0, row = r * taps;
      for (o = 0; o < taps; o++) {
        var dd = frac - (o - reach);
        if (Math.abs(dd) >= half) continue;
        var uu = dd * scale;
        var ww = (uu === 0 ? 1 : Math.sin(Math.PI * uu) / (Math.PI * uu)) * 0.5 * (1 + Math.cos(Math.PI * dd / half));
        table[row + o] = ww; wsum += ww;
      }
      for (o = 0; o < taps; o++) table[row + o] /= wsum;
    }
    for (n = 0; n < nOut; n++) {
      var base = Math.floor(n * inStep / period), row2 = (n % period) * taps, acc = 0;
      for (q = 0; q < taps; q++) {
        var ix = base + q - reach;
        if (ix >= 0 && ix < x.length) acc += x[ix] * table[row2 + q];
      }
      out[n] = acc;
    }
    return out;
  }

  // 整数でないレート用の、遅いが一般的な版
  function resampleSlow(x, origSr, targetSr, nOut) {
    var n, idx, d, u, w;
    var scale = Math.min(1, targetSr / origSr), half = HALF_WIDTH / scale, reach = Math.ceil(half);
    var out = new Float64Array(nOut), step = origSr / targetSr;
    for (n = 0; n < nOut; n++) {
      var pos = n * step, base = Math.floor(pos), wsum = 0, acc = 0;
      for (var o = -reach; o <= reach; o++) {
        idx = base + o; d = pos - idx;
        if (Math.abs(d) >= half) continue;
        u = d * scale;
        w = (u === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u)) * 0.5 * (1 + Math.cos(Math.PI * d / half));
        wsum += w;
        if (idx >= 0 && idx < x.length) acc += x[idx] * w;
      }
      out[n] = acc / wsum;
    }
    return out;
  }

  // 一番大きい音の付近を、1秒だけ切り出す
  function cropAroundPeak(x) {
    var length = Math.round((PRE_SEC + POST_SEC) * TARGET_SR), pre = Math.round(PRE_SEC * TARGET_SR);
    var peak = 0, best = -1;
    for (var n = 0; n < x.length; n++) { var a = Math.abs(x[n]); if (a > best) { best = a; peak = n; } }
    var start = Math.max(0, Math.min(peak - pre, x.length - length));
    var clip = new Float64Array(length);
    for (var m = 0; m < length && start + m < x.length; m++) clip[m] = x[start + m];
    return clip;
  }

  // 音声（Float32Array など）から、55個の特徴量を作る
  function extractFeatures(audio, sr, topDb) {
    var x = cropAroundPeak(resample(audio, sr, TARGET_SR)), n, t, c;
    var top = 0;
    for (n = 0; n < x.length; n++) top = Math.max(top, Math.abs(x[n]));
    for (n = 0; n < x.length; n++) x[n] = x[n] / (top + 1e-9);

    var nFrames = 1 + Math.floor(x.length / HOP), pad = N_FFT / 2;
    var padded = new Float64Array(x.length + N_FFT);
    padded.set(x, pad);

    var re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), power = new Float64Array(NBINS);
    var mfcc = [], logs = [], maxLog = -Infinity, rms = new Float64Array(nFrames), centroid = new Float64Array(nFrames);
    for (t = 0; t < nFrames; t++) {
      var sq = 0;
      for (i = 0; i < N_FFT; i++) { var v = padded[t * HOP + i]; sq += v * v; re[i] = v * WINDOW[i]; im[i] = 0; }
      rms[t] = Math.sqrt(sq / N_FFT);
      fft(re, im);
      var total = 0, weighted = 0;
      for (k = 0; k < NBINS; k++) {
        power[k] = re[k] * re[k] + im[k] * im[k];
        var mag = Math.sqrt(power[k]); total += mag; weighted += mag * FREQS[k];
      }
      centroid[t] = total < 1e-12 ? 0 : weighted / total;
      var lm = new Float64Array(N_MELS);
      for (j = 0; j < N_MELS; j++) {
        var e = 0, row = BANK[j];
        for (k = 0; k < NBINS; k++) e += row[k] * power[k];
        lm[j] = 10 * Math.log10(Math.max(e, 1e-10));
        if (lm[j] > maxLog) maxLog = lm[j];
      }
      logs.push(lm);
    }
    var useFloor = topDb !== null && topDb !== undefined, floor = maxLog - (useFloor ? topDb : 0);
    for (t = 0; t < nFrames; t++) {
      var lmt = logs[t], coef = new Float64Array(N_MFCC);
      if (useFloor) for (j = 0; j < N_MELS; j++) if (lmt[j] < floor) lmt[j] = floor;
      for (c = 0; c < N_MFCC; c++) { var s2 = 0, drow = DCT[c]; for (j = 0; j < N_MELS; j++) s2 += drow[j] * lmt[j]; coef[c] = s2; }
      mfcc.push(coef);
    }

    var feats = new Float64Array(4 * N_MFCC + 3);
    var baseSize = Math.floor(nFrames / 4), extra = nFrames % 4, from = 0;
    for (var seg = 0; seg < 4; seg++) {
      var size = baseSize + (seg < extra ? 1 : 0);
      for (c = 0; c < N_MFCC; c++) {
        var sum = 0;
        for (t = from; t < from + size; t++) sum += mfcc[t][c];
        feats[seg * N_MFCC + c] = sum / size;
      }
      from += size;
    }
    var maxRms = 0, a10 = 0, a30 = 0, cen = 0;
    for (t = 0; t < nFrames; t++) maxRms = Math.max(maxRms, rms[t]);
    for (t = 0; t < nFrames; t++) {
      var env = rms[t] / (maxRms + 1e-9);
      if (env > 0.1) a10++;
      if (env > 0.3) a30++;
      cen += centroid[t];
    }
    feats[4 * N_MFCC] = a10 / nFrames;
    feats[4 * N_MFCC + 1] = a30 / nFrames;
    feats[4 * N_MFCC + 2] = cen / nFrames / 1000;
    return feats;
  }

  // 学習済みモデル（JSON）で、「鍵の音」である確率を返す
  function predictKey(model, feats) {
    var n = feats.length, z = new Float64Array(n), s, d2, t;
    for (i = 0; i < n; i++) z[i] = (feats[i] - model.mean[i]) / model.scale[i];
    var df = model.intercept;
    for (s = 0; s < model.support_vectors.length; s++) {
      var sv = model.support_vectors[s]; d2 = 0;
      for (i = 0; i < n; i++) { t = z[i] - sv[i]; d2 += t * t; }
      df += model.dual_coef[s] * Math.exp(-model.gamma * d2);
    }
    var pOther = 1 / (1 + Math.exp(model.sigmoid_a * df + model.sigmoid_b));
    return 1 - pOther;
  }

  var api = {
    TARGET_SR: TARGET_SR,
    resample: resample,
    extractFeatures: extractFeatures,
    predictKey: predictKey,
    predict: function (model, audio, sr) { return predictKey(model, extractFeatures(audio, sr, model.top_db)); }
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.LockLog = api;
})(typeof self !== "undefined" ? self : this);