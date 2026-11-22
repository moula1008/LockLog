"""特徴量の計算（numpyだけ）。app/static/locklog_core.js と、同じ式で書いてある。

流れ: 16kHz系→32kHzにそろえる → 最大音の付近1秒を切り出す → 音量をそろえる
      → STFT → メル → MFCC(13) → 時間4分割の平均(52個) + 音の形(3個) = 55個
"""
import numpy as np
import soundfile as sf

TARGET_SR = 32000
N_MFCC = 13
N_MELS = 40
N_FFT = 1024
HOP = 256
TOP_DB = 80.0     # メルのdB値を、最大値からこの幅より下には下げない（None にすると無効）
PRE_SEC = 0.2       # ピークの何秒前から切り出すか
POST_SEC = 0.8      # ピークの何秒後まで切り出すか
HALF_WIDTH = 16     # リサンプリングの、sinc関数の片側の長さ（ゼロ交差の数）
CHUNK = 8192        # リサンプリングを、この数ずつ計算する（メモリ節約）


def resample(audio, orig_sr, target_sr=TARGET_SR):
    """窓付きsincによるリサンプリング（JavaScript側と同じ式）"""
    audio = np.asarray(audio, dtype=np.float64)
    if orig_sr == target_sr:
        return audio
    scale = min(1.0, target_sr / orig_sr)
    half = HALF_WIDTH / scale
    reach = int(np.ceil(half))
    offsets = np.arange(-reach, reach + 1)
    n_out = (len(audio) * target_sr) // orig_sr
    step = orig_sr / target_sr
    out = np.empty(n_out)
    for first in range(0, n_out, CHUNK):
        n = np.arange(first, min(first + CHUNK, n_out))
        position = n * step
        index = np.floor(position).astype(np.int64)[:, None] + offsets[None, :]
        distance = position[:, None] - index
        u = distance * scale
        safe = np.where(u == 0, 1.0, u)
        weight = np.where(u == 0, 1.0, np.sin(np.pi * safe) / (np.pi * safe))
        weight = weight * 0.5 * (1.0 + np.cos(np.pi * distance / half))
        weight = np.where(np.abs(distance) < half, weight, 0.0)
        valid = (index >= 0) & (index < len(audio))
        samples = np.where(valid, audio[np.clip(index, 0, len(audio) - 1)], 0.0)
        out[first:first + len(n)] = (samples * weight).sum(axis=1) / weight.sum(axis=1)
    return out


def load_mono(path, target_sr=TARGET_SR):
    """WAVを読み込み、モノラル・指定のレートにそろえる"""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return resample(audio, sr, target_sr)


def crop_around_peak(audio, sr=TARGET_SR):
    """一番大きい音の位置を中心に、1秒だけ切り出す"""
    length = int(round((PRE_SEC + POST_SEC) * sr))
    pre = int(round(PRE_SEC * sr))
    peak = int(np.argmax(np.abs(audio)))
    start = max(0, min(peak - pre, len(audio) - length))
    clip = audio[start:start + length]
    if len(clip) < length:
        clip = np.pad(clip, (0, length - len(clip)))
    return clip


def _mel_filterbank():
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(0.0), hz_to_mel(TARGET_SR / 2.0), N_MELS + 2)
    hz = mel_to_hz(mels)
    freqs = np.arange(N_FFT // 2 + 1) * TARGET_SR / N_FFT
    bank = np.zeros((N_MELS, N_FFT // 2 + 1))
    for j in range(N_MELS):
        low, mid, high = hz[j], hz[j + 1], hz[j + 2]
        bank[j] = np.maximum(0.0, np.minimum((freqs - low) / (mid - low), (high - freqs) / (high - mid)))
    return bank, freqs


def _dct_matrix():
    n = np.arange(N_MELS)
    k = np.arange(N_MFCC)[:, None]
    matrix = np.cos(np.pi * k * (n + 0.5) / N_MELS) * np.sqrt(2.0 / N_MELS)
    matrix[0] *= np.sqrt(0.5)
    return matrix


_WINDOW = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT)
_BANK, _FREQS = _mel_filterbank()
_DCT = _dct_matrix()


def extract_feature_dict(audio, sr=TARGET_SR, crop=True):
    """音声から、55個の特徴量を作る"""
    audio = resample(audio, sr, TARGET_SR)
    if crop:
        audio = crop_around_peak(audio, TARGET_SR)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)           # 音量の違いを打ち消す

    n_frames = 1 + len(audio) // HOP
    padded = np.concatenate([np.zeros(N_FFT // 2), audio, np.zeros(N_FFT // 2)])
    index = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = padded[index]
    spectrum = np.fft.rfft(frames * _WINDOW, axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2

    log_mel = 10.0 * np.log10(np.maximum(power @ _BANK.T, 1e-10))
    if TOP_DB is not None:
        log_mel = np.maximum(log_mel, log_mel.max() - TOP_DB)
    mfcc = log_mel @ _DCT.T                                  # (フレーム数, 13)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    envelope = rms / (rms.max() + 1e-9)
    magnitude = np.sqrt(power)
    total = magnitude.sum(axis=1)
    centroid = np.where(total < 1e-12, 0.0, (magnitude * _FREQS).sum(axis=1) / np.where(total < 1e-12, 1.0, total))

    features = {}
    for segment, part in enumerate(np.array_split(mfcc, 4, axis=0)):
        means = part.mean(axis=0)
        for i in range(N_MFCC):
            features[f"mfcc_s{segment}_{i:02d}"] = float(means[i])
    features["shape_active10"] = float((envelope > 0.1).mean())   # 音が続く割合（小さい音も含む）
    features["shape_active30"] = float((envelope > 0.3).mean())   # 音が続く割合（大きい音だけ）
    features["shape_centroid"] = float(centroid.mean() / 1000.0)  # 音の高さの重心
    return features