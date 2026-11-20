import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 32000   # 学習・判定の両方でこのレートにそろえる（比較実験の結果から暫定）
N_MFCC = 13
PRE_SEC = 0.2       # ピークの何秒前から切り出すか
POST_SEC = 0.8      # ピークの何秒後まで切り出すか


def load_mono(path, target_sr=TARGET_SR):
    """WAVを読み込み、モノラル・指定のレートにそろえる"""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def crop_around_peak(audio, sr=TARGET_SR):
    """一番大きい音の位置を中心に、1秒だけ切り出す"""
    length = int((PRE_SEC + POST_SEC) * sr)
    peak = int(np.argmax(np.abs(audio)))
    start = max(0, min(peak - int(PRE_SEC * sr), len(audio) - length))
    clip = audio[start:start + length]
    if len(clip) < length:
        clip = np.pad(clip, (0, length - len(clip)))
    return clip


def _chunk_means(matrix, chunks=4):
    """時間方向に4分割して、それぞれの平均を並べる（時間の流れを残す）"""
    return [part.mean(axis=1) for part in np.array_split(matrix, chunks, axis=1)]


def extract_feature_dict(audio, sr=TARGET_SR, crop=True):
    """音量をそろえ、時間4分割のMFCC(52個)と、音の形(3個)を返す"""
    if crop:
        audio = crop_around_peak(audio, sr)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)  # 音量の違いを打ち消す
    n_fft = 2 ** int(np.round(np.log2(sr * 0.025)))  # 約25msの窓にそろえる
    hop = n_fft // 4

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=n_fft, hop_length=hop, n_mels=40)
    envelope = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop)[0]
    envelope = envelope / (envelope.max() + 1e-9)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop).mean()

    features = {}
    for segment, means in enumerate(_chunk_means(mfcc)):
        for index in range(N_MFCC):
            features[f"mfcc_s{segment}_{index:02d}"] = float(means[index])
    features["shape_active10"] = float((envelope > 0.1).mean())   # 音が続く割合（小さい音も含む）
    features["shape_active30"] = float((envelope > 0.3).mean())   # 音が続く割合（大きい音だけ）
    features["shape_centroid"] = float(centroid / 1000.0)         # 音の高さの重心
    return features