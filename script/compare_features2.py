import re
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from extract_features import load_sources
from features import TARGET_SR, crop_around_peak, extract_feature_dict, load_mono

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
LABELS = ("lock", "unlock", "door", "other")
# PCのマイクで録った分の、ラベルごとの最大番号（これより後の番号が、iPhoneで録った分）
PC_MAX_INDEX = {"lock": 22, "unlock": 20, "door": 18, "other": 8}
N_FFT, HOP = 1024, 256


def chunk_means(matrix, chunks=4):
    """時間方向に4分割して、それぞれの平均を並べる（時間の流れを残す）"""
    return np.concatenate([part.mean(axis=1) for part in np.array_split(matrix, chunks, axis=1)])


def build_features(audio):
    sr = TARGET_SR
    clip = crop_around_peak(audio, sr)
    peak = float(np.max(np.abs(clip)))
    norm = clip / (peak + 1e-9)
    rms = float(np.sqrt(np.mean(clip ** 2)))

    mfcc = librosa.feature.mfcc(y=norm, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP, n_mels=40)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=norm, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=24))
    envelope = librosa.feature.rms(y=norm, frame_length=N_FFT, hop_length=HOP)[0]
    envelope = envelope / (envelope.max() + 1e-9)
    centroid = librosa.feature.spectral_centroid(y=norm, sr=sr, n_fft=N_FFT, hop_length=HOP).mean()

    f0 = np.array(list(extract_feature_dict(audio).values()))
    f1 = chunk_means(mfcc)
    f2 = chunk_means(mel)
    shape = np.array([float((envelope > 0.1).mean()), float((envelope > 0.3).mean()), centroid / 1000.0])
    level = np.array([np.log10(peak + 1e-6), np.log10(rms + 1e-6), peak / (rms + 1e-9)])
    return {
        "F0 現状(MFCC平均・標準偏差)": f0,
        "F1 時間4分割MFCC": f1,
        "F2 時間4分割メル": f2,
        "F3 現状+音の形": np.concatenate([f0, shape]),
        "F4 時間4分割MFCC+音の形": np.concatenate([f1, shape]),
        "F5 F4+音量(生の大きさ)": np.concatenate([f1, shape, level]),
    }


def make_model():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))


def evaluate(table, names, title):
    y = table["label"].to_numpy()
    groups = table["group"].to_numpy()
    n_splits = int(max(2, min(5, pd.Series(groups).groupby(y).nunique().min())))
    binary = "key" in set(y)
    print(f"\n=== {title} ===")
    print("件数:", {k: int(v) for k, v in pd.Series(y).value_counts().items()}, "/ 録音の単位:", len(set(groups)), f"/ {n_splits}分割×3回")
    for name in names:
        X = np.vstack(table[name].to_list())
        accuracies, extras = [], []
        for seed in range(3):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            try:
                pred = cross_val_predict(make_model(), X, y, cv=cv, groups=groups)
            except ValueError:
                print(f"  {name:<28} 録音の単位が少なく、評価できません")
                break
            accuracies.append(accuracy_score(y, pred))
            if binary:
                hit = int(((y == "key") & (pred == "key")).sum())
                recall = hit / max(1, int((y == "key").sum()))
                precision = hit / max(1, int((pred == "key").sum()))
                extras.append((recall, precision))
            else:
                extras.append(int(((y == "lock") & (pred == "unlock")).sum() + ((y == "unlock") & (pred == "lock")).sum()))
        if len(accuracies) < 3:
            continue
        if binary:
            recall = np.mean([e[0] for e in extras])
            precision = np.mean([e[1] for e in extras])
            detail = f"鍵の音の見逃しなし率 {recall:.2f} / 鍵と判定したうちの正解率 {precision:.2f}"
        else:
            n_lu = int(np.isin(y, ["lock", "unlock"]).sum())
            detail = f"lock/unlock取り違え {np.mean(extras):.0f}/{n_lu}件"
        print(f"  {name:<28} 正解率 {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})   {detail}")


def main():
    sources = load_sources()
    rows = []
    for label in LABELS:
        for path in sorted((DATASET_DIR / label).glob("*.wav")):
            number = int(re.search(r"_(\d+)\.wav$", path.name).group(1))
            row = {
                "label": label,
                "group": sources.get((label, path.name), f"{label}/{path.name}"),
                "is_pc": number <= PC_MAX_INDEX[label],
            }
            row.update(build_features(load_mono(path)))
            rows.append(row)
    table = pd.DataFrame(rows)
    names = [column for column in table.columns if column[:1] == "F"]

    phone = table[(~table["is_pc"]) & (table["label"] != "door")].reset_index(drop=True)
    evaluate(phone, names, "iPhone録音だけ（lock / unlock / other の3クラス・偶然は約33%）")
    evaluate(table, names, "全データ（4クラス・偶然は25%）")

    # 2クラス: lock と unlock を「key（鍵の操作）」にまとめる
    binary = table.copy()
    binary["label"] = binary["label"].map(lambda name: "key" if name in ("lock", "unlock") else "other")
    phone_binary = binary[(~binary["is_pc"]) & (table["label"] != "door")].reset_index(drop=True)
    evaluate(phone_binary, names, "【2クラス】iPhone録音だけ（key / other）")
    evaluate(binary, names, "【2クラス】全データ（key / その他[door+other]）")


if __name__ == "__main__":
    main()
