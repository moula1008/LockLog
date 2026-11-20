from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

"""
データの件数と整合性を確認する
今データがどうなっているのか確認する
"""


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
FEATURES_PATH = DATASET_DIR / "mfcc_features.csv"
EXPECTED_LABELS = ("lock", "unlock", "door", "other")


def main():
    if not DATASET_DIR.is_dir():
        print(f"datasetフォルダが見つかりません: {DATASET_DIR}")
        return

    wav_counts = Counter()
    wav_files = []
    bad_files = []
    unexpected_labels = []

    for label_dir in sorted(path for path in DATASET_DIR.iterdir() if path.is_dir()):
        label = label_dir.name
        if label not in EXPECTED_LABELS:
            unexpected_labels.append(label)

        for audio_path in sorted(label_dir.glob("*.wav")):
            wav_counts[label] += 1
            wav_files.append((label, audio_path.name))
            try:
                info = sf.info(audio_path)
                if info.frames == 0 or info.samplerate <= 0:
                    bad_files.append(f"{label}/{audio_path.name}: 音声データが空か不正です")
            except Exception as error:
                bad_files.append(f"{label}/{audio_path.name}: 読み込めません ({error})")

    print("=== LockLog データセット確認 ===")
    print("WAVファイル数:", len(wav_files))
    print("\nラベル別の件数:")
    for label in EXPECTED_LABELS:
        count = wav_counts[label]
        state = "あり" if count else "まだありません"
        print(f"  {label}: {count}件 ({state})")

    missing_labels = [label for label in EXPECTED_LABELS if wav_counts[label] == 0]
    if missing_labels:
        print("\n未収集のラベル:", ", ".join(missing_labels))
        print("全ラベルが揃うまでは、4種類の分類モデルを学習できません。")
    else:
        print("\n4種類すべてのラベルにWAVがあります。")

    if unexpected_labels:
        print("\n想定外のフォルダ:", ", ".join(unexpected_labels))
        print("必要ならEXPECTED_LABELSとラベル名を揃えてください。")

    if bad_files:
        print("\n読み込みに問題があるファイル:")
        for message in bad_files:
            print("  -", message)
    else:
        print("\nWAVファイルの読み込み確認: 問題なし")

    if FEATURES_PATH.is_file():
        try:
            features = pd.read_csv(FEATURES_PATH)
            csv_counts = Counter(features["label"].astype(str)) if "label" in features else Counter()
            print(f"\n特徴量CSV: {len(features)}行")
            if "label" not in features or "file" not in features:
                print("CSVにlabel列またはfile列がありません。特徴量抽出を再実行してください。")
            else:
                csv_files = set(zip(features["label"].astype(str), features["file"].astype(str)))
                wav_file_set = set(wav_files)
                if csv_files != wav_file_set:
                    print("CSVとWAVの内容が一致しません。特徴量抽出を再実行してください。")
                    print("  CSVにないWAV:", len(wav_file_set - csv_files), "件")
                    print("  WAVにないCSV行:", len(csv_files - wav_file_set), "件")
                else:
                    print("CSVとWAVの対応: 一致")
                print("CSVのラベル別件数:")
                for label in EXPECTED_LABELS:
                    print(f"  {label}: {csv_counts[label]}件")
        except Exception as error:
            print(f"\n特徴量CSVを読み込めません: {error}")
    else:
        print("\n特徴量CSVがありません。先にextract_features.pyを実行してください。")


if __name__ == "__main__":
    main()
