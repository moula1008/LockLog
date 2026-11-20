from pathlib import Path

import pandas as pd

from features import TARGET_SR, extract_feature_dict, load_mono

"""
dataset の音声から、特徴量CSVを作る
データを追加した後(split_clips.py→extract_features.pyの順)
"""

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_PATH = DATASET_DIR / "mfcc_features.csv"
SOURCES_PATH = DATASET_DIR / "clip_sources.csv"


def load_sources():
    """clip_sources.csv から、クリップ → 元の録音名 の対応を読む"""
    if not SOURCES_PATH.is_file():
        return {}
    table = pd.read_csv(SOURCES_PATH)
    return {(row.label, row.file): row.source for row in table.itertuples()}


def main():
    sources = load_sources()
    rows = []

    for label_dir in sorted(path for path in DATASET_DIR.iterdir() if path.is_dir()):
        label = label_dir.name
        for audio_path in sorted(label_dir.glob("*.wav")):
            try:
                audio = load_mono(audio_path)
            except Exception as error:
                print(f"読み込めないのでスキップします: {audio_path.name} ({error})")
                continue
            if audio.size == 0:
                print(f"空の音声なのでスキップします: {audio_path.name}")
                continue

            # 同じ録音から切り出したクリップは、同じ group にする（評価で一緒に扱うため）
            group = sources.get((label, audio_path.name), f"{label}/{audio_path.name}")
            row = {"label": label, "file": audio_path.name, "group": group}
            row.update(extract_feature_dict(audio))
            rows.append(row)

    if not rows:
        print("WAVファイルが見つかりませんでした。dataset/<label>/ に音声を置いてください。")
        return

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"特徴量を保存しました: {OUTPUT_PATH}")
    print(f"処理した音声: {len(table)}件 / 録音の単位(group): {table['group'].nunique()}個 / 変換後のレート: {TARGET_SR}Hz")


if __name__ == "__main__":
    main()