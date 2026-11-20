import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import extract_feature_dict, load_mono

"""
音声1ファイルを判定する
動作確認用
"""

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "locklog_model.joblib"


def main():
    parser = argparse.ArgumentParser(description="LockLogの学習済みモデルでWAVファイルを分類します。")
    parser.add_argument("audio_path", help="分類するWAVファイルのパス")
    args = parser.parse_args()

    audio_path = Path(args.audio_path).expanduser()
    if not audio_path.is_absolute():
        audio_path = Path.cwd() / audio_path
    audio_path = audio_path.resolve()

    if not MODEL_PATH.is_file():
        print(f"学習済みモデルが見つかりません: {MODEL_PATH}")
        print("先に python script/train_model.py を実行してください。")
        return
    if not audio_path.is_file():
        print(f"音声ファイルが見つかりません: {audio_path}")
        return

    try:
        bundle = joblib.load(MODEL_PATH)
        sample_rate = bundle["sample_rate"]
        audio = load_mono(audio_path, target_sr=sample_rate)
    except Exception as error:
        print(f"モデルまたは音声ファイルを読み込めません: {error}")
        return
    if audio.size == 0:
        print("音声データが空です。")
        return

    features = extract_feature_dict(audio, sr=sample_rate)
    columns = bundle["feature_columns"]
    if any(name not in features for name in columns):
        print("モデルと特徴量の形式が一致しません。モデルを作り直してください。")
        return

    X = pd.DataFrame([[features[name] for name in columns]], columns=columns)
    probabilities = bundle["pipeline"].predict_proba(X)[0]
    classes = bundle["classes"]
    order = np.argsort(probabilities)[::-1]

    print("予測したファイル:", audio_path.name)
    print(f"予測ラベル: {classes[order[0]]} (確信度 {probabilities[order[0]]:.2f})")
    print("内訳:", " / ".join(f"{classes[i]} {probabilities[i]:.2f}" for i in order))


if __name__ == "__main__":
    main()