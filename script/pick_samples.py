import re
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import extract_feature_dict, load_mono


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "models" / "locklog_model.joblib"
OUT_DIR = BASE_DIR / "app" / "static" / "samples"

# PCのマイクで録った分の、ラベルごとの最大番号（これより後の番号が、iPhoneで録った分）
PC_MAX_INDEX = {"lock": 22, "unlock": 20, "door": 18, "other": 8}
# 鍵の音(lock/unlock)は「keyの確率が高い」もの、door/other は「低い」ものを選ぶ
WANT_KEY = {"lock": True, "unlock": True, "door": False, "other": False}


def main():
    if not MODEL_PATH.is_file():
        print("モデルがありません。先に train_model.py を実行してください。")
        return
    bundle = joblib.load(MODEL_PATH)
    key_index = bundle["classes"].index("key")
    columns = bundle["feature_columns"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, want_key in WANT_KEY.items():
        scored = []
        for path in sorted((DATASET_DIR / label).glob("*.wav")):
            number = int(re.search(r"_(\d+)\.wav$", path.name).group(1))
            if number <= PC_MAX_INDEX[label]:
                continue   # PCマイクの録音は、デモに使わない
            audio = load_mono(path, target_sr=bundle["sample_rate"])
            if float(np.max(np.abs(audio))) >= 0.99:
                continue   # 音割れの疑いがあるものは避ける
            features = extract_feature_dict(audio, sr=bundle["sample_rate"])
            X = pd.DataFrame([[features[name] for name in columns]], columns=columns)
            p_key = float(bundle["pipeline"].predict_proba(X)[0][key_index])
            scored.append((p_key if want_key else 1.0 - p_key, p_key, path))

        scored.sort(key=lambda item: item[0], reverse=True)
        print(f"\n[{label}]  候補（上位3件）")
        for score, p_key, path in scored[:3]:
            print(f"  {path.name:<18} keyの確率 {p_key:.2f}")
        if scored:
            shutil.copyfile(scored[0][2], OUT_DIR / f"{label}.wav")
            print(f"  → {scored[0][2].name} を {label}.wav としてコピーしました")
        else:
            print("  候補がありません（iPhoneで録ったクリップが必要です）")

    print(f"\n保存先: {OUT_DIR}")
    print("※ 必ず再生して、話し声など、公開したくない音が入っていないか確認してください。")


if __name__ == "__main__":
    main()