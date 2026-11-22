import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import POST_SEC, PRE_SEC, TARGET_SR, TOP_DB


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "dataset" / "mfcc_features.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "locklog_model.joblib"
MODEL_JSON_PATH = BASE_DIR / "app" / "static" / "locklog_model.json"   # ブラウザで判定するためのモデル
EXPECTED_LABELS = ("lock", "unlock", "door", "other")
KEY_LABELS = ("lock", "unlock")   # この2つを「鍵の操作(key)」にまとめる
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def make_model():
    svm = SVC(kernel="rbf", class_weight="balanced")
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(svm, method="sigmoid", cv=3, ensemble=False),
    )


def export_json(model, feature_columns, path):
    """ブラウザ(JavaScript)で判定するために、モデルの中身をJSONで書き出す"""
    scaler = model.named_steps["standardscaler"]
    calibrated = model.named_steps["calibratedclassifiercv"]
    inner = calibrated.calibrated_classifiers_[0]
    svm = getattr(inner, "estimator", None) or inner.base_estimator
    sigmoid = inner.calibrators[0]
    assert list(calibrated.classes_) == ["key", "other"], "クラスの順番が想定と違います"
    payload = {
        "feature_names": list(feature_columns),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "gamma": float(svm._gamma),
        "intercept": float(svm.intercept_[0]),
        "dual_coef": np.round(svm.dual_coef_[0], 10).tolist(),
        "support_vectors": np.round(svm.support_vectors_, 8).tolist(),
        "sigmoid_a": float(sigmoid.a_),
        "sigmoid_b": float(sigmoid.b_),
        "positive_class": "key",
        "top_db": TOP_DB,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main():
    if not FEATURES_PATH.is_file():
        print(f"特徴量CSVがありません: {FEATURES_PATH}")
        print("先に python script/extract_features.py を実行してください。")
        return

    data = pd.read_csv(FEATURES_PATH)
    if not {"label", "file", "group"}.issubset(data.columns):
        print("CSVにlabel/file/group列がありません。extract_features.py を再実行してください。")
        return

    feature_columns = [c for c in data.columns if c not in ("label", "file", "group")]
    if not feature_columns or data[feature_columns].isna().any().any():
        print("特徴量の列がない、または空欄があります。extract_features.py を再実行してください。")
        return

    data["label"] = data["label"].astype(str)
    unexpected = sorted(set(data["label"]) - set(EXPECTED_LABELS))
    if unexpected:
        print("想定外のラベルがあります:", ", ".join(unexpected))
        return

    counts = data["label"].value_counts()
    print("=== LockLog 分類モデル（鍵の操作 or それ以外）===")
    print("特徴量:", len(feature_columns), "個 / 録音の単位(group):", data["group"].nunique(), "個")
    for label in EXPECTED_LABELS:
        mark = "→ key" if label in KEY_LABELS else "→ other"
        print(f"  {label}: {int(counts.get(label, 0))}件 {mark}")
    if any(counts.get(label, 0) < N_SPLITS for label in EXPECTED_LABELS):
        print(f"各ラベルに最低{N_SPLITS}件必要です。音声を追加してから実行してください。")
        return

    X = data[feature_columns].astype(float)
    original = data["label"].to_numpy()
    y = np.where(np.isin(original, KEY_LABELS), "key", "other")
    groups = data["group"].to_numpy()
    classes = np.array(["key", "other"])
    key_index = 0

    accuracies = []
    for repeat in range(N_REPEATS):
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat)
        probabilities = cross_val_predict(make_model(), X, y, cv=cv, groups=groups, method="predict_proba")
        accuracies.append(accuracy_score(y, classes[probabilities.argmax(axis=1)]))
        if repeat == 0:
            first = probabilities
    predictions = classes[first.argmax(axis=1)]
    p_key = first[:, key_index]

    print(f"\n=== 交差検証（録音ごとに分割・{N_SPLITS}分割×{N_REPEATS}回） ===")
    print(f"正解率: {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})")
    print("\nクラス別の評価（1回目の分割）:")
    print(classification_report(y, predictions, labels=list(classes), zero_division=0))
    print("混同行列（行=正解、列=予測） 順番: key / other")
    print(confusion_matrix(y, predictions, labels=list(classes)))

    print("\n元のラベルごとの「key」と判定された割合（1回目の分割）:")
    for label in EXPECTED_LABELS:
        mask = original == label
        print(f"  {label:<7} {(predictions[mask] == 'key').mean():6.1%}  ({int(mask.sum())}件)")

    print("\n=== しきい値（keyの確率）と、見逃し・誤検出（1回目の分割） ===")
    print("しきい値 | 鍵の音を検出できた割合 | doorを誤検出 | otherを誤検出")
    for threshold in THRESHOLDS:
        detected = p_key >= threshold
        print(
            f"  {threshold:.1f}    |        {detected[y == 'key'].mean():6.1%}         |"
            f"   {detected[original == 'door'].mean():6.1%}   |   {detected[original == 'other'].mean():6.1%}"
        )

    web = np.array([str(name).startswith("locklog_clips_") for name in groups])
    if web.any():
        print(f"\n=== ブラウザで集めたクリップだけの結果（{int(web.sum())}件・録音の単位 {len(set(groups[web]))}個・しきい値0.5） ===")
        print("鍵の音(lock/unlock)は「keyと判定された割合」が高いほど良く、door/otherは低いほど良い")
        for label in EXPECTED_LABELS:
            mask = web & (original == label)
            if mask.any():
                print(f"  {label:<7} {int(mask.sum()):3d}件   keyと判定: {(p_key[mask] >= 0.5).mean():6.1%}")

    wrong = np.flatnonzero(predictions != y)
    print(f"\n誤分類したファイル（1回目の分割）: {wrong.size}件")
    for index in wrong:
        print(f"  {original[index]}/{data.loc[index, 'file']}: 予測={predictions[index]} (keyの確率 {p_key[index]:.2f})")

    model = make_model()
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "feature_columns": feature_columns,
            "labels": list(EXPECTED_LABELS),
            "classes": [str(name) for name in model.classes_],
            "mode": "key",
            "sample_rate": TARGET_SR,
            "crop_seconds": (PRE_SEC, POST_SEC),
        },
        MODEL_PATH,
    )
    export_json(model, feature_columns, MODEL_JSON_PATH)
    print(f"\n全{len(data)}件で学習したモデルを保存しました: {MODEL_PATH}")
    print(f"ブラウザ用のモデルを書き出しました: {MODEL_JSON_PATH}（{MODEL_JSON_PATH.stat().st_size // 1024} KB）")


if __name__ == "__main__":
    main()