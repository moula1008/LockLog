import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "dataset" / "mfcc_features.csv"
LABELS = ("lock", "unlock", "door", "other")

# PCのマイクで録った分の、ラベルごとの最大番号（これより後の番号は、iPhoneで録った分）
PC_MAX_INDEX = {"lock": 22, "unlock": 20, "door": 18, "other": 8}


def file_number(name):
    return int(re.search(r"_(\d+)\.wav$", name).group(1))


def make_model():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))


def main():
    data = pd.read_csv(FEATURES_PATH)
    features = [column for column in data.columns if column.startswith("mfcc_")]
    data["is_pc"] = data["file"].map(file_number) <= data["label"].map(PC_MAX_INDEX)

    print("=== データの内訳（PCマイク / iPhone） ===")
    print(pd.crosstab(data["label"], data["is_pc"].map({True: "PC", False: "iPhone"})).reindex(LABELS))

    pc = data[data["is_pc"]]
    phone = data[~data["is_pc"]]
    if phone.empty:
        print("iPhoneのデータがありません。PC_MAX_INDEX を確認してください。")
        return

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    same_device = cross_val_score(make_model(), pc[features], pc["label"], cv=cv).mean()
    print(f"\nA) PCのデータ同士（交差検証）の正解率: {same_device:.3f}")

    model = make_model().fit(pc[features], pc["label"])
    predictions = model.predict(phone[features])
    accuracy = accuracy_score(phone["label"], predictions)
    print(f"B) PCで学習 → iPhoneの{len(phone)}件を判定した正解率: {accuracy:.3f}")

    print("\nB の混同行列（行=正解、列=予測）  順番: " + " / ".join(LABELS))
    print(confusion_matrix(phone["label"], predictions, labels=list(LABELS)))

    print("\nB の予測の内訳:")
    for label, count in pd.Series(predictions).value_counts().items():
        print(f"  {label}: {count}件")

    print("\nA より B が大きく低ければ、PCとiPhoneの音の違い（録音条件の差）が原因です。")


if __name__ == "__main__":
    main()