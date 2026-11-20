from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_PATH = BASE_DIR / "outputs" / "clip_quality.csv"
LABELS = ("lock", "unlock", "door", "other")

ZERO_TAIL = 0.05   # 完全な無音(0)が、全体のこの割合を超えたら「0埋め」とみなす
QUIET_PEAK = 0.06  # 最大音量がこれ未満なら「小さい」
NOISY_RANK = 0.10  # ラベルの中で、音の目立ち方が下位10%のものを「雑音っぽい」とする


def main():
    rows = []
    for label in LABELS:
        for path in sorted((DATASET_DIR / label).glob("*.wav")):
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            magnitude = np.abs(audio)
            peak = float(magnitude.max())
            floor = float(np.median(magnitude)) + 1e-6
            rows.append({
                "label": label,
                "file": f"{label}/{path.name}",
                "peak": round(peak, 3),
                "stand_out_db": round(20 * np.log10(peak / floor + 1e-9), 1),  # 最大音が、普段の音より何dB大きいか
                "zero_fraction": round(float((audio == 0).mean()), 3),
                "seconds": round(len(audio) / sr, 2),
            })

    table = pd.DataFrame(rows)
    limit = table.groupby("label")["stand_out_db"].transform(lambda values: values.quantile(NOISY_RANK))
    table["flag_zero_tail"] = table["zero_fraction"] > ZERO_TAIL
    table["flag_quiet"] = table["peak"] < QUIET_PEAK
    table["flag_noisy"] = table["stand_out_db"] <= limit

    def reasons(row):
        names = []
        if row["flag_zero_tail"]:
            names.append("端が無音(0埋め)")
        if row["flag_quiet"]:
            names.append("音が小さい")
        if row["flag_noisy"]:
            names.append("雑音っぽい")
        return " / ".join(names)

    table["reason"] = table.apply(reasons, axis=1)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    flagged = table[table["reason"] != ""].sort_values(["label", "stand_out_db"])
    print(f"全{len(table)}件のうち、確認をお勧めするクリップ: {len(flagged)}件\n")
    for label in LABELS:
        part = flagged[flagged["label"] == label]
        if part.empty:
            continue
        print(f"[{label}]  {len(part)}件")
        for row in part.itertuples():
            print(f"  {row.file:<24} peak={row.peak:.2f}  目立ち={row.stand_out_db:5.1f}dB  {row.reason}")
        print()
    print(f"全クリップの数値を保存しました: {OUTPUT_PATH}")
    print("確認のしかた: inspect_clips.py で波形を見て、実際に聞いてから判断してください。")


if __name__ == "__main__":
    main()