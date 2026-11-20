import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面を開かず、画像として保存する
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_PATH = BASE_DIR / "outputs" / "inspect.png"
MAX_CLIPS = 6


def main():
    parser = argparse.ArgumentParser(description="クリップの波形とスペクトログラムを、1枚の画像にまとめます。")
    parser.add_argument("clips", nargs="+", help="dataset からの相対パス。例: lock/lock_036.wav")
    args = parser.parse_args()
    if len(args.clips) > MAX_CLIPS:
        print(f"一度に指定できるのは{MAX_CLIPS}件までです。")
        return

    figure, axes = plt.subplots(2, len(args.clips), figsize=(3.4 * len(args.clips), 5.6), squeeze=False)
    for column, name in enumerate(args.clips):
        path = DATASET_DIR / name
        if not path.is_file():
            print(f"見つかりません: {path}")
            return
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        times = np.arange(len(audio)) / sr
        peak_time = float(np.argmax(np.abs(audio))) / sr
        axes[0, column].plot(times, audio, linewidth=0.5)
        axes[0, column].axvline(peak_time, color="red", linewidth=0.8, alpha=0.7)  # 最大音量の位置
        axes[0, column].set_ylim(-1, 1)
        axes[0, column].set_title(f"{name}\npeak {np.max(np.abs(audio)):.2f} @ {peak_time:.2f}s", fontsize=8)

        spectrum = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=1024, hop_length=256)), ref=1.0)
        librosa.display.specshow(
            spectrum, sr=sr, hop_length=256, x_axis="time", y_axis="hz",
            ax=axes[1, column], vmin=-90, vmax=-10,
        )
        axes[1, column].set_ylim(0, 12000)

    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=110)
    print(f"画像を保存しました: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
