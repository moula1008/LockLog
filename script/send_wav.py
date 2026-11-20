import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description="WAVファイルをAPIに送って、判定結果を表示します。")
    parser.add_argument("wav_path", help="送るWAVファイル")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="APIのURL")
    args = parser.parse_args()

    audio, sample_rate = sf.read(Path(args.wav_path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    request = urllib.request.Request(
        f"{args.url}/api/predict?sample_rate={sample_rate}",
        data=pcm,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
    except urllib.error.HTTPError as error:
        print("エラー:", error.code, error.read().decode("utf-8"))


if __name__ == "__main__":
    main()