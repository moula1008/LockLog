import argparse
import csv
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "raw"
DATASET_DIR = BASE_DIR / "dataset"
SOURCES_PATH = DATASET_DIR / "clip_sources.csv"
LABELS = ("lock", "unlock", "door", "other")

PRE_SEC = 1.0        # 一番大きい音の何秒前から切り出すか
POST_SEC = 2.0       # 一番大きい音の何秒後まで切り出すか（合計3秒）
FRAME_SEC = 0.02     # 音量を測る窓の長さ
HOP_SEC = 0.01       # 窓をずらす間隔
MERGE_GAP_SEC = 1.5  # この時間以内に続いた音は、1回の操作とみなす


def next_index(save_dir, label):
    """label_001.wav, label_002.wav... の最大番号 + 1 を返す"""
    pattern = re.compile(rf"^{re.escape(label)}_(\d+)\.wav$")
    numbers = []
    for path in save_dir.glob("*.wav"):
        match = pattern.match(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0) + 1


def log_source(label, file_name, source_name):
    """どの録音から切り出したかを記録する（評価で、録音ごとに分けるため）"""
    is_new = not SOURCES_PATH.exists()
    if not is_new and SOURCES_PATH.stat().st_size > 0:
        # 手作業で編集したCSVの末尾に改行がないと、次の行がつながるので、先に改行を補う
        with open(SOURCES_PATH, "rb+") as handle:
            handle.seek(-1, 2)
            if handle.read(1) not in (b"\n", b"\r"):
                handle.write(b"\r\n")
    with open(SOURCES_PATH, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["label", "file", "source"])
        writer.writerow([label, file_name, source_name])


def measure(audio, sr, factor, min_rms):
    """短い窓ごとの音量を測り、(窓の開始位置, 音量, 静かさ, しきい値, 窓の長さ) を返す"""
    frame = int(FRAME_SEC * sr)
    hop = int(HOP_SEC * sr)
    if len(audio) < frame:
        return None
    squares = audio.astype(np.float64) ** 2
    cumulative = np.concatenate([[0.0], np.cumsum(squares)])
    starts = np.arange(0, len(audio) - frame + 1, hop)
    rms = np.sqrt((cumulative[starts + frame] - cumulative[starts]) / frame)
    noise = float(np.median(rms))                  # 普段の静かさ
    threshold = max(noise * factor, min_rms)       # これを超えたら「音がした」
    return starts, rms, noise, threshold, frame


def find_events(audio, sr, factor, min_rms, gap_sec):
    """音がした場所(最大音量の位置)をサンプル番号のリストで返す"""
    measured = measure(audio, sr, factor, min_rms)
    if measured is None:
        return [], 0.0, 0.0
    starts, rms, noise, threshold, frame = measured

    active = np.flatnonzero(rms > threshold)
    if active.size == 0:
        return [], threshold, noise

    gap_frames = int(gap_sec / HOP_SEC)
    groups = []
    group_start = prev = active[0]
    for index in active[1:]:
        if index - prev > gap_frames:
            groups.append((group_start, prev))
            group_start = index
        prev = index
    groups.append((group_start, prev))

    peaks = []
    for first, last in groups:
        begin = starts[first]
        end = starts[last] + frame
        peaks.append(begin + int(np.argmax(np.abs(audio[begin:end]))))
    return peaks, threshold, noise


def find_fixed_windows(audio, sr, factor, min_rms):
    """先頭から3秒ごとに区切り、音がある区間の開始位置を返す（静かな区間は捨てる）"""
    measured = measure(audio, sr, factor, min_rms)
    if measured is None:
        return [], 0.0, 0.0
    starts, rms, noise, threshold, frame = measured

    length = int((PRE_SEC + POST_SEC) * sr)
    windows = []
    for begin in range(0, len(audio) - length + 1, length):  # 端の中途半端な部分は捨てる
        inside = (starts >= begin) & (starts + frame <= begin + length)
        if inside.any() and rms[inside].max() > threshold:
            windows.append(begin)
    return windows, threshold, noise


def cut_clip(audio, sr, peak):
    """最大音量の位置を基準に、3秒のクリップを切り出す"""
    length = int((PRE_SEC + POST_SEC) * sr)
    start = peak - int(PRE_SEC * sr)
    clip = audio[max(0, start):min(len(audio), start + length)]
    pad_left = max(0, -start)
    pad_right = max(0, length - len(clip) - pad_left)
    return np.pad(clip, (pad_left, pad_right))


def main():
    parser = argparse.ArgumentParser(description="長い録音を、3秒のクリップに切り分けます。")
    parser.add_argument("--dry-run", action="store_true", help="保存せず、検出結果だけ表示する")
    parser.add_argument("--factor", type=float, default=6.0, help="静かさの何倍を音とみなすか（小さいほど敏感）")
    parser.add_argument("--min-rms", type=float, default=0.004, help="音とみなす最小の音量")
    parser.add_argument("--gap", type=float, default=MERGE_GAP_SEC, help="この秒数以内に続いた音は1回とみなす")
    parser.add_argument("--expect", type=int, default=None, help="1ファイルで期待するクリップ数")
    parser.add_argument(
        "--sequence",
        default=None,
        help="raw/session/ の録音を、音の順番でラベル付けする。例: unlock,lock（この順に繰り返す）",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="音が続く録音向け。先頭から3秒ごとに区切る（静かな区間は捨てる）",
    )
    args = parser.parse_args()

    if args.sequence and args.fixed:
        print("--sequence と --fixed は同時に使えません。")
        return

    sequence = None
    folders = list(LABELS)
    if args.sequence:
        sequence = [name.strip() for name in args.sequence.split(",")]
        unknown = [name for name in sequence if name not in LABELS]
        if unknown:
            print("ラベル名が違います:", ", ".join(unknown), "/ 使えるのは", ", ".join(LABELS))
            return
        folders = ["session"]

    clip_seconds = PRE_SEC + POST_SEC
    found_any = False
    for folder in folders:
        for raw_path in sorted((RAW_DIR / folder).glob("*.wav")):
            found_any = True
            audio, sr = sf.read(raw_path, dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            # 切り出し結果を (開始秒, 最大音量, クリップ) のリストにそろえる
            clips = []
            if args.fixed:
                windows, threshold, noise = find_fixed_windows(audio, sr, args.factor, args.min_rms)
                size = int(clip_seconds * sr)
                for begin in windows:
                    segment = audio[begin:begin + size]
                    clips.append((begin / sr, float(np.max(np.abs(segment))), segment, True))
            else:
                peaks, threshold, noise = find_events(audio, sr, args.factor, args.min_rms, args.gap)
                for peak in peaks:
                    # 前後の長さが足りない(録音の端に近い)クリップは、0埋めになるので保存しない
                    complete = peak - int(PRE_SEC * sr) >= 0 and peak + int(POST_SEC * sr) <= len(audio)
                    clips.append((peak / sr, abs(float(audio[peak])), cut_clip(audio, sr, peak), complete))

            if sequence:
                labels = [sequence[i % len(sequence)] for i in range(len(clips))]
            else:
                labels = [folder] * len(clips)

            unit = "区間" if args.fixed else "音"
            print(f"\n[{folder}] {raw_path.name}  ({len(audio) / sr:.1f}秒, {sr}Hz)")
            print(f"  静かさ={noise:.4f}  検出しきい値={threshold:.4f}  検出した{unit}: {len(clips)}件")
            for number, ((time_sec, level, _, complete), label) in enumerate(zip(clips, labels), start=1):
                warning = "  ← 音割れの疑い" if level >= 0.99 else ""
                if not complete:
                    warning += "  ← 録音の端に近いので保存しません"
                print(f"    {number:2d}: {time_sec:7.2f}秒  {label:<6} 最大音量={level:.3f}{warning}")

            if args.dry_run:
                continue
            if not clips:
                print("  検出できなかったので、スキップします。--factor を小さくしてみてください。")
                continue
            if args.expect is not None and len(clips) != args.expect:
                print(f"  期待({args.expect}件)と違うので保存しません。--factor を調整するか、--dry-run で確認してください。")
                continue
            if sequence and len(clips) % len(sequence) != 0:
                print(f"  検出数が、順番({args.sequence})の繰り返しに合わないので保存しません。--dry-run で確認してください。")
                continue

            saved_labels = []
            for (_, _, clip, complete), label in zip(clips, labels):
                if not complete:
                    continue
                saved_labels.append(label)
                save_dir = DATASET_DIR / label
                save_dir.mkdir(parents=True, exist_ok=True)
                index = next_index(save_dir, label)
                file_name = f"{label}_{index:03d}.wav"
                sf.write(save_dir / file_name, clip, sr)
                log_source(label, file_name, raw_path.name)
            summary = ", ".join(f"{name}={saved_labels.count(name)}" for name in dict.fromkeys(saved_labels)) or "なし"
            print(f"  保存しました（{summary}）")

            done_dir = RAW_DIR / folder / "done"
            done_dir.mkdir(exist_ok=True)
            shutil.move(str(raw_path), str(done_dir / raw_path.name))

    if not found_any:
        print(f"WAVファイルがありません。{RAW_DIR / folders[0]} に置いてください。")


if __name__ == "__main__":
    main()