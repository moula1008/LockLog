import argparse
import zipfile
from pathlib import Path

from split_clips import DATASET_DIR, LABELS, log_source, next_index


def main():
    parser = argparse.ArgumentParser(description="収集モードで作ったZIPを、dataset に取り込みます。")
    parser.add_argument("zip_path", help="locklog_clips_*.zip のパス")
    parser.add_argument("--dry-run", action="store_true", help="保存せず、取り込む内容だけ表示する")
    args = parser.parse_args()

    zip_path = Path(args.zip_path).expanduser().resolve()
    if not zip_path.is_file():
        print(f"ZIPファイルが見つかりません: {zip_path}")
        return

    source = zip_path.stem   # 同じZIPのクリップは、同じ「録音の単位(group)」として扱う
    counts = {}
    with zipfile.ZipFile(zip_path) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            parts = info.filename.split("/")
            if info.is_dir() or len(parts) != 2 or parts[0] not in LABELS or not parts[1].lower().endswith(".wav"):
                print(f"  スキップ: {info.filename}")
                continue
            label = parts[0]
            counts[label] = counts.get(label, 0) + 1
            if args.dry_run:
                continue
            save_dir = DATASET_DIR / label
            save_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{label}_{next_index(save_dir, label):03d}.wav"
            (save_dir / file_name).write_bytes(archive.read(info))
            log_source(label, file_name, source)

    summary = ", ".join(f"{name}={counts[name]}" for name in LABELS if name in counts) or "なし"
    print(f"{zip_path.name}: {'（確認のみ）' if args.dry_run else '取り込みました'} {summary}")


if __name__ == "__main__":
    main()