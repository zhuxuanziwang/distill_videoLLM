#!/usr/bin/env python3
import argparse
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError("opencv-python is required. Please install requirements.txt first.") from e

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


def uniform_indices(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def extract_frames_for_video(video_path: Path, out_dir: Path, num_frames: int, image_size: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback when frame_count metadata is missing.
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened() or total <= 0:
            return 0

    picks = set(uniform_indices(total, num_frames))
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    cur = 0
    target = sorted(picks)
    tptr = 0
    while True:
        ok, frame = cap.read()
        if not ok or tptr >= len(target):
            break
        if cur == target[tptr]:
            if image_size > 0:
                frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            out_path = out_dir / f"frame_{saved:03d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            tptr += 1
        cur += 1
    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract uniform frames from MSRVTT videos.")
    parser.add_argument("--video_dir", type=str, default="data/msrvtt_raw/msrvtt_qa/MSRVTT-QA/video")
    parser.add_argument("--out_dir", type=str, default="data/msrvtt_frames")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_videos", type=int, default=0, help="0 means all videos.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    videos = sorted(video_dir.glob("*.mp4"))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError(f"No .mp4 files found under {video_dir}")

    ok_count = 0
    skip_count = 0
    fail_count = 0
    for vp in tqdm(videos, desc="extract_frames"):
        vid = vp.stem
        dst = out_dir / vid
        if dst.exists() and not args.overwrite and len(list(dst.glob("frame_*.jpg"))) >= args.num_frames:
            skip_count += 1
            continue
        saved = extract_frames_for_video(vp, dst, num_frames=args.num_frames, image_size=args.image_size)
        if saved >= 1:
            ok_count += 1
        else:
            fail_count += 1

    print(
        f"done. videos={len(videos)} extracted={ok_count} skipped={skip_count} failed={fail_count} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
