#!/usr/bin/env python3
"""Check local LeRobot v3 episode rows against decoded video segments.

This is a read-only, local audit. It does not contact the Hub or download files.
It is intended to run on an AgiBot-to-LeRobot conversion output.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import av
import pyarrow.parquet as pq


def count_frames_in_interval(
    frame_times: list[float], start: float, end: float, *, epsilon: float = 1e-6
) -> int:
    """Count video presentation timestamps in the half-open interval [start, end).

    LeRobot v3 stores episode video segments as start/end timestamps within a
    shared video file. The tiny epsilon absorbs float serialization noise at
    segment boundaries while keeping a frame exactly at ``end`` out.
    """
    if end < start:
        raise ValueError(f"Video interval ends before it starts: {start} > {end}")
    upper = end - epsilon
    lower = start - epsilon
    return sum(lower <= timestamp < upper for timestamp in frame_times)


def _read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        records.extend(pq.read_table(path).to_pylist())
    return records


def _decode_frame_times(video_path: Path) -> list[float]:
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("file has no video stream")
        stream = container.streams.video[0]
        times = []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("decoded frame has no presentation timestamp")
            times.append(float(frame.pts * stream.time_base))
    return times


def audit_dataset(root: Path) -> tuple[list[str], int, int, int]:
    """Return findings, episode count, data-row count, and video frame count."""
    root = root.expanduser().resolve()
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        return [f"Missing {info_path}"], 0, 0, 0

    info = json.loads(info_path.read_text())
    version = info.get("codebase_version")
    if version != "v3.0":
        return [f"Expected LeRobot v3.0, found {version!r}"], 0, 0, 0

    video_features = [
        key
        for key, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    ]
    if not video_features:
        return ["No video features declared in meta/info.json"], 0, 0, 0

    findings: list[str] = []
    episode_paths = list((root / "meta/episodes").glob("**/*.parquet"))
    data_paths = list((root / "data").glob("**/*.parquet"))
    if not episode_paths:
        return ["No episode metadata Parquet files found"], 0, 0, 0
    if not data_paths:
        return ["No data Parquet files found"], 0, 0, 0

    episodes = _read_records(episode_paths)
    data_counts: Counter[int] = Counter()
    total_data_rows = 0
    for path in sorted(data_paths):
        table = pq.read_table(path, columns=["episode_index"])
        indices = table.column("episode_index").to_pylist()
        data_counts.update(int(index) for index in indices)
        total_data_rows += len(indices)

    episode_ids = [int(row["episode_index"]) for row in episodes]
    duplicate_ids = sorted(
        index for index, count in Counter(episode_ids).items() if count > 1
    )
    if duplicate_ids:
        findings.append(f"Duplicate episode metadata rows: {duplicate_ids}")

    video_template = info.get("video_path")
    if not video_template:
        return (
            ["meta/info.json has no video_path template"],
            len(episodes),
            total_data_rows,
            0,
        )

    # Decode each video file once, even when it contains many episode segments.
    file_rows: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        for feature in video_features:
            prefix = f"videos/{feature}"
            try:
                chunk_index = int(row[f"{prefix}/chunk_index"])
                file_index = int(row[f"{prefix}/file_index"])
                file_rows[(feature, chunk_index, file_index)].append(row)
            except (KeyError, TypeError, ValueError) as exc:
                findings.append(
                    f"Episode {row.get('episode_index', '?')} / {feature}: "
                    f"incomplete video metadata ({exc})"
                )

    decoded_files: dict[tuple[str, int, int], list[float]] = {}
    total_video_frames = 0
    for (feature, chunk_index, file_index), rows in sorted(file_rows.items()):
        video_path = root / video_template.format(
            video_key=feature, chunk_index=chunk_index, file_index=file_index
        )
        if not video_path.is_file():
            findings.append(f"Missing video file: {video_path}")
            continue
        try:
            frame_times = _decode_frame_times(video_path)
        except (av.error.FFmpegError, OSError, ValueError) as exc:
            findings.append(f"Cannot decode {video_path}: {exc}")
            continue

        decoded_files[(feature, chunk_index, file_index)] = frame_times
        total_video_frames += len(frame_times)
        expected_file_frames = sum(int(row["length"]) for row in rows)
        if len(frame_times) != expected_file_frames:
            findings.append(
                f"{feature} chunk {chunk_index} file {file_index}: "
                f"decoded {len(frame_times)} frames, episode metadata expects "
                f"{expected_file_frames}"
            )

    for row in episodes:
        episode_index = int(row["episode_index"])
        length = int(row["length"])
        actual_rows = data_counts.get(episode_index, 0)
        if actual_rows != length:
            findings.append(
                f"Episode {episode_index}: Parquet has {actual_rows} rows, "
                f"metadata declares {length}"
            )

        for feature in video_features:
            prefix = f"videos/{feature}"
            try:
                chunk_index = int(row[f"{prefix}/chunk_index"])
                file_index = int(row[f"{prefix}/file_index"])
                start = float(row[f"{prefix}/from_timestamp"])
                end = float(row[f"{prefix}/to_timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            frame_times = decoded_files.get((feature, chunk_index, file_index))
            if frame_times is None:
                continue
            count = count_frames_in_interval(frame_times, start, end)
            if count != length:
                findings.append(
                    f"Episode {episode_index} / {feature}: timestamp range "
                    f"[{start:.6f}, {end:.6f}) contains {count} video frames; "
                    f"metadata declares {length}"
                )

    expected_total = info.get("total_frames")
    if expected_total is not None and int(expected_total) != total_data_rows:
        findings.append(
            f"info.json declares {expected_total} total frames, "
            f"but data Parquet contains {total_data_rows} rows"
        )
    return findings, len(episodes), total_data_rows, total_video_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root", type=Path, help="Local LeRobot v3 dataset directory"
    )
    args = parser.parse_args()

    findings, episode_count, data_rows, video_frames = audit_dataset(args.dataset_root)
    print(f"Dataset: {args.dataset_root.expanduser()}")
    print(
        f"Episodes: {episode_count} | Parquet rows: {data_rows} | "
        f"decoded video frames: {video_frames}"
    )
    if findings:
        print(f"FAIL: {len(findings)} alignment/integrity finding(s)")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("PASS: episode metadata, Parquet rows, and decoded video segments agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
