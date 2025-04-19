# What's New in This Version Converter Script

> [!IMPORTANT]
>
> This is not a universally applicable method, so we decided not to save it as an executable python script, but to write it in a tutorial for reference by those who need it.
>
> If you are using `libx264` encoding and want to use `decord` as the video backend to speed up the stats conversion, or want to use the process pool to speed up the conversion when converting the huge dataset like droid, you can use this script. 
>
> However, please note that the droid dataset may get stuck at episode 5545 during the conversion process.

Key improvements:

- support loading the local dataset
- support use decord as video backend (NOTICE: decord is not supported to 'libsvtav1' encode method, we test it using 'libx264', ref: https://github.com/dmlc/decord/issues/319)
- support process pool for huge dataset like droid to accelerate conversation speed

# 1. Convert LeRobot Dataset v20 to v21 Utils

## Installation

Install decord: https://github.com/dmlc/decord


## Default usage

This equal to lerobot projects, it will use dataset from huggingface hub, delete `stats.json` and push to huggingface hub (multi-thread and `pyav` as video backend), you can:

```bash
python utils/version_convert/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial \
    --delete-old-stats \
    --push-to-hub \
    --num-workers=8 \
    --video-backend=pyav
```



## Using `decord` as video backend

> [!IMPORTANT]
>
> 1.We recommend use default method to convert stats and use decord and process pool if you want to convert huge dataset like droid.
>
> 2.If you want to use decord as video backend, you should modify the `video_utils.py` source code from lerobot.

```python
def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
    video_path (Path): Path to the video file.
    timestamps (list[float]): List of timestamps to extract frames.
    tolerance_s (float): Allowed deviation in seconds for frame retrieval.
    backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
    torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    elif backend == "decord":
        return decode_video_frames_decord(video_path, timestamps)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_decord(
    video_path: Path | str,
    timestamps: list[float],
) -> torch.Tensor:
    video_path = str(video_path)
    vr = decord.VideoReader(video_path)
    num_frames = len(vr)
    frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
    indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
    frames = vr.get_batch(indices)

    frames_tensor = torch.tensor(frames.asnumpy()).type(torch.float32).permute(0, 3, 1, 2) / 255
    return frames_tensor
```

This will load local dataset, use `decord` as video backend and process pool, you can:

```bash
python utils/version_convert/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial \
    --root=/home/path/to/your/lerobot/dataset/path \
    --num-workers=8 \
    --video-backend=decord \
    --use-process-pool
    
```

## Speed Test

Table I. dataset conversation time use stats.

| dataset              | episodes | video_backend | method  | workers | video_encode | Time  |
| -------------------- | -------- | ------------- | ------- | ------- | ------------ | ----- |
| bekerley_autolab_ur5 | 896      | pyav          | thread  | 16      | libx264      | 10:56 |
| bekerley_autolab_ur5 | 896      | pyav          | process | 16      | libx264      | --    |
| bekerley_autolab_ur5 | 896      | decord        | thread  | 16      | libx264      | 11:44 |
| bekerley_autolab_ur5 | 896      | decord        | process | 16      | libx264      | 14:26 |

