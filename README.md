
---

# Gaze Evaluation Pipeline

## 1. Video Placement

Place your input video files (e.g., `.mp4` format) in the following directory:

```
./data/Raw_Video/videos
```

## 2. Set Head Numbers

For each video, you must specify the number of detected heads/tracks (`head_num`).
You can either:

* **Manually specify** a list, e.g.:

  ```python
  head_nums = [3, 2, 2, 2, 2, 2]  # One number per video in order
  ```
> ⚠️ Make sure `head_nums` matches the order and number of videos in `video_dir`.

## 3. Run the Pipeline

In your project root, run:

```bash
python eval_gaze.py
```

This will process all videos in `video_dir` using the corresponding `head_num` values.

## 4. Results

After completion, you will find the evaluation summary at:

```
summary.csv
```

This file contains per-video and overall summary statistics for gaze estimation.

---

## Example Workflow

1. Place `001.mp4`, `002.mp4`, etc. in `./data/Raw_Video/videos/`.
2. Edit `eval_gaze.py` to match the number of heads for each video.
3. Run:

   ```bash
   python eval_gaze.py
   ```
4. Check `summary.csv` for results.
