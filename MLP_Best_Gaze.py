import numpy as np
import pandas as pd
import os
import joblib
import traceback

IMG_WIDTH = 1280
IMG_HEIGHT = 720

def calculate_gaze_k(head_point, model_mlp):
    # Predict the gaze slope (k) using the MLP model and head position.
    return model_mlp.predict([head_point])[0]

def calculate_closest_gaze(gaze_point, gaze_k_predict, head_point):
    # Calculate the gaze point closest to the predicted slope (k).
    if not gaze_point or len(gaze_point) != 4:
        return None
    try:
        gaze_x = (gaze_point[0] + gaze_point[2]) / 2
        gaze_y = (gaze_point[1] + gaze_point[3]) / 2
        k_maskrcnn = (gaze_y - head_point[1]) / (gaze_x - head_point[0])
        return [int(gaze_x), int(gaze_y)]
    except ZeroDivisionError:
        return None

def parse_bbox(bbox_str):
    # Convert a bbox string representation to a list of floats.
    bbox_str = bbox_str.replace('[', '').replace(']', '')
    return [float(v) for v in bbox_str.strip().split()]

def bbox_center(bbox):
    # Get the center point of a bbox.
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

def calculate_l2_distance_norm(point1, point2):
    # Compute normalized L2 distance between two points.
    x1, y1 = point1[0] / IMG_WIDTH, point1[1] / IMG_HEIGHT
    x2, y2 = point2[0] / IMG_WIDTH, point2[1] / IMG_HEIGHT
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def compute_min_l2_with_gt(row):
    # Compute the minimum L2 distance between predicted and GT gaze points.
    gaze_bboxes = row['gaze_point']
    gt_point = [row['gaze_x'], row['gaze_y']]
    if not gaze_bboxes or pd.isnull(gt_point[0]) or pd.isnull(gt_point[1]):
        return None
    if isinstance(gaze_bboxes[0], float) or isinstance(gaze_bboxes[0], int):
        gaze_bboxes = [gaze_bboxes]
    distances = [calculate_l2_distance_norm(bbox_center(bbox), gt_point) for bbox in gaze_bboxes]
    return np.min(distances)

def append_video_summary(l2_summary_output_file, video_name, filtered_df):
    # Append summary statistics for the current video to the summary CSV.
    if 'is_speaker' not in filtered_df.columns or 'min_l2' not in filtered_df.columns:
        raise ValueError("filtered_df must contain 'is_speaker' and 'min_l2' columns")
    video_summary = filtered_df.groupby('is_speaker').agg(
        avg_min_l2=('min_l2', 'mean'),
        frame_count=('min_l2', 'count')
    ).reset_index()
    video_summary.insert(0, 'video_name', video_name)
    file_exists = os.path.exists(l2_summary_output_file) and os.path.getsize(l2_summary_output_file) > 0
    video_summary.to_csv(
        l2_summary_output_file, mode='a' if file_exists else 'w',
        index=False, header=not file_exists, encoding='utf-8'
    )

def update_overall_summary(l2_summary_output_file):
    # Update or append the overall summary row at the end of the summary CSV.
    if not os.path.exists(l2_summary_output_file):
        print("Summary file not found, cannot update overall summary.")
        return
    df = pd.read_csv(l2_summary_output_file)
    df_videos = df[df['video_name'] != 'overall_summary']
    overall_summary = df_videos.groupby('is_speaker').agg(
        avg_min_l2=('avg_min_l2', 'mean'),
        frame_count=('frame_count', 'sum')
    ).reset_index()
    overall_summary.insert(0, 'video_name', 'overall_summary')
    df_final = pd.concat([df_videos, overall_summary], ignore_index=True)
    df_final.to_csv(l2_summary_output_file, index=False, encoding='utf-8')

def label_speakers_from_csv(filtered_df: pd.DataFrame, csv_path: str, head_column: str = "head_num") -> pd.DataFrame:
    # Assign speaker label (1 for speaker, 0 for non-speaker) using a mapping CSV.
    df_speaker = pd.read_csv(csv_path)
    df_speaker["frame_num_str"] = df_speaker["frame_id"].astype(str).str.zfill(3)
    df_speaker["speaker_id"] = df_speaker["speaker_face_idx"] + 1 # to match head_num starting from 1
    df_speaker["key"] = df_speaker["frame_num_str"]
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df["frame_index"] = filtered_df.index
    filtered_df["frame_num_str"] = filtered_df["frame_index"].astype(str).str.zfill(3)
    filtered_df["key"] = filtered_df["frame_num_str"]
    filtered_df["head_num_int"] = filtered_df[head_column].astype(int)
    merged = pd.merge(filtered_df, df_speaker[["key", "speaker_id"]], on="key", how="left")
    merged["is_speaker"] = (merged["head_num_int"] == merged["speaker_id"]).astype(int)
    return merged

def mlp_best_gaze(imgPath, df_result, video_name, head_bbox_file, mlp_checkpoint,
                  imagepath, savepath, config_file, checkpoint_file,
                  all_points_output_file, best_gaze_output_file, out_path_cat,
                  gt_csv_path, l2_summary_output_file, speaker_map):

    try:
        # Load trained MLP model for gaze slope prediction.
        model_mlp = joblib.load(mlp_checkpoint)
        # Load head bounding boxes.
        arr = np.load(head_bbox_file, allow_pickle=True).item()
        df = pd.DataFrame(arr)
        df['frame_num'] = range(len(df))
        df['frame_num'] = df['frame_num'].apply(str).apply(lambda x: x.zfill(3))
        df = pd.melt(df, id_vars='frame_num', var_name='head_num', value_name='head_bbox')
        df['head_num'] = df['head_num'].astype(int).map(lambda x: x + 1).astype(str)
        df['video_id'] = video_name
        df['frame_name'] = df['video_id'] + df['head_num'] + df['frame_num'] + '.jpg'
        df_head_bbox = df[['frame_name', 'head_bbox']]
        df_result['frame_name'] = df_result['frame_name'].astype(str).str.strip()
        df_head_bbox['frame_name'] = df_head_bbox['frame_name'].astype(str).str.strip()
        if not df_result['frame_name'].iloc[0].endswith('.jpg'):
            df_result['frame_name'] = df_result['frame_name'] + '.jpg'
        if not df_head_bbox['frame_name'].iloc[0].endswith('.jpg'):
            df_head_bbox['frame_name'] = df_head_bbox['frame_name'] + '.jpg'
        # Merge result predictions and head bounding box data.
        merged_df = pd.merge(df_result, df_head_bbox, on='frame_name', how='inner')
        if merged_df.empty:
            print("[WARN] merged_df is empty, aborting further processing.")
            return
        # Load ground truth gaze points.
        gt_df = pd.read_csv(gt_csv_path, header=None)
        gt_df.columns = ['frame_name', 'head_xmin', 'head_ymin', 'head_xmax', 'head_ymax', 'gaze_x', 'gaze_y']
        if not gt_df['frame_name'].iloc[0].endswith('.jpg'):
            gt_df['frame_name'] = gt_df['frame_name'].astype(str) + '.jpg'
        merged_df = pd.merge(merged_df, gt_df[['frame_name', 'gaze_x', 'gaze_y']], on='frame_name', how='left')
        # Add additional columns for each frame.
        merged_df['head_num'] = merged_df['frame_name'].str[3:4]
        merged_df['frame_seq'] = merged_df['frame_name'].str[4:-4]
        merged_df['head_point'] = merged_df['head_bbox'].apply(lambda bbox: [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)])
        merged_df['gaze_point'] = merged_df['gaze_bbox'].apply(parse_bbox)
        merged_df['gaze_k_predict'] = merged_df['head_point'].apply(lambda x: calculate_gaze_k(x, model_mlp))
        merged_df['gaze_best'] = merged_df.apply(lambda row: calculate_closest_gaze(row['gaze_point'], row['gaze_k_predict'], row['head_point']), axis=1)
        merged_df['min_l2'] = merged_df.apply(compute_min_l2_with_gt, axis=1)
        # Select the row with the minimum difference in predicted k for each frame.
        filtered_df = merged_df.loc[merged_df.groupby('frame_name')['gaze_k_predict'].idxmin()].reset_index(drop=True)
        # Optionally, label speaker/non-speaker if speaker map is provided.
        if speaker_map and os.path.exists(speaker_map):
            filtered_df = label_speakers_from_csv(filtered_df, speaker_map)
        filtered_df.to_csv(best_gaze_output_file, index=False)
        # Write per-video summary and overall summary.
        append_video_summary(l2_summary_output_file, video_name, filtered_df)
        update_overall_summary(l2_summary_output_file)
    except Exception:
        print("[FATAL ERROR] Exception in mlp_best_gaze:")
        traceback.print_exc()
