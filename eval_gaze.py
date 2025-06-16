import os
import glob
from tqdm import tqdm
import sys
import subprocess
import pandas as pd
import ast
import MLP_Best_Gaze
from speaker_feature import fusion
from utils.Mask_RCNN.tools.run_maskrcnn_inference import run_inference_on_folder

def run_command(cmd_list, cwd=None):
    """
    Run a system command as a subprocess, optionally specifying the working directory.
    If the command fails, print an error and exit.
    """
    print(f"Running command: {' '.join(cmd_list)}")
    try:
        subprocess.run(cmd_list, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def get_paths(video_name):
    """
    Generate commonly used paths for a given video name.
    All paths are relative to the project root.
    """
    return {
        'raw_video_dir': './data/Raw_Video/videos',
        'image_frame_dir': './output/image_frame',
        'speaker_detector_dir': './utils/Speaker_detector/',
        'speaker_detect_output_dir': './output/speaker_detect',
        'mask_rcnn_tools_dir': './utils/Mask_RCNN/tools',
        'mlp_checkpoint': './utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/model_mlp.m',
        'fusion_speaker_dir': f'./utils/Speaker_detector/speaker_detect/data/face_maps/{video_name}',
        'fusion_output_dir': './output/fusion_speaker',
        'gaze_detection_all_dir': './output/gaze_detection_all',
        'config_file': './utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_coco.py',
        'checkpoint_file': './utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth',
        'all_points_output_file': f'./output/inference_result/{video_name}/inference_results.txt',
        'gaze_detection_dir': './output/gaze_detection',
        'gt_data_dir': './data/GT_VideoFormat',
        'l2_summary_output_file': './summary.csv',
        'speaker_info_path': f'./output/speaker-info/{video_name}_speaker_info.csv',
    }

def read_inference_result(filepath):
    """
    Read inference results from a text file, line by line.
    Each line is: filename,array([...])
    Handles parsing errors gracefully and returns a DataFrame.
    """
    data = []
    with open(filepath, 'r') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) != 2:
                print(f"Warning: malformed line {lineno}: {line}")
                continue
            filename, array_str = parts
            try:
                bbox_array = ast.literal_eval(array_str)
            except Exception as e:
                print(f"Warning: failed to parse array at line {lineno}: {e}")
                bbox_array = None
            data.append({
                'filename': filename,
                'bbox_array': bbox_array,
                'raw_array_str': array_str,
            })
    df = pd.DataFrame(data)
    return df

def process_single_video(video_name, paths, head_num=1):
    """
    Complete pipeline to process a single video with a given head number.
    Steps: Data preprocessing, speaker detection, feature fusion, Mask R-CNN inference, and gaze prediction.
    """
    print(f"\n=== Processing video {video_name}, head_num={head_num} ===")

    # 1. Data Preprocessing (extract frames for each head)
    run_command([
        "python", "Data_Preprocessing.py",
        "--videoPath", os.path.join(paths['raw_video_dir'], f"{video_name}.mp4"),
        "--imgPath", os.path.join(paths['image_frame_dir'], video_name, str(head_num)),
        "--head_num", str(head_num)
    ])

    # 2. Speaker detection
    raw_video_path = os.path.abspath(os.path.join(paths['raw_video_dir'], f"{video_name}.mp4"))
    output_base_path = os.path.abspath(paths['speaker_detect_output_dir'])
    print(f"Running speaker detection with raw video: {raw_video_path}")
    print(f"Output base path: {output_base_path}")
    run_command([
        "sh", "build_face_map_Generate_FaceMap.sh",
        raw_video_path,
        video_name,
        output_base_path
    ], cwd=paths['speaker_detector_dir'])

    # 3. Feature fusion (combine speaker and non-speaker features)
    fusion_raw_path = os.path.join(paths['image_frame_dir'], video_name, str(head_num))
    fusion_speaker_path = paths['fusion_speaker_dir']
    fusion_nonspeaker_path = paths['fusion_speaker_dir']  # usually the same in your setup
    fusion_output_path = os.path.join(paths['fusion_output_dir'], video_name)
    fusion(fusion_raw_path, fusion_speaker_path, fusion_nonspeaker_path, fusion_output_path)
    print(f"Checking fusion output directory: {fusion_output_path}")

    # 4. Mask R-CNN inference (detect faces in each frame)
    run_inference_on_folder(video_name)

    # 5. MLP_Best_Gaze: calculate the best gaze points for each head
    imgPath = os.path.join(paths['image_frame_dir'], video_name, str(head_num))
    head_bbox_file = os.path.join(paths['speaker_detect_output_dir'], "head_boxes", video_name, f"{video_name}_frame_faces_boxes.npy")
    df_result = pd.read_csv(paths['all_points_output_file'])
    best_gaze_output_file = os.path.join(paths['gaze_detection_dir'], video_name, "best_gaze.csv")
    out_path_cat = os.path.join(paths['gaze_detection_dir'], video_name)
    os.makedirs(out_path_cat, exist_ok=True)

    MLP_Best_Gaze.mlp_best_gaze(
        imgPath=imgPath,
        df_result=df_result,
        video_name=video_name,
        head_bbox_file=head_bbox_file,
        mlp_checkpoint=paths['mlp_checkpoint'],
        imagepath=os.path.join(paths['fusion_speaker_dir'], video_name, "com_speaker_feature"),
        savepath=os.path.join(paths['gaze_detection_all_dir'], video_name),
        config_file=paths['config_file'],
        checkpoint_file=paths['checkpoint_file'],
        all_points_output_file=paths['all_points_output_file'],
        best_gaze_output_file=best_gaze_output_file,
        out_path_cat=out_path_cat,
        gt_csv_path=os.path.join(paths['gt_data_dir'], f"{video_name}_GT_VideoFormat.txt"),
        l2_summary_output_file=paths['l2_summary_output_file'],
        speaker_map=paths['speaker_info_path'],
    )

if __name__ == "__main__":
    """
    Main execution: batch process all videos in the raw video directory.
    For each video, process all head numbers as specified in `head_nums`.
    """
    video_dir = './data/Raw_Video/videos'
    video_paths = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    video_names = [os.path.splitext(os.path.basename(p))[0] for p in video_paths]

    print("Processing videos in this order:")
    for i, name in enumerate(video_names):
        print(f"{i+1}: {name}")

    # For each video, write how many heads in the video 
    head_nums = []  # Consisty with the vdeo names 

    assert len(video_names) == len(head_nums), "video_names and head_nums must have the same length!"

    for video_name, max_head_num in tqdm(zip(video_names, head_nums), total=len(video_names), desc="Processing videos"):
        for head_num in range(1, max_head_num + 1):
            paths = get_paths(video_name)
            process_single_video(video_name, paths, head_num)
        print(f"{video_name} (head_num={max_head_num}) finished!")
