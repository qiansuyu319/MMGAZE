import os
import subprocess
import argparse

def run_speaker_pipeline(video_path, ref_name, out_dir):
    # Ensure absolute paths
    video_path = os.path.abspath(video_path)
    out_dir = os.path.abspath(out_dir)
    
    # Paths used in all steps
    pywork_dir = os.path.join(out_dir, 'pywork', ref_name)
    V_feats = os.path.join(pywork_dir, 'V_feats.npy')
    A_feats = os.path.join(pywork_dir, 'A_feats.npy')
    tracks = os.path.join(pywork_dir, 'tracks.pckl')
    scores = os.path.join(pywork_dir, 'whospeaks.npy')

    out_video = os.path.join(out_dir, 'pyavi', ref_name)
    face_maps = os.path.join(out_dir, 'face_maps', ref_name)
    STS_maps = os.path.join(out_dir, 'ST_maps', ref_name)
    head_boxes = os.path.join(out_dir, 'head_boxes', ref_name)

    os.makedirs(face_maps, exist_ok=True)
    os.makedirs(STS_maps, exist_ok=True)
    os.makedirs(head_boxes, exist_ok=True)

    # Step 1: run_pipeline_fixed.py
    print("🚀 Step 1: run_pipeline_fixed.py")
    subprocess.run([
        "python", "speaker_detect/run_pipeline_fixed.py",
        "--videofile", video_path,
        "--reference", ref_name,
        "--data_dir", out_dir
    ], check=True)

    # Step 2: run_syncnet_fixed.py
    print("🚀 Step 2: run_syncnet_fixed.py")
    subprocess.run([
        "python", "speaker_detect/run_syncnet_fixed.py",
        "--videofile", video_path,
        "--reference", ref_name,
        "--data_dir", out_dir
    ], check=True)

    # Step 3: get_speaker_score.py
    print("🚀 Step 3: get_speaker_score.py")
    subprocess.run([
        "python", "speaker_detect/get_speaker_score.py",
        "--video_feats", V_feats,
        "--audio_feats", A_feats,
        "--out_scores", pywork_dir
    ], check=True)

    # Step 4: create_face_maps_faceMap.py
    print("🚀 Step 4: create_face_maps_faceMap.py")
    subprocess.run([
        "python", "create_face_maps_faceMap.py",
        "--tracks", tracks,
        "--video", video_path,
        "--scores", scores,
        "--out_video", out_video,
        "--maps_out", face_maps,
        "--bboxes_out", head_boxes
    ], check=True)

    print("✅ Speaker detection pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("--ref_name", type=str, required=True, help="Video reference name, e.g., '012'")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory (e.g., ./output/speaker_detect)")
    args = parser.parse_args()

    run_speaker_pipeline(args.video, args.ref_name, args.out_dir)
