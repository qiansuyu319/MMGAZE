import pandas as pd
import MLP_Best_Gaze
import os

video_name = '011'
head_num = '1'  


imgPath = f'/root/MMGaze_VGS/output/image_frame/{video_name}/{head_num}/'
head_bbox_file = f'/root/MMGaze_VGS/data/output/speaker_detect/head_boxes/{video_name}/{video_name}_frame_faces_boxes.npy'
mlp_checkpoint = '/root/MMGaze_VGS/utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/model_mlp.m'
imagepath = f'/root/MMGaze_VGS/data/output/fusion_speaker/{video_name}/com_speaker_feature/'
savepath = f'/root/MMGaze_VGS/output/gaze_detection_all/{video_name}/'
config_file = '/root/MMGaze_VGS/utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py'
checkpoint_file = '/root/MMGaze_VGS/utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth'
all_points_output_file = '/root/MMGaze_VGS/x101_1x_test.txt'
best_gaze_output_file = f'/root/MMGaze_VGS/output/gaze_detection/{video_name}/best_gaze.csv'
out_path_cat = f'/root/MMGaze_VGS/output/gaze_detection/{video_name}/'
gt_csv_path = f'/root/MMGaze_VGS/data/GT_VideoFormat/011_GT_VideoFormat.txt'
l2_summary_output_file = '/root/MMGaze_VGS/summary.csv'


speaker_map = f'/root/MMGaze_VGS/output/speaker_detect/{video_name}/pywork/{video_name}/faces.pckl'


os.makedirs(out_path_cat, exist_ok=True)

df_result = pd.read_csv(all_points_output_file)

MLP_Best_Gaze.mlp_best_gaze(
    imgPath=imgPath,
    df_result=df_result,
    video_name=video_name,
    head_bbox_file=head_bbox_file,
    mlp_checkpoint=mlp_checkpoint,
    imagepath=imagepath,
    savepath=savepath,
    config_file=config_file,
    checkpoint_file=checkpoint_file,
    all_points_output_file=all_points_output_file,
    best_gaze_output_file=best_gaze_output_file,
    out_path_cat=out_path_cat,
    gt_csv_path=gt_csv_path,
    l2_summary_output_file=l2_summary_output_file,
    speaker_map=speaker_map
)

print("✅ MLP gaze prediction complete.")
