from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
import os

def run_inference_on_folder(video_name):
    
    imagepath = f'./output/fusion_speaker/{video_name}/'
    savepath = f'./output/gaze_detection'
    config_file = f'./utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/mask_rcnn_x101_32x4d_fpn_1x_coco.py'
    checkpoint_file = f'./utils/Mask_RCNN/yuqi_x101_32x4d_fpn_1x_HS/epoch_12.pth'
    inference_result_dir = f'./output/inference_result/{video_name}/'
    
    # Make sure output directories exist
    os.makedirs(inference_result_dir, exist_ok=True)
    os.makedirs(savepath, exist_ok=True)
    
    inference_result_file = os.path.join(inference_result_dir, 'inference_results.txt')
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint_file, device=device)

    write_header = not os.path.exists(inference_result_file)
    with open(inference_result_file, 'a') as f:
        if write_header:
            f.write("frame_name,gaze_bbox\n")

        for filename in os.listdir(imagepath):
            img = os.path.join(imagepath, filename)
            result = inference_detector(model, img)
            out_file = os.path.join(savepath, filename)

            # bbox_str: convert numpy array to string, empty if no detection
            bbox_str = str(result[0][0]) if len(result[0]) > 0 else ""
            line = f"{filename},{bbox_str}"
            f.write(line + "\n")
            print('txt line is', line)

            # Save visualization results
            show_result_pyplot(model, img, result, out_file=out_file,
                               score_thr=0, title='result', wait_time=0, palette=None)
