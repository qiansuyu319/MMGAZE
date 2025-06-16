from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os
import pandas as pd

device = 'cuda:0'  # or 'cpu' if CUDA is not available

def gaze_detect(config_file, checkpoint_file, imagepath, savepath, all_points_output_file): 
    print("🟢 Initializing model...")
    model = init_detector(config_file, checkpoint_file, device=device)

    os.makedirs(savepath, exist_ok=True)
    df_result = pd.DataFrame(columns=['frame_name', 'gaze_bbox', 'gaze_score'])

    # loop over all frames
    for filename in sorted(os.listdir(imagepath)):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue  # skip non-image files

        img_path = os.path.join(imagepath, filename)
        out_file = os.path.join(savepath, filename)

        print(f'🔍 Processing image: {filename}')
        result = inference_detector(model, img_path)

        if result is None or len(result) == 0 or result[0] is None:
            print(f'⚠️ No detection result for {filename}')
            continue

        for box in result[0]:  # result[0] contains list of [x1, y1, x2, y2, score]
            if len(box) == 5:
                gaze_bbox = box[:4]
                score = box[4]
                df_result = df_result.append({
                    'frame_name': filename,
                    'gaze_bbox': gaze_bbox,
                    'gaze_score': score
                }, ignore_index=True)

        # Visualize and save
        show_result_pyplot(model, img_path, result, out_file=out_file, score_thr=0.0,
                           title='result', wait_time=0, palette=None)

    # Save to CSV
    df_result.to_csv(all_points_output_file, index=False)
    print('✅ All gaze results saved in:', all_points_output_file)
