import os
import cv2

def get_all_file(dir_name, extensions):
    fullname_list, filename_list = [], []
    for root, dirs, files in os.walk(dir_name):
        for filename in files:
            full_path = os.path.join(root, filename)
            if ("Detectors" not in full_path) and any(filename.lower().endswith(ext) for ext in extensions):
                fullname_list.append(full_path)
                filename_list.append(filename)
    print(f"[get_all_file] Found {len(fullname_list)} files in {dir_name}")
    return fullname_list, filename_list

def com_speaker_feature(speaker_frames, no_speaker_frames, raw_frame, file_name, output_path):
    os.makedirs(output_path, exist_ok=True)
    print(f"[com_speaker_feature] Processing {file_name}")
    print(f"  raw_frame: {raw_frame}")
    print(f"  speaker_frames: {speaker_frames}")
    print(f"  no_speaker_frames: {no_speaker_frames}")

    frame = cv2.imread(raw_frame)
    no_speak_h = cv2.imread(no_speaker_frames[0]) if no_speaker_frames else None
    speak_h = cv2.imread(speaker_frames[0]) if speaker_frames else None

    if frame is None:
        print(f"  ERROR: raw frame failed to load: {raw_frame}")
    if no_speak_h is None:
        print(f"  ERROR: no speaker frame failed to load: {no_speaker_frames}")
    if speak_h is None:
        print(f"  ERROR: speaker frame failed to load: {speaker_frames}")

    if frame is None or no_speak_h is None or speak_h is None:
        print(f"  Skipping fusion for {file_name} due to load error.")
        return

    masked_frame = cv2.addWeighted(no_speak_h, 0.5, speak_h, 1, 0)
    com_frame = cv2.addWeighted(masked_frame, 0.7, frame, 1, 0)

    save_path = os.path.join(output_path, file_name)
    success = cv2.imwrite(save_path, com_frame)
    if success:
        print(f"  ✅ Fusion image saved at: {save_path}")
    else:
        print(f"  ❌ Failed to save fusion image at: {save_path}")

def fusion(raw_path, speaker_path, nonspeaker_path, output_path):
    print('========================Start to combine the speaker and raw frames.========================')
    print('Current working directory:', os.getcwd())

    raw_fullname, raw_filename = get_all_file(raw_path, ['.jpg', '.png'])
    speaker_fullname, _ = get_all_file(speaker_path, ['.jpg', '.png'])
    nonspeaker_fullname, _ = get_all_file(nonspeaker_path, ['.jpg', '.png'])

    for f_path, f_name in zip(raw_fullname, raw_filename):
        video_num = f_name[:3]
        frame_num = f_name[4:7]

        print(f"\n[frame] Processing {f_name} (video {video_num}, frame {frame_num})")

        speaker_imgs = []
        for s in speaker_fullname:
            base = os.path.basename(s)
            parts = base.split('_')
            if len(parts) < 3:
                continue
            cond = (base[:3] == video_num and
                    parts[1][1:].zfill(3) == frame_num and
                    'speaker' in base)
            if cond:
                speaker_imgs.append(s)
        print(f"  Found {len(speaker_imgs)} speaker images")

        nonspeaker_imgs = []
        for s in nonspeaker_fullname:
            base = os.path.basename(s)
            parts = base.split('_')
            if len(parts) < 3:
                continue
            cond = (base[:3] == video_num and
                    parts[1][1:].zfill(3) == frame_num and
                    'nonspeaker' in base)
            if cond:
                nonspeaker_imgs.append(s)
        print(f"  Found {len(nonspeaker_imgs)} nonspeaker images")

        if speaker_imgs:
            com_speaker_feature(speaker_imgs, nonspeaker_imgs, f_path, f_name, output_path)
        else:
            print(f"  No speaker image found for {f_name}, skipping.")
