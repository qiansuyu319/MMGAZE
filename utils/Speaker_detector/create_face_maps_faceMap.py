import numpy as np
import pickle as pkl
from skimage.draw import polygon_perimeter
import skvideo.io
import argparse
import ntpath
import os
import imageio
import skimage
import pandas as pd

def get_face_blob(img_shape, x, y, s, k1=0.6, k2=0.7):
    img = np.zeros(img_shape, dtype=np.uint8)
    r_center = y
    c_center = x
    r_radius = s * k2
    c_radius = s * k1
    rr, cc = skimage.draw.ellipse(r=r_center, c=c_center, r_radius=r_radius, c_radius=c_radius, rotation=0)
    rr = np.clip(rr, 0, img_shape[0] - 1)
    cc = np.clip(cc, 0, img_shape[1] - 1)
    img[rr, cc] = 1
    return img

parser = argparse.ArgumentParser(description="Get_speaker_score")
parser.add_argument('--tracks', type=str, required=True, help='Path to tracks file')
parser.add_argument('--video', type=str, required=True, help='Path to original video')
parser.add_argument('--scores', type=str, required=True, help='Path to speaker scores file')
parser.add_argument('--out_video', type=str, required=True, help='Output path for boxed video')
parser.add_argument('--maps_out', type=str, required=True, help='Output path for saving face maps')
parser.add_argument('--bboxes_out', type=str, required=True, help='Output path for saving bounding boxes')
parser.add_argument('--reference', type=str, default='', help='Name of the video');

opt = parser.parse_args()

# 
os.makedirs(opt.maps_out, exist_ok=True)
os.makedirs(opt.bboxes_out, exist_ok=True)
os.makedirs(opt.out_video, exist_ok=True)

with open(opt.tracks, 'rb') as fil:
    tracks = pkl.load(fil)

videodata = skvideo.io.vread(opt.video)
whospeaks = np.load(opt.scores)

nfaces = len(tracks)
nframes = min(len(tracks[0][0][0]), len(whospeaks), videodata.shape[0])

face_boxes = {str(face): [] for face in range(nfaces)}
frame_faces_boxes = {str(face): [] for face in range(nfaces)}

vidName = os.path.basename(opt.video)[:-4]
vdata_boxes = []
# add a list to store speaker and non-speaker maps
speak_info_list = []

for frame in range(nframes):
    print(f'Processing frame {frame+1} of {nframes}')
    whospeaks_now = whospeaks[frame]
    speak_info_list.append({'frame_id': frame, 'speaker_face_idx': int(whospeaks_now)})

    curr_frame = videodata[frame].copy()
    img_shape = curr_frame.shape[:2]

    speaker_map = np.zeros(img_shape, dtype=np.uint8)
    non_speaker_map = np.zeros(img_shape, dtype=np.uint8)

    for face in range(nfaces):
        sizes = tracks[face][1][0]
        xs = tracks[face][1][1]
        ys = tracks[face][1][2]

        s = int(sizes[frame])
        x = int(xs[frame])
        y = int(ys[frame])
        s = int(s * 1.8)

        start = np.array([x - s, y - s])
        extent = np.array([x + s, y + s]) - start

        m = get_face_blob(img_shape, x, y, s)

        p1 = (start[0], start[1])
        p2 = (start[0] + extent[0], start[1])
        p3 = (start[0] + extent[0], start[1] + extent[1])
        p4 = (start[0], start[1] + extent[1])

        face_boxes[str(face)].append([p1, p2, p3, p4])

        r = np.array([p1[0], p2[0], p3[0], p4[0]])
        c = np.array([p1[1], p2[1], p3[1], p4[1]])

        xmin, xmax = r.min(), r.max()
        ymin, ymax = c.min(), c.max()

        frame_faces_boxes[str(face)].append([xmin, ymin, xmax, ymax])

        rr, cc = polygon_perimeter(r, c)
        rr = np.clip(rr, 0, img_shape[0] - 1)
        cc = np.clip(cc, 0, img_shape[1] - 1)

        if face == whospeaks_now:
            curr_frame[rr, cc, :] = [255, 0, 0]  # Red for speaker
            speaker_map += m
        else:
            curr_frame[rr, cc, :] = [0, 0, 255]  # Blue for non-speaker
            non_speaker_map += m

    vdata_boxes.append(np.expand_dims(curr_frame, axis=0))

    # store face-map 
    name_speaker = os.path.join(opt.maps_out, f'{vidName}_f{frame}_speaker.jpg')
    name_non_speaker = os.path.join(opt.maps_out, f'{vidName}_f{frame}_nonspeaker.jpg')

    imageio.imwrite(name_speaker, (speaker_map * 255).astype(np.uint8))
    imageio.imwrite(name_non_speaker, (non_speaker_map * 255).astype(np.uint8))

boxed_video = np.vstack(vdata_boxes)

video_fname = ntpath.basename(opt.video)
v_split = video_fname.split('.')
video_fname, extension = v_split[0], v_split[1]

writer = skvideo.io.FFmpegWriter(os.path.join(opt.out_video, f'{video_fname}_boxed.{extension}'), outputdict={'-b': '300000000'})
for i in range(boxed_video.shape[0]):
    writer.writeFrame(boxed_video[i])
writer.close()

np.save(os.path.join(opt.bboxes_out, f'{video_fname}_faces_bboxes.npy'), face_boxes)
np.save(os.path.join(opt.bboxes_out, f'{video_fname}_frame_faces_boxes.npy'), frame_faces_boxes)
os.makedirs(opt.bboxes_out, exist_ok=True)
# store speaker information 
video_name = opt.reference
speaker_info_dir = './output/speaker-info'
speaker_csv_path = os.path.join(speaker_info_dir, f'{video_name}_speaker_info.csv')

# Ensure the directory exists
os.makedirs(speaker_info_dir, exist_ok=True)
os.makedirs(opt.bboxes_out, exist_ok=True)

df_speaker_info = pd.DataFrame(speak_info_list)
df_speaker_info.to_csv(speaker_csv_path, index=False)
print(f"✅ File saved: {speaker_csv_path}")