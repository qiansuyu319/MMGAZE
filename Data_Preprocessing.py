import argparse
import cv2
import os

def Video2Pic(videoPath, imgPath, head_num):
    cap = cv2.VideoCapture(videoPath)
    print('Now deal with video:', videoPath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    suc = cap.isOpened()
    print("suc", suc)
    frame_count = 0

    if not os.path.exists(imgPath):
        os.makedirs(imgPath)

    while suc:
        suc, frame = cap.read()
        if not suc:
            break

        frame_name = os.path.join(
            imgPath,
            f"{os.path.basename(videoPath)[-7:-4]}{head_num}{str(frame_count).zfill(3)}.jpg"
        )
        print('Save image into path:', frame_name)
        try:
            cv2.imwrite(frame_name, frame)
            frame_count += 1
            cv2.waitKey(1)
        except Exception as e:
            print("Failed to save image:", e)
            continue

    cap.release()
    print("Converting videos to images finished！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videoPath', type=str, required=True, help='Input video path')
    parser.add_argument('--imgPath', type=str, required=True, help='Output image folder')
    parser.add_argument('--head_num', type=int, required=True, help='Head number in the video')

    args = parser.parse_args()
    Video2Pic(args.videoPath, args.imgPath, args.head_num)
