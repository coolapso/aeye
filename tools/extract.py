import cv2
import os
from tqdm import tqdm
import argparse

def save_frames(video_path, output_folder): 
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    filename = os.path.basename(video_path).split('.')[0]

    total_frames_to_extract = int(total_frames / fps)
    frame_count = 0
    saved_count = 0

    with tqdm(total=total_frames_to_extract, desc="Extracting Frames", unit="frame") as pbar:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_count % fps == 0:
                frame_filename = os.path.join(output_folder, f"{filename}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1

    cap.release()
    print(f"\nExtracted {saved_count} frames to {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Tool to extract frames from videos')
    parser.add_argument(
        '-i',
        '--input',
        dest='input',
        required=True,
        help='Video to extract frames from',
        type=str
    )

    parser.add_argument(
        '-o',
        '--output',
        dest='output',
        required=True,
        help='Frames destination',
    )

    args = parser.parse_args()
    save_frames(args.input, args.output)


if __name__ == "__main__":
    main()
