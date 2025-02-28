#!/bin/python

import cv2
import os
import sys
import shutil
import argparse

RESIZED_DIMENSION = (128, 128)

dirs = {
        "no": "00-not-detected",
        "yes": "01-detected",
    }

texts = {
        "close": "[esc]: Close Window",
        "yes": "[y] Valid image",
        "no": "[n]: Invalid image",
    }

aurora_frames = []
no_aurora_frames = []
skipped_frames = []


def review_frames(dataset_root_dir: str, frames_dir: str):
    frames = os.listdir(os.path.join(dataset_root_dir, frames_dir))

    total_frames = len(frames)

    for frame in frames:
        remaining_frames = total_frames - (len(aurora_frames) + len(no_aurora_frames))
        remaining_frames_text = f"Remaining frames: {remaining_frames}"
        while True:
            frame_path = os.path.join(dataset_root_dir, frames_dir, frame)
            img = cv2.imread(frame_path, cv2.IMREAD_ANYCOLOR)
            img = cv2.putText(img, texts["close"], (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            img = cv2.putText(img, texts["yes"], (0, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            img = cv2.putText(img, texts["no"], (0, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            img = cv2.putText(img, remaining_frames_text, (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
            cv2.imshow("frames", img)
            key = cv2.waitKey(0)

            match key:
                case 27:
                    print('esc pressed, closing all windows')
                    cv2.destroyAllWindows()
                    return
                case 121:
                    print(f'''{frame} is valid''')
                    aurora_frames.append(frame)
                case 110:
                    print(f'''{frame} is not valid''')
                    no_aurora_frames.append(frame)
                case _:
                    print(f'''{frame} was skipped''')
                    skipped_frames.append(frame)

            break


def resize_and_save(src: str, dst_dir: str):
    original = cv2.imread(src)
    resized = cv2.resize(original, RESIZED_DIMENSION)
    name = os.path.basename(src)
    dst = os.path.join(dst_dir, name)
    cv2.imwrite(dst, resized)


def move_frames(dataset_root_dir: str, frames_dir: str, frames: list, destination: str):
    for frame in frames:
        frame_path = os.path.join(dataset_root_dir, frames_dir, frame)
        dest_dir = os.path.join(dataset_root_dir, "orig", destination)
        resized_dest_dir = os.path.join(dataset_root_dir, "resized", destination)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        if not os.path.exists(resized_dest_dir):
            os.makedirs(resized_dest_dir)

        resize_and_save(frame_path, resized_dest_dir)
        shutil.move(frame_path, os.path.join(dest_dir, frame))


def main():
    parser = argparse.ArgumentParser(
            description='Tool to assist with image sorting before using them for training'
        )

    parser.add_argument(
            '-r',
            '--root',
            '--dataset-root',
            dest='dataset_root',
            default='dataset',
            help='The root directory of the dataset',
            type=str,
        )

    parser.add_argument(
            '-f',
            '--frames',
            '--frames-dir',
            dest='frames_dir',
            default='unclassified',
            help='Name of the directory containing the frames to review',
            type=str,
        )

    args = parser.parse_args()
    dataset_root_dir = args.dataset_root
    frames_dir = os.path.basename(args.frames_dir)

    review_frames(dataset_root_dir, frames_dir)

    if len(aurora_frames) > 0:
        print(f'''Moving frames classifed as aurora to {dirs["yes"]}''')
        move_frames(dataset_root_dir, frames_dir, aurora_frames, dirs["yes"])

    if len(no_aurora_frames) > 0:
        print(f'''Moving frames without aurora to {dirs["no"]}''')
        move_frames(dataset_root_dir, frames_dir, no_aurora_frames, dirs["no"])

    cv2.destroyAllWindows()
    if len(skipped_frames) > 0:
        print(f'''WARNING: {len(skipped_frames)} frames were skipped!''')

    sys.exit(0)


if __name__ == "__main__":
    main()
