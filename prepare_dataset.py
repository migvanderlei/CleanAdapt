import os
import shutil
import glob
import argparse
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi


def group_frames_by_video(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    jpg_files = glob.glob(os.path.join(base_dir, '**', '*.jpg'), recursive=True)
    print("Source:", base_dir)
    print("Found:", len(jpg_files), "frames")
    for file_path in tqdm(jpg_files, desc="Processing HMDB101 files"):
        file_name = os.path.basename(file_path)
        parts = file_name.split('-')
        if len(parts) < 2:
            print(f"Skipping unexpected file name format: {file_name}")
            continue

        video_prefix = '-'.join(parts[:-1])
        frame_id = parts[-1].split('.')[0]

        video_dir = os.path.join(output_dir, video_prefix)
        os.makedirs(video_dir, exist_ok=True)

        new_file_name = f"frame_{int(frame_id):04d}.jpg"
        new_file_path = os.path.join(video_dir, new_file_name)

        shutil.copy(file_path, new_file_path)


def clean_empty_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            print(f"Removed empty folder: {dirpath}")


def move_hmdb51_folders(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    video_folders = glob.glob(os.path.join(source_dir, '**', '*.jpg'), recursive=True)
    print(source_dir)
    print("Found", len(video_folders), "frames")

    for video_path in tqdm(video_folders, desc="Processing HMDB51 files"):

        video_name = os.path.basename(video_path).replace('img_', 'frame_')
        dest_path = os.path.join(target_dir, video_name)

        shutil.copy(video_path, dest_path)


def download_and_unzip(dataset, output_dir):
    marker_path = os.path.join(output_dir, '.extracted')
    if os.path.exists(marker_path):
        print(f"{dataset} already downloaded and extracted. Skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print(f"Downloading and extracting {dataset} into {output_dir}...")
    api.dataset_download_files(dataset, path=output_dir, unzip=True)

    with open(marker_path, 'w') as f:
        f.write("done")
    print(f"Finished downloading and extracting {dataset}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process HMDB51 and UCF101 datasets."
    )
    parser.add_argument("--download_dir", type=str, required=True, help="Path to download raw datasets.")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Path to output RGB frame folders.")
    parser.add_argument("--skip_download", action="store_true", help="Skip downloading datasets.")
    parser.add_argument("--skip_processing", action="store_true", help="Skip post-processing datasets.")
    args = parser.parse_args()

    hmdb51_dataset = 'jizeyong/hmdb51'
    ucf101_dataset = 'pevogam/ucf101-frames'

    hmdb51_download_dir = os.path.join(args.download_dir, 'hmdb51-frames')
    ucf101_download_dir = os.path.join(args.download_dir, 'ucf101-frames')

    hmdb51_rgb_dir = os.path.join(args.rgb_dir, 'hmdb51')
    ucf101_rgb_dir = os.path.join(args.rgb_dir, 'ucf101')

    if not args.skip_download:
        download_and_unzip(hmdb51_dataset, hmdb51_download_dir)
        download_and_unzip(ucf101_dataset, ucf101_download_dir)

    if not args.skip_processing:
        print("Post-processing HMDB51...")
        move_hmdb51_folders(hmdb51_download_dir+"/rawframes", hmdb51_rgb_dir)

        print("Post-processing UCF101...")
        group_frames_by_video(ucf101_download_dir, ucf101_rgb_dir)



if __name__ == "__main__":
    main()
