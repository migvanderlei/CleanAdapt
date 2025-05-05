import os
import shutil
import glob
import asyncio

def group_frames_by_video(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = glob.glob(os.path.join(base_dir, '**', '*.jpg'), recursive=True)

    for file_path in jpg_files:
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

        shutil.move(file_path, new_file_path)
        print(f"Moved: {file_path} -> {new_file_path}")

def clean_empty_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            print(f"Removed empty folder: {dirpath}")

def move_hmdb51_folders(source_dir, target_dir):
    """
    Move HMDB51 extracted frame folders from rawframes/rawframes/class_name/video_name/
    directly into target_dir/video_name/, using glob.
    """
    os.makedirs(target_dir, exist_ok=True)

    # Use glob to find all video folders two levels deep: class_name/video_name/
    video_folders = glob.glob(os.path.join(source_dir, '*', '*'))

    for video_path in video_folders:
        if not os.path.isdir(video_path):
            continue

        video_name = os.path.basename(video_path).replace('img_', 'frame_')
        dest_path = os.path.join(target_dir, video_name)

        shutil.move(video_path, dest_path)
        print(f"Moved: {video_path} -> {dest_path}")

async def download_and_unzip(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {dataset} into {output_dir}...")
    cmd = [
        "kaggle", "datasets", "download", "-d", dataset, "-p", output_dir, "--unzip"
    ]
    process = await asyncio.create_subprocess_exec(*cmd)
    await process.communicate()
    print(f"Finished downloading and unzipping {dataset}")

async def main():
    download_dir = '/home/miguel/Workspace/personal/CleanAdapt/data/raw'
    rgb_dir = '/home/miguel/Workspace/personal/CleanAdapt/data/rgb'

    hmdb51_download_dir = os.path.join(download_dir, 'hmdb51-frames')
    ucf101_download_dir = os.path.join(download_dir, 'ucf101-frames')

    hmdb51_rgb_dir = os.path.join(rgb_dir, 'hmdb51')
    ucf101_rgb_dir = os.path.join(rgb_dir, 'ucf101')

    await asyncio.gather(
        download_and_unzip('jizeyong/hmdb51', hmdb51_download_dir),
        download_and_unzip('pevogam/ucf101-frames', ucf101_download_dir),
    )

    print("Post-processing HMDB51...")
    move_hmdb51_folders(hmdb51_download_dir, hmdb51_rgb_dir)
    clean_empty_folders(hmdb51_download_dir)

    print("Post-processing UCF101...")
    tmp_ucf101_dir = os.path.join(ucf101_download_dir, 'ucf101-frames')
    group_frames_by_video(tmp_ucf101_dir, ucf101_rgb_dir)
    clean_empty_folders(tmp_ucf101_dir)

    if os.path.exists(tmp_ucf101_dir):
        shutil.rmtree(tmp_ucf101_dir)
        print(f"Removed tmp directory: {tmp_ucf101_dir}")

if __name__ == "__main__":
    asyncio.run(main())
