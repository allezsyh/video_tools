import sys
import os.path as osp
ProjectRoot = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(ProjectRoot)
from video_process.video_stitcher import stitcher_videos


if __name__ == '__main__':
    '''
    运行说明：python
    '''
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--video_files', type=str, nargs='+', default=[
        '/mnt/ssd/projects/PythonProjects/CapsuleManager/data/camera_fm_wide1.mp4',
        '/mnt/ssd/projects/PythonProjects/CapsuleManager/data/camera_fm_wide2.mp4',
        '/mnt/ssd/projects/PythonProjects/CapsuleManager/data/camera_fm_wide3.mp4',
        '/mnt/ssd/projects/PythonProjects/CapsuleManager/data/camera_fm_wide4.mp4',
    ])
    parser.add_argument('--video_tags', type=str, nargs='+', default=[
        '/mnt/ssd/projects/PythonProjects/CapsuleManager/data/camera_fm_wide1.mp4',
        '/mnt/ssd/projects/PythonProjects/CapsuleManager/data/camera_fm_wide2.mp4',
        '/mnt/ssd/projects/PythonProjects/CapsuleManager/data/camera_fm_wide3.mp4',
        '/mnt/ssd/projects/PythonProjects/CapsuleManager/data/camera_fm_wide4.mp4',
    ])
    parser.add_argument('--output_file', type=str, default='output/output_video.mp4')

    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--fps', type=float, default=None)
    args = parser.parse_args()

    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    # ------------------------------
    stitcher_videos(args.video_files, args.output_file, video_tags=args.video_tags, target_width=args.width, target_height=args.height, target_fps=args.fps)