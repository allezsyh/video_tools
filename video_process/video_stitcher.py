import cv2
import numpy as np
from tqdm import tqdm

def resize_frame_with_border(frame, target_size):
    """
    调整帧的大小以适应目标大小，同时保持原始长宽比，并在边界像素不够的地方补充黑色。

    :param frame: 输入帧
    :param target_size: 目标大小 (宽, 高)
    :return: 调整后的帧
    """
    original_height, original_width = frame.shape[:2]
    target_width, target_height = target_size

    # 计算比例
    ratio_width = target_width / original_width
    ratio_height = target_height / original_height
    ratio = min(ratio_width, ratio_height)

    # 计算新的尺寸
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # 调整帧大小
    resized_frame = cv2.resize(frame, (new_width, new_height))
    # cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)

    # 计算边界的大小
    top = (target_height - new_height) // 2
    bottom = target_height - new_height - top
    left = (target_width - new_width) // 2
    right = target_width - new_width - left

    # 添加边界
    bordered_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return bordered_frame


def stitcher_videos(video_paths, output_video, video_tags=None, grid_size=(2, 2), target_width=None, target_height=None, target_fps=None):
    '''
    grid_size: (width-cnt, height-cnt)
    '''
    # 读取视频并获取视频参数
    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    widths, heights, fps, frame_nums = zip(*((
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        cap.get(cv2.CAP_PROP_FPS),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ) for cap in caps))
    # 输入视频的最大范围
    width_max = max(widths)
    height_max = max(heights)
    frame_num = max(frame_nums)

    # 计算输出视频的尺寸
    width = target_width or width_max * grid_size[0]
    height = target_height or height_max * grid_size[1]
    fps = target_fps or min(fps)

    # 均等分每个子视频的尺寸
    child_width = int(width / grid_size[0])
    child_height = int(height / grid_size[1])
    print(f"out video: {width} x {height}\t fps: {fps}")
    print(f"child video: {child_width} x {child_height}\t grid_size: {grid_size}")

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    frame_num = 20 # debug
    # 播放所有视频
    for _ in tqdm(range(frame_num)):
        frames = []
        frame_end_cnt = 0
        for cap_idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # 如果视频结束，创建黑幕
                frame = np.zeros((child_height, child_width, 3), dtype=np.uint8)
                frame_end_cnt += 1
            else:
                frame = resize_frame_with_border(frame, (child_width, child_height))
                if video_tags and video_tags[cap_idx]:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, video_tags[cap_idx], (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            frames.append(frame)

        # 视频全部播放完成
        if frame_end_cnt == len(caps):
            break

        # 将视频帧拼接成网格
        combined_frame = np.vstack([
            np.hstack(frames[i * grid_size[0]:(i + 1) * grid_size[0]])  # 每一行水平拼接 (width-cnt)
            for i in range(grid_size[1])  # 总共grid_size[1]行 (height-cnt)
        ])

        # 将拼接后的帧写入到输出视频
        out.write(combined_frame)

    # 释放资源
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"out video: {output_video}")


if __name__ == '__main__':
    '''
    运行说明：python
    '''
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--video_files', type=str, nargs='+', default=[])
    parser.add_argument('--video_tags', type=str, nargs='+', default=[])
    parser.add_argument('--output_file', type=str, default='output_video.mp4')

    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--fps', type=float, default=None)
    args = parser.parse_args()

    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))

    # ------------------------------
    stitcher_videos(args.video_files, args.output_file, video_tags=args.video_tags, target_width=args.width, target_height=args.height, target_fps=args.fps)