from script_generator.state.app_state import AppState
from script_generator.video.data_classes.video_info import get_cropped_dimensions
from script_generator.video.ffmpeg.filters import get_video_filters
from script_generator.video.ffmpeg.hwaccel import get_hwaccel_read_args, supports_scale_cuda


def get_ffmpeg_read_cmd(state: AppState, frame_start: int | None, output="-", disable_opengl=False):
    video = state.video_info
    width, height = get_cropped_dimensions(video)
    vf = get_video_filters(video, state.video_reader, state.ffmpeg_hwaccel, width, height, disable_opengl)
    start_time = (frame_start / video.fps) * 1000
    frame_size = width * height * 3  # Size of one frame in bytes

    # Regular ffmpeg command
    if video.bit_depth == 8 or state.ffmpeg_hwaccel != "cuda":

        # Get supported hardware acceleration backends
        hwaccel_read = get_hwaccel_read_args(state)

        video_filter = ["-vf", vf] if vf else []
        if state.ffmpeg_hwaccel == "vaapi":
            # VAAPI requires specific pixel formats and filters
            video_filter = ["-vf", f"{vf},format=nv12,hwupload"] if vf else ["-vf", "format=nv12,hwupload"]

        if supports_scale_cuda(state):
            video_filter = ["-noautoscale"] + video_filter  # explicitly tell ffmpeg that scaling is done by cuda

        return [
            state.ffmpeg_path,
            *hwaccel_read,
            "-nostats", "-loglevel", "warning",
            "-ss", str(start_time / 1000),  # Seek to start time in seconds
            "-i", video.path,
            "-an",  # Disable audio processing
            *video_filter,
            "-f", "rawvideo", "-pix_fmt", "bgr24",  # cv2 requires bgr (over rgb) and Yolo expects bgr images when using numpy frames (converts them internally)
            "-threads", "0", # all threads
            output
        ], frame_size, width, height

    # 10 bit ffmpeg cuda pipe command
    else:

        return [
            [
                state.ffmpeg_path,
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-nostats", "-loglevel", "warning",
                "-ss", str(start_time / 1000),  # Seek to start time in seconds
                "-i", video.path,
                "-an",  # Disable audio processing
                # we output to 1000x1000 so the v360 filter has enough raw input data
                "-vf", f"crop={int(video.height)}:{int(video.height)}:0:0,scale_cuda=1000:1000",
                "-c:v", "hevc_nvenc", "-preset", "fast", "-qp", "0",
                "-f", "matroska",
                "-"
            ],
            [
                state.ffmpeg_path,
                "-hwaccel", "cuda",
                "-i", "-",
                "-vf", vf,
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-threads", "0",  # all threads
                output
            ]
        ], frame_size, width, height
