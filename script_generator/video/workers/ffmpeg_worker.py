import subprocess

import numpy as np

from script_generator.debug.errors import FFMpegError
from script_generator.debug.logger import log_vid
from script_generator.tasks.workers.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes
from script_generator.video.analyse_frame_task import AnalyzeFrameTask
from script_generator.video.ffmpeg.commands import get_ffmpeg_read_cmd


class VideoWorker(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.VIDEO
    process = None
    read_frames = True

    def task_logic(self):
        self.process_pre = None
        self.process = None
        self.read_frames = True

        cmd, frame_size, width, height = get_ffmpeg_read_cmd(
            self.state,
            self.state.frame_start
        )
        ffmpeg_command_str = ' '.join(cmd[0]) + " | " + ' '.join(cmd[1]) if isinstance(cmd[0], list) else ' '.join(cmd)
        log_vid.info(f"FFMPEG executing command: {ffmpeg_command_str}")

        if self.state.video_info.bit_depth == 8 or self.state.ffmpeg_hwaccel != "cuda":
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.process_pre = subprocess.Popen(cmd[0], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.process = subprocess.Popen(cmd[1], stdin=self.process_pre.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.process_pre.stdout.close()

        current_frame = self.state.frame_start

        try:
            while self.read_frames:
                in_bytes = self.process.stdout.read(frame_size)
                if not in_bytes:
                    if current_frame == self.state.frame_start:
                        if self.process_pre:
                            error_output = self.process_pre.stderr.read().decode('utf-8', errors='replace')
                            log_vid.error(f"FFMPEG could not read frames from this video\nFFMPEG command:\n{ffmpeg_command_str}\nFFMPEG ERROR:\n{error_output}")
                        error_output = self.process.stderr.read().decode('utf-8', errors='replace')
                        log_vid.error(f"FFMPEG could not read frames from this video\nFFMPEG command:\n{ffmpeg_command_str}\nFFMPEG ERROR:\n{error_output}")
                        raise FFMpegError(f"FFMPEG could not read frames from this video. See the log for details.")
                    else:
                        log_vid.info("FFMPEG received last frame")
                    break

                task = AnalyzeFrameTask(frame_pos=current_frame)
                frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])

                if self.state.video_reader == "FFmpeg":
                    task.rendered_frame = frame
                else:
                    task.preprocessed_frame = frame

                task.end(str(self.process_type))

                self.finish_task(task)
                current_frame += 1

                # For the ffmpeg pipe we need to detect the last frame and close manually
                if current_frame >= self.state.video_info.total_frames - 1:
                    log_vid.info("FFMPEG received last frame")
                    break

        except Exception as e:
            # Suppress any errors when the thread is force closed
            if self.read_frames:
                log_vid.error(f"Error reading frame: {e}")
                raise e

        finally:
            self.stop_process()
            self.output_queue.put(None)
            self.release()

    def release(self):
        log_vid.debug("Stopping FFmpeg reader")
        self.read_frames = False
        self.stop_process()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        if self.process_pre:
            self.process_pre.terminate()
            try:
                self.process_pre.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process_pre.kill()
            self.process_pre = None
