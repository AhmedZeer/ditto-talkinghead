# video_stream_writer.py
import subprocess
import imageio
import numpy as np
import os
import sys
import threading

class VideoStreamWriter:
    def __init__(self, stream_command, fps=25, width=640, height=480, pix_fmt="yuv420p", crf=18, **kwargs):
        """
        Initialize the VideoStreamWriter.

        :param stream_command: The FFmpeg command to execute for streaming.
        :param fps: Frames per second.
        :param width: Width of the video frames.
        :param height: Height of the video frames.
        :param pix_fmt: Pixel format.
        :param crf: Constant Rate Factor for quality.
        :param kwargs: Additional FFmpeg parameters.
        """
        self.fps = fps
        self.width = width
        self.height = height
        self.pix_fmt = pix_fmt
        self.crf = crf
        self.stream_command = stream_command

        # Start FFmpeg subprocess
        self.process = subprocess.Popen(
            self.stream_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

        if self.process.stdin is None:
            raise ValueError("Failed to open FFmpeg subprocess stdin.")

        # Initialize ImageIO writer to write to FFmpeg's stdin
        self.writer = imageio.get_writer(
            self.process.stdin,
            format='ffmpeg',
            mode='I',
            fps=fps,
            codec='libx264',
            pixelformat=self.pix_fmt,
            ffmpeg_params=[
                '-preset', 'veryfast',
                '-crf', str(self.crf),
                '-f', 'flv'  # Default format for RTMP
            ]
        )

        # Thread lock for thread-safe operations
        self.lock = threading.Lock()

    def __call__(self, img, fmt="rgb"):
        """
        Append a frame to the stream.

        :param img: The image/frame to append as a NumPy array.
        :param fmt: The format of the image ('rgb' or 'bgr').
        """
        with self.lock:
            if fmt == "bgr":
                frame = img[..., ::-1]  # Convert BGR to RGB
            else:
                frame = img
            self.writer.append_data(frame)
            print("write")

    def close(self):
        """
        Close the writer and the FFmpeg subprocess.
        """
        with self.lock:
            self.writer.close()
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()

    def is_alive(self):
        """
        Check if the FFmpeg subprocess is still running.

        :return: Boolean indicating if FFmpeg is alive.
        """
        return self.process.poll() is None
