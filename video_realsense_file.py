import pyrealsense2 as rs
import numpy as np


class Video(object):
    def __init__(self):
        self.points = rs.points()
        self.pipeline = rs.pipeline()
        self.dirname = ""
        self.config = rs.config()

    def start(self):
        print("Starting the video" + self.dirname)
        if self.dirname == "":
            print("Invalid Filename!")
            return
        self.config.enable_device_from_file(self.dirname, repeat_playback=False)
        profile = self.pipeline.start(self.config)
        profile.get_device().as_playback().set_real_time(False)

    def stop(self):
        self.pipeline.stop()
        print("Stopped the video")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        nir_frame = frames.get_infrared_frame()
        color_out = np.asanyarray(color_frame.get_data())
        nir_out = np.asanyarray(nir_frame.get_data(), dtype='uint8')
        return color_out, nir_out
