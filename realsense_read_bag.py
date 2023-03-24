import pyrealsense2 as rs
import numpy as np
import  cv2
import argparse
import os.path

parser = argparse.ArgumentParser(description="Read recorded bag file and displat depth ....")
parser.add_argument("-i","--input",type=str,help="Path to the bag file")
args = parser.parse_args()
args.input = r'D:\CCU\HR_Estimator\dataset\emotion_10s\521_emotion_10s.bag'

if not args.input:
    print("No input parameter have been given.")
    print("For help type --help")
    exit()

if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
try:
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config,args.input,repeat_playback=False)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)

    pipeline.start(config)

    device = pipeline.get_active_profile().get_device()
    playback = device.as_playback()
    playback.set_real_time(False)

    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

    colorizer = rs.colorizer()

    number = 0
    timestamp_tmp = 0

    while True:
        frames = pipeline.wait_for_frames()
        number = number + 1
        print(number, ' ', frames.timestamp, frames.timestamp - timestamp_tmp)
        timestamp_tmp = frames.timestamp

        depth_frame = frames.get_depth_frame()

        depth_color_frame = colorizer.colorize(depth_frame)

        depth_color_frame = np.asanyarray(depth_color_frame.get_data())

        cv2.imshow("Depth Stream", depth_color_frame)
        key = cv2.waitKey(1)
except RuntimeError:
    print("There are no more frames left in the .bag file!")
    cv2.destroyAllWindows()

finally:
    pass

