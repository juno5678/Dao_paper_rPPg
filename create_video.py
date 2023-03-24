import cv2
import os
import sys
from moviepy.editor import *
import shutil

#def copy_gt(gt_path, save_path):


def create_video(image_folder, save_path, video_file):
    images_file = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    #print("image folder :", image_folder)
    #print("video file :", video_file)
    save_file_path = os.path.join(save_path, video_file)
    save_folder_path = os.path.split(save_file_path)[0]
    print("save folder path : ", save_folder_path)
    print("save video file :", save_file_path)
    if not os.path.isdir(save_folder_path):
        os.mkdir(save_folder_path)
    images_file.sort()
    fps = 30

    images = [ImageClip(os.path.join(image_folder, img)).set_duration(1/fps) for img in images_file]

    video = concatenate_videoclips(images, method="compose")
    video.write_videofile(save_file_path, fps=fps, codec="libx264", bitrate="18432k", preset="medium")
    #video.write_videofile(video_file, fps=fps, codec="rawvideo", bitrate="36864k", preset="medium")

    #frame = cv2.imread(os.path.join(image_folder, images[0]))
    #height, width, layers = frame.shape

    #video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"XVID"), 30, (width, height))

    #for image in images:
    #    video.write(cv2.imread(os.path.join(image_folder, image)))

    #cv2.destroyAllWindows()
    #video.release()

def convert_folder_to_video(root_directory, save_path,  level=2):
    for item in os.listdir(root_directory):
        full_path = os.path.join(root_directory, item)
        #print(full_path)
        if os.path.isdir(full_path) and level >= 1:
            images = [img for img in os.listdir(full_path) if img.endswith(".png")]
            ground_truth = [gt for gt in os.listdir(full_path) if gt.endswith(".json")]
            if images:
                #print(images)
                video_file = os.path.join(root_directory, f"{item}.avi")
                create_video(full_path, save_path, video_file)
            else:
                convert_folder_to_video(full_path, save_path, level - 1)
            if ground_truth:
                gt_file = os.path.join(root_directory, ground_truth[0])
                shutil.copy(os.path.join(full_path, gt_file), os.path.join(save_path, full_path))
                print("gt : ", os.path.join(full_path, gt_file))
                print("save path : ", os.path.join(save_path, full_path))

root_directory = sys.argv[1]
save_path = sys.argv[2]
print("save path : ", save_path)
convert_folder_to_video(root_directory, save_path)