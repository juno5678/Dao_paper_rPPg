import csv
import cv2
import os
import json
from HR_Estimator_Single import DAO_HRE
from video_realsense_file import Video
import sys
import pandas as pd
import numpy as np
import argparse
import warnings


def scan_all_data(data_path, root_directory, search_file, level=2):
    for item in os.listdir(root_directory):
        full_path = os.path.join(root_directory, item)
        # search sub dir
        if os.path.isdir(full_path) and level >= 1:
            # find .avi file
            if search_file == '.avi' or search_file == '.json':
                video_file = [video for video in os.listdir(full_path) if video.endswith(search_file)]
                if video_file:
                    video_path = os.path.join(full_path, video_file[0])
                    data_path.append(video_path)
            # find image sequence dir
            elif search_file == '.png':
                img_file = [image for image in os.listdir(full_path) if image.endswith(search_file)]
                #print(img_file)
                if img_file:
                    data_path.append(full_path)
                    #print(full_path)
                else:
                    scan_all_data(data_path, full_path, search_file, level - 1)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Dao's rPPG estimate heart rate")
    parser.add_argument(
        "-d", "--dataPath", type=str, required=True, help="video clip path"
    )
    parser.add_argument(
        "--save",
        action='store_true',
        default=False,
        help="save result or not (default : %(default)s)",
    )
    parser.add_argument(
        "--savePath",
        default="./result/result.csv",
        type=str,
        help="the path of result (default : %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--process_mode",
        default=0,
        type=int,
        help="RGB : 0, RGB + NIR : 1, CIEa + CIEb + NIR :2 , (default : %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output_mode",
        default=2,
        type=int,
        help="estimate single : 0, estimate sequence : 1, estimate fragment : 2 , (default : %(default)s)",
    )
    parser.add_argument(
        "-i",
        "--input_mode",
        default=0,
        type=int,
        help=" rgb video : 0, rgb image : 1, rgb camera : 2"
             " realsense video : 3, realsense camera : 4, (default : %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--second",
        default=10,
        type=int,
        help="time length used for estimate HR (default : %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--file_num",
        default=0,
        type=int,
        help="number of file (default : %(default)'s file)",
    )
    parser.add_argument(
        "-w",
        "--without_wd",
        action='store_true',
        default=False,
        help="wavelet decompose or not (default : %(default)s)",
    )
    args = parser.parse_args(argv)
    return args


def get_all_gt(gt_list, gt_path, input_max_length):
    count = 0
    with open(gt_path) as file:
        data = json.loads(file.read())
        data = data['/FullPackage']
        for p in data:
            gt_list.append(p['Value']['pulseRate'])
            count += 1
            if count == input_max_length:
                break


def get_avg_gt(gt_list, gt_avg_list, window_size, interval):
    window_size *= 30
    max_length = len(gt_list)
    if max_length % interval != 0:
        estimate_end_frame = max_length-window_size+1-interval
    else:
        estimate_end_frame = max_length-window_size-interval

    for i in range(0, estimate_end_frame, interval):
        #print(i)
        gt_avg_list.append(np.mean(gt_list[i:i+window_size]))
    print(" len of gt avg : ", len(gt_avg_list))

if __name__ == '__main__':

    args = parse_args(sys.argv[1:])
    #root_directory = sys.argv[1]
    #result_path = sys.argv[2]
    root_directory = args.dataPath
    if args.save:
        result_path = args.savePath
    file_num = args.file_num
    data_path = []
    gt_path = []
    gt_data = []
    if args.input_mode == 0:
        scan_all_data(data_path, root_directory, '.avi', 2)
    elif args.input_mode == 1:
        scan_all_data(data_path, root_directory, '.png', 2)
    elif args.input_mode == 3:
        # open .csv file include path and ground truth
        path_file = args.dataPath
        with open(path_file, newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            #print(rows)
            for row in rows:
                gt_data.append(float(row['GT']))
                root_directory = os.path.dirname(path_file)
                #print(dataset_root_path, row['data path'])
                data_path.append(os.path.join(root_directory, row['data path']))

    if args.input_mode == 0 or args.input_mode == 1:
        scan_all_data(gt_path, root_directory, '.json', 2)
    #print(len(data_path))

    HR_Estimator = DAO_HRE('')
    HR_Estimator.set_input(args.input_mode)
    HR_Estimator.set_output(args.output_mode)
    HR_Estimator.set_process_mode(args.process_mode)
    HR_Estimator.set_length(args.second)
    HR_Estimator.process.without_wd = args.without_wd
    #print(data_path)
    # for i in range(0, len(data_path)):
    for i in range(file_num, file_num+1):
    #for i in range(52, 53):
    #for i in range(0, 12):
    #for item in data_path:
        #print(i)
        gt_all_data = []
        gt_avg_data = []
        bpm_list = []
        fps_list = []
        item = data_path[i]

        print(item)
        HR_Estimator.set_data_path(item)
        bpm, fps = HR_Estimator.run()
        if args.output_mode == 0:
            bpm_list.append(bpm)
        fps_list.append(fps)
        HR_Estimator.reset()

        # type average information
        #bpm_list.append(np.mean(bpm_list))
        #fps_list.append(np.mean(fps_list))

        #bpm = [10]*51
        #fps_list = [1]

        if args.output_mode == 0:
            if args.input_mode == 3:
                #print('gt : ', gt_data[i], 'bpm : ', bpm)
                absolute_error = abs(gt_data[i]-bpm)
                data = {"data path": [item], "fps": fps_list, "bpm": bpm_list, "MAE": [absolute_error]}
            else:
                data = {"data path": [item], "fps": fps_list, "bpm": bpm_list}
        else:
            bpm_n_key = []
            for j in range(len(bpm)):
                bpm_n = "bpm " + str(j)
                bpm_n_key.append(bpm_n)
            bpm_data = {k: [v] for k, v in zip(bpm_n_key, bpm)}
            data = {"data path": [item], "fps": fps_list}
            data.update(bpm_data)

            # get ground truth
            get_all_gt(gt_all_data, gt_path[i], HR_Estimator.max_input_size)

            if args.output_mode == 1:
                get_avg_gt(gt_all_data, gt_avg_data, args.second, 1)
            else:
                get_avg_gt(gt_all_data, gt_avg_data, args.second, 6)

            gt_n_key = []
            for k in range(len(gt_avg_data)):
                #print(len(gt_avg_data))
                gt_n = "gt " + str(k)
                gt_n_key.append(gt_n)
            gt_data = {k: [v] for k, v in zip(gt_n_key, gt_avg_data)}
            data.update(gt_data)

            # calculate mean absolute error (MAE)
            absolute_error_list = []
            ae_n_key = []
            for l in range(len(gt_avg_data)):
                ae_n = "AE " + str(l)
                ae_n_key.append(ae_n)
                # encounter SVD didn't converge bug
                if bpm[l] == 0:
                    absolute_error = 0
                else:
                    absolute_error = abs(gt_avg_data[l] - bpm[l])
                absolute_error_list.append(absolute_error)
            ae_data = {k: [v] for k, v in zip(ae_n_key, absolute_error_list)}

            data.update(ae_data)

            data['MAE'] = [np.mean(absolute_error_list)]

            #print(data)


        df = pd.DataFrame(data)
        #with open(result_path, "a") as f:
        #    df.to_csv(f, header=True, index=False)
        if i == 0:
            df.to_csv(result_path, header=True, index=False, mode='a')
        else:
            df.to_csv(result_path, header=False, index=False, mode='a')

