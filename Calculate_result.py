import pandas as pd
import numpy as np
import csv

dataPath = ['../dataset/paper_dataset/10s_path_gt.csv',
            '../dataset/PURE_raw_dataset/',
            '../dataset/PURE_784k/', '../dataset/PURE_979k/','../dataset/PURE_1223k/', '../dataset/PURE_1467k/']
dataset = ['CCUHR', 'PURE_image', 'PURE_video']
bitrate = ['784k', '979k', '1223k', '1467k']
adjust = "_p4_MIL_all_"

#def get_result_with_suc_est(bpms, gts, aes):



if __name__ == '__main__':

    #path = "../result/DAO/"+dataset[2]+adjust+bitrate[3]+"_60s.csv"
    path = "../result/DAO/PURE_video_p4_asf_twoline_on_CSRT_1467k/PURE_video_p4_asf_twoline_on_CSRT_1467k_10s.csv"

    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)

        total_AE = np.array([])
        total_low_AE = np.array([])
        i = 0
        success_estimate_n = 0
        for row in reader:
            if i > 0:
                length = len(row)
                bpms_length = int((length-3)/3)

                # split BPMs, GTs, AEs
                bpms = np.array(row[2: 2+bpms_length], dtype=float)
                gts = np.array(row[2+bpms_length: 2+bpms_length*2], dtype=float)
                aes = np.array(row[2+bpms_length*2: 2+bpms_length*3], dtype=float)


                # BPM > 0
                not_zero_bpms_idx = np.where(bpms != 0)
                true_aes = aes[not_zero_bpms_idx].copy()


                # find which GT bigger then 100
                high_gt_idx = np.where(gts >= 100)
                # Let BPM equal 0 when GT bigger then 100
                low_gt_bpms = bpms.copy()
                low_gt_bpms[high_gt_idx] = 0
                # Don't count AE which GT bigger then 100
                not_zero_bpms_idx = np.where(low_gt_bpms != 0)
                # GT < 100 or BPM > 0
                true_low_aes = aes[not_zero_bpms_idx].copy()

                total_AE = np.append(total_AE, true_aes)
                total_low_AE = np.append(total_low_AE, true_low_aes)

            i += 1

        # calculate all AEs' mean which bpm success estimate
        total_AE = total_AE.flatten()
        MAE = np.mean(total_AE)

        # calculate all AEs' mean which bpm success estimate and GT bigger then 100
        total_low_AE = total_low_AE.flatten()
        total_AE_n = len(total_low_AE)
        low_MAE = np.mean(total_low_AE)
        print("length of total AEs : ", total_AE_n)
        print('MAE : ', MAE)
        print('low MAE : ', low_MAE)
