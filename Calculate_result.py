import pandas as pd
import numpy as np
import csv





def split_data(row, second, mode):

    length = len(row)
    bpms_length = int((length - 3) / 3)

    # split BPMs, GTs, AEs
    fps = np.array(row[1: 2], dtype=float)
    bpms = np.array(row[2: 2 + bpms_length], dtype=float)
    gts = np.array(row[2 + bpms_length: 2 + bpms_length * 2], dtype=float)
    aes = np.array(row[2 + bpms_length * 2: 2 + bpms_length * 3], dtype=float)

    if mode == 'all_video':
        pass
    elif mode == 'first_60s':
        n = int((60*30 - second * 30) / 6 + 1)
        bpms = bpms[:n]
        gts = gts[:n]
        aes = aes[:n]
    elif mode == 'first_62s':
        n = int((60 * 30 - second * 30) / 6 + 10 + 1)
        bpms = bpms[:n]
        gts = gts[:n]
        aes = aes[:n]

    return fps, bpms, gts, aes

if __name__ == '__main__':

    dataPath = ['../dataset/paper_dataset/10s_path_gt.csv',
                '../dataset/PURE_raw_dataset/',
                '../dataset/PURE_784k/', '../dataset/PURE_979k/', '../dataset/PURE_1223k/', '../dataset/PURE_1467k/']
    dataset = ['CCUHR', 'PURE_image', 'PURE_video']
    bitrate = ['784k', '979k', '1223k', '1467k']
    mode = ['all_video', 'first_60s', 'first_62s']
    adjust = "_p4_MIL_all_"
    second = [60, 30, 20, 10]

    #for j in range(len(dataPath) - 1, len(dataPath) - 2, -1):
    for j in range(1, len(dataPath)):
        for m in mode:
            # print(j)
            for s in second:

                if j == 1:
                    path = "../result/DAO/" + dataset[j] + adjust + str(s) + "s.csv"
                    save_path = "../result/DAO/unified_result/" + dataset[j] + adjust + "unified.csv"
                elif j > 1:
                    path = "../result/DAO/" + dataset[2] + adjust + bitrate[j - 2] + "_" + str(s) + "s.csv"
                    save_path = "../result/DAO/unified_result/" + dataset[2] + adjust + bitrate[j - 2] + "_unified.csv"
                #path = "../result/DAO/"+dataset[2]+adjust+bitrate[3]+"_60s.csv"
                #path = "../result/DAO/PURE_video_p4_asf_twoline_on_CSRT_1467k/PURE_video_p4_asf_twoline_on_CSRT_1467k_10s.csv"

                print(path)
                with open(path, newline='') as csvfile:
                    reader = csv.reader(csvfile)

                    total_AE = np.array([])
                    total_fps = np.array([])
                    total_low_AE = np.array([])
                    i = 0

                    for row in reader:
                        if i > 0:

                            fps, bpms, gts, aes = split_data(row, s, m)

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

                            total_fps = np.append(total_fps, fps)
                            total_AE = np.append(total_AE, true_aes)
                            total_low_AE = np.append(total_low_AE, true_low_aes)

                        i += 1

                    # calculate all AEs' mean which bpm success estimate
                    total_fps = total_fps.flatten()
                    FPS = np.mean(total_fps)

                    # calculate all AEs' mean which bpm success estimate
                    total_AE = total_AE.flatten()
                    total_AE_n = len(total_AE)
                    success_AE = len(np.where(total_AE < 5)[0])
                    success_rate = success_AE / total_AE_n
                    MAE = np.mean(total_AE)
                    RMSE = np.sqrt(np.mean(np.power(total_AE, 2)))

                    # calculate all AEs' mean which bpm success estimate and GT bigger then 100
                    total_low_AE = total_low_AE.flatten()
                    total_low_AE_n = len(total_low_AE)
                    success_low_AE = len(np.where(total_low_AE < 5)[0])
                    success_rate_low = success_low_AE / total_low_AE_n
                    low_MAE = np.mean(total_low_AE)
                    low_RMSE = np.sqrt(np.mean(np.power(total_low_AE, 2)))
                    #print("length of total AEs : ", total_low_AE_n)
                    print("mode : ", m)
                    print('FPS : ', FPS)
                    print('MAE : ', MAE)
                    print('RMSE : ', RMSE)
                    print('success rate : ', success_rate)
                    print('GT < 100 , MAE : ', low_MAE)
                    print('GT < 100 , RMSE : ', low_RMSE)
                    print('GT < 100 , success rate : ', success_rate_low)
                    print("----------------------------------------------------------------")

                    if j == 1:
                        data = {"dataset": [dataset[j]], "adjust": [adjust], "mode": [m], "second": [s], "fps": [FPS],
                                "estimate n": [total_AE_n], "MAE": [MAE], "RMSE": [RMSE], "success rate": [success_rate],
                                "low GT, estimate n": [total_low_AE_n], "low GT, MAE": [low_MAE], "low GT, RMSE": [low_RMSE],
                                "low GT, success rate": [success_rate_low]
                                }
                    elif j > 1:
                        data = {"dataset": [dataset[2]], "adjust": [adjust], "mode": [m], "second": [s],
                                "bitrate": [bitrate[j - 2]], "fps": [FPS],
                                "estimate n": [total_AE_n], "MAE": [MAE], "RMSE": [RMSE], "success rate": [success_rate],
                                "low GT, estimate n": [total_low_AE_n], "low GT, MAE": [low_MAE], "low GT, RMSE": [low_RMSE],
                                "low GT, success rate": [success_rate_low]
                                }

                    df = pd.DataFrame(data)
                    # with open(result_path, "a") as f:
                    #    df.to_csv(f, header=True, index=False)
                    if s == 60:
                        df.to_csv(save_path, header=True, index=False, mode='a')
                    else:
                        df.to_csv(save_path, header=False, index=False, mode='a')
