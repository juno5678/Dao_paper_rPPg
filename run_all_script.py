import os

#dataRoot = '../../dataset/paper_dataset'
#savePath = './result/test_10s_1.csv'
#dataPath = ['../dataset/paper_dataset/10s_path_gt.csv', '../dataset/PURE/', '../dataset/PURE_raw_dataset/']
dataPath = ['../rPPG_dataset/paper_dataset/10s_path_gt.csv',
            '../rPPG_dataset/PURE_raw_dataset/',
            '../rPPG_dataset/PURE_784k/', '../dataset/PURE_979k/','../dataset/PURE_1223k/', '../dataset/PURE_1467k/']
dataset = ['CCUHR', 'PURE_image', 'PURE_video']
bitrate = ['784k', '979k', '1223k', '1467k']
adjust = "_p4_MIL_all_"
#second = [1, 2, 3, 6]
second = [6, 3, 2, 1]
#second = [1]
input_mode = 0
process_mode = 0
output_mode = 2
for j in range(1, len(dataPath)-1):
    for i in second:
        # realsense video
        #if input_mode == 3:
        #    dataPath = list(dataPath)
        #    dataPath[-15] = str(i)
        #    dataPath = ''.join(dataPath)

        if j == 1:
            savePath = "../result/DAO/"+dataset[j]+adjust+bitrate[j-1]+"_10s.csv"
            input_mode = j
        elif j > 1:
            savePath = "../result/DAO/"+dataset[2]+adjust+bitrate[j-1]+"_10s.csv"
            input_mode = 2

        # change second
        savePath = list(savePath)
        savePath[-7] = str(i)
        savePath = ''.join(savePath)

        comment = 'python ./record_result.py -d ' + dataPath[j] + ' --savePath ' + savePath + ' -i ' + str(input_mode) + \
                  ' -p ' + str(process_mode) + ' -o ' + str(output_mode) + ' -s ' + str(i*10) + ' --save'
        os.system(comment)
        #print(comment)
