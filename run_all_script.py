import os

#dataRoot = '../../dataset/paper_dataset'
#savePath = './result/test_10s_1.csv'
#dataPath = ['../dataset/paper_dataset/10s_path_gt.csv', '../dataset/PURE/', '../dataset/PURE_raw_dataset/']
dataPath = ['../dataset/paper_dataset/10s_path_gt.csv', '../dataset/PURE_784k/', '../dataset/PURE_raw_dataset/']
dataset = ['CCUHR', 'PURE_video', 'PURE_image']
adjust = "_p4_asf_twoline_on_CSRT_784k"
#second = [1, 2, 3, 6]
second = [6, 3, 2, 1]
#second = [1]
input_mode = 0
process_mode = 0
output_mode = 2
for j in range(1, 0, -1):
    for i in second:
        if input_mode == 3:
            dataPath = list(dataPath)
            dataPath[-15] = str(i)
            dataPath = ''.join(dataPath)

        input_mode = j - 1
        savePath = "../result/DAO/"+dataset[j]+adjust+"_10s.csv"
        # change second
        savePath = list(savePath)
        savePath[-7] = str(i)
        savePath = ''.join(savePath)

        comment = 'python ./record_result.py -d ' + dataPath[j] + ' --savePath ' + savePath + ' -i ' + str(input_mode) + \
                  ' -p ' + str(process_mode) + ' -o ' + str(output_mode) + ' -s ' + str(i*10) + ' --save'
        os.system(comment)
        #print(comment)
