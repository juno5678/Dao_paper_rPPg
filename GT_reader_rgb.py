import json
import math
import h5py
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
import sys

success = 0
mode = "json"
bpm = 0.0
bpm_avg = 0.0
rmse = 0.0
error = []
sampling_rate = 256  # samplingrate of COHFACE
count = 0  # Reading at 60Hz
i = 0
test_length = 60
test = [150, 150, 150]
if mode == "json":
    #with open('D:/CCU/HR_Estimator/dataset/w/01-02/01-02.json') as file:
    with open(sys.argv[1]) as file:
        data = json.loads(file.read())
        data = data['/FullPackage']
        for p in data:
            bpm += p['Value']['pulseRate']
            count += 1
            i += 1
            bpm_avg = bpm/count
            if count == 30*test_length:
                break
    print('Count:' + str(count))
else:
    file = "./test videos/cohface/40/3/data.hdf5"
    with h5py.File(file, "r") as f:
        print("Key: %s" % f.keys())
        PPG_Signal = np.asarray(f['pulse'])

        time = np.asarray(f['time'])
        idx = np.where(time == 12)[0]
        idx = int(idx)
        print(idx)
        PPG_Signal = PPG_Signal[:idx]
        time = time[:idx]

        psd = np.abs(scipy.fft.rfft(PPG_Signal, PPG_Signal.size))
        freqs = scipy.fft.rfftfreq(PPG_Signal.size, 1 / sampling_rate)

        mean = np.mean(PPG_Signal)
        peak_indices, _ = signal.find_peaks(PPG_Signal, distance=220)
        print(len(peak_indices))
        IBI = []
        for i in np.arange(len(peak_indices) - 1):
            tmp = peak_indices[i + 1] - peak_indices[i]
            IBI.append(tmp)
        IBI = np.asarray(IBI) / sampling_rate
        IBI = np.mean(IBI)

        # Print average HR based on IBI
        bpm_avg = (60/IBI)

        plt.subplot(121)
        plt.plot(time, PPG_Signal)
        plt.subplot(122)
        plt.plot(freqs[:80], psd[:80])
        plt.show()

# Print average HR based on IBI
print("GT BPM is: " + str(bpm_avg))
for i in range(len(test)):
    error.append(test[i] - bpm_avg)

# RMSE:
rmse = np.sum(np.square(error))
rmse = np.sqrt(rmse/len(test))
print('RMSE is: ' + str(np.round(rmse, 2)))

# Success Rate
error = np.asarray(error)
success = np.where(np.abs(error) <= 5.)[0]
print('Success Rate is: ' + str(len(success)/len(test)))
# print('Success Rate is ', 1 - fail/count)

# Mean Error:
print('Mean Error is: ' + str(np.round(np.mean(np.abs(error)), 2)))

# Standard Deviation:
print('Standard Deviation is: ' + str(np.round(np.std(error), 2)))
