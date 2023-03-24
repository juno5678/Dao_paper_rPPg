import numpy as np
from scipy import signal
from math import *
import scipy
import matplotlib.pyplot as plt

#Todo: For bag file, we need to skip the first frame

# Sampling rate of the PPG monitor
sampling_rate = 500
duration = 10
time = int(sampling_rate*duration)
t_axis = np.arange(time)
test = [85.2, 90, 91.2]
error = []
# GT = 86
rmse = 0
fail = 0
count = 0

# Read the data from file and convert it to integer
file = open("ground truth files/athony.lvm", "r")
PPG_signal = file.readlines()
PPG_signal = list((map(float, PPG_signal)))
# Set the length of the PPG signal based on number of samples
PPG_signal = np.asarray(PPG_signal)
PPG_signal = PPG_signal[1:time+1]

psd = np.abs(scipy.fft.rfft(PPG_signal, PPG_signal.size))
freqs = scipy.fft.rfftfreq(PPG_signal.size, 1/sampling_rate)

# Peak detection, IBI calculation
mean = np.mean(PPG_signal)
peak_indices, _ = signal.find_peaks(PPG_signal, distance=280)
# print(len(peak_indices))
IBI = []
for i in np.arange(len(peak_indices)-1):
    tmp = peak_indices[i+1]-peak_indices[i]
    IBI.append(tmp)
IBI = np.asarray(IBI)/sampling_rate
IBI = np.mean(IBI)
# print(IBI)
GT = 60/IBI

# Print average HR based on IBI
print("Heart rate of this data signal is: " + str(GT))
for i in range(len(test)):
    error.append(test[i] - GT)

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

# plt.subplot(121)
# plt.plot(t_axis, PPG_signal)
# plt.subplot(122)
# plt.plot(freqs[:80], psd[:80])
# plt.show()
