import math
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
import scipy
import scipy.ndimage.filters as ndif
from scipy import signal, stats
from scipy.sparse import spdiags
from scipy.spatial.distance import cdist

from ASF import amplitudeSelectiveFiltering
from face_detection import FaceDetection
from face_segmentation import FaceSegmentation
from jade import jadeR

# from MFOCUSS import MFOCUSS
# from tMFOCUSS import tMFOCUSS


def wiener_filter(input_signal):
    sym4 = pywt.Wavelet('db4')
    input_signal_1 = np.asarray(pywt.swt(input_signal, sym4, level=2))[0][0]

    vector = scipy.signal.find_peaks(input_signal_1, distance=11, prominence=0.6)[0]
    for i in np.arange(len(vector)-1):
        vector[i] = int((vector[i] + vector[i+1])/2)
    np.insert(vector, [0, -1], [0, len(input_signal_1)-1])

    estimated_noise = []
    for i in np.arange(len(vector)-1):
        estimated_noise.append(.01*np.median(np.abs(input_signal_1[vector[i]:vector[i+1]])))

    input_signal = np.asarray(pywt.swt(input_signal, sym4, level=2))[0][0]
    for i in np.arange(len(vector)-1):
        input_signal[vector[i]:vector[i+1]] = np.square(input_signal_1[vector[i]:vector[i+1]]) / (np.square(
            input_signal_1[vector[i]:vector[i+1]]) + np.square(estimated_noise[i]))

    return input_signal


def wavelet_decompose(input_signal):
    # input_signal[0] = (input_signal[0] - np.mean(input_signal[0])) / np.std(input_signal[0])
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 22}
    # plt.rc('font', **font)
    # plt.subplot(2, 1, 1)
    # plt.plot(input_signal[0])

    sym4 = pywt.Wavelet('sym4')
    for i in np.arange(input_signal.shape[0]):
        input_coefficients = np.asarray(pywt.swt(input_signal[i], sym4, level=2))
        input_signal[i] = input_coefficients[0][0]

    # input_signal[0] = (input_signal[0] - np.mean(input_signal[0])) / np.std(input_signal[0])

    # plt.subplot(2, 1, 2)
    # plt.plot(input_signal[0])
    # plt.show()
    # plt.waitforbuttonpress()
    return input_signal


def dtw(x, sampling_rate, length_of_time, freq=1):
    time_axis = np.arange(0, length_of_time, 1/sampling_rate)
    PPG = np.cos(2 * np.pi * freq * time_axis)
    D = []
    x = x.reshape(-1, 1)
    PPG = PPG.reshape(-1, 1)
    '''
    manhattan_distance = lambda a, b: np.abs(a - b)
    _, cost_matrix, _, path = dtw(x, PPG, dist=manhattan_distance)
    for i in np.arange(np.shape(path)[1]):
        distance = cost_matrix[path[0][i]][path[1][i]]
        D.append(distance)
    '''
    distance_matrix = cdist(x, PPG, 'cityblock')  # Simplified distance calculation, not the dtw
    for i in np.arange(distance_matrix.shape[0]):
        temp = distance_matrix[i][i]
        D.append(temp)

    peaks_indices, _ = signal.find_peaks(D)  # Todo: investigate this value
    peaks = []
    for i in peaks_indices:
        peaks.append(D[i])

    diff_vector = []
    k = 0
    while k < len(peaks)-1:
        diff = peaks[k+1] - peaks[k]
        diff_vector.append(diff)
        k += 1

    skewness = stats.skew(np.array(diff_vector))  # todo: investigate this function
    return abs(skewness)


def RGB_2_Lab(RGB):
    Lab = RGB
    XYZ = RGB[:3, :]
    XYZ = np.matmul(
        np.array([[0.412453, 0.357580, 0.180432], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]), XYZ)
    XYZ[0] = XYZ[0] / 0.950456
    XYZ[2] = XYZ[2] / 1.088754
    fX = XYZ[0]
    fY = XYZ[1]
    fZ = XYZ[2]
    for i in range(len(fX)):
        if fX[i] > 0.008856:
            fX[i] = fX[i] ** (1 / 3)
        else:
            fX[i] = 7.787 * fX[i] + 16 / 116
        if fY[i] > 0.008856:
            fY[i] = fY[i] ** (1 / 3)
        else:
            fY[i] = 7.787 * fY[i] + 16 / 116
        if fZ[i] > 0.008856:
            fZ[i] = fZ[i] ** (1 / 3)
        else:
            fZ[i] = 7.787 * fZ[i] + 16 / 116
    for i in range(Lab.shape[1]):
        if XYZ[1, i] > 0.008856:
            Lab[0, i] = 116 * XYZ[1, i] ** (1 / 3) - 16
        else:
            Lab[0, i] = 903.3 * XYZ[1, i]
    Lab[1] = 500 * (fX - fY) + 128 * 2
    Lab[2] = 200 * (fY - fZ) + 128 * 2
    Lab[0] = Lab[0] * 255/100
    return Lab


def autocorr_method(x):
    result = np.correlate(x, x, mode='full')
    result = result[result.size//2:]
    result = np.sum(result)
    return 5/result


def detrend(input_signal, lamb=1000):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diagonal_data = np.array([ones, minus_twos, ones])
    diagonal_index = np.array([0, 1, 2])
    D = spdiags(diagonal_data, diagonal_index, (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (lamb ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def spectral_subtraction(input_signals, noise_signals, idx=0):
    # if noise_signals.shape[1] <= 1:
    if noise_signals.shape[1] <= 1:
        print('Going here first')
        return input_signals
    else:
        input_spectrum = scipy.fft.rfft(input_signals, n=input_signals.shape[1], axis=1)

        input_mag = np.abs(input_spectrum)
        input_phase = np.exp(1.0j * np.angle(input_spectrum))

        noise_mag = np.abs(scipy.fft.rfft(noise_signals, n=noise_signals.shape[1], axis=1))
        frequency_axis = 60*scipy.fft.rfftfreq(n=input_signals.shape[1], d=1/30)

        # plt.plot(frequency_axis, input_mag[2], 'r')
        # plt.title('Original power spectrum')

        max_bound = np.where(frequency_axis <= idx+5)[0][-1]
        min_bound = np.where(frequency_axis >= idx-5)[0][0]
        noise_mag[:, min_bound:max_bound + 1] *= .01

        for i in np.arange(input_signals.shape[0]):
            for j in np.arange(noise_signals.shape[0]):
                # noise_mag[j] *= np.amax(input_mag[i]) / np.amax(noise_mag[j])
                # input_mag[i] -= np.mean(noise_mag[j])
                input_mag[i] -= noise_mag[j]
                input_mag[i] = np.clip(input_mag[i], a_min=0, a_max=None)

        # plt.plot(frequency_axis, input_mag[2], 'b')
        # plt.title('Before (Red) and after (Blue) Spectral Subtraction of a power spectrum')
        # plt.show()
        # plt.waitforbuttonpress()

        input_spectrum = np.multiply(input_mag, input_phase)
        input_signals = scipy.fft.irfft(input_spectrum, n=input_signals.shape[1], axis=1)
        return input_signals


def extractRGB(frame, RGB_black_point):

    frame_size = frame.shape[0]*frame.shape[1]
    #print("face size : ", frame_size)
    face_point = frame_size - RGB_black_point

    if face_point != 0:
        actual_count = frame_size/face_point
        mean = np.stack((np.mean(frame[:, :, 0])*actual_count, np.mean(frame[:, :, 1])*actual_count,
                         np.mean(frame[:, :, 2])*actual_count), axis=0).reshape((3, 1))
        return mean
    else:
        None

    #return np.stack((np.mean(frame[:, :, 0]), np.mean(frame[:, :, 1]), np.mean(frame[:, :, 2])), axis=0).reshape((3, 1))


def extractNIR(frame, nir_black_point):
    actual_count = frame.shape[0]*frame.shape[1]/(frame.shape[0]*frame.shape[1]-nir_black_point)
    mean = (np.mean(frame)*actual_count)
    return mean


def asc(x, length_segment):
    distance_vector = []
    peak_indices, _ = signal.find_peaks(x, prominence=0.5, width=(5, 45))
    for i in np.arange((len(peak_indices))-1):
        difference = peak_indices[i+1] - peak_indices[i]
        distance_vector.append(difference)
    if (len(distance_vector) <= np.floor(40.2 * length_segment / 60)) or (len(distance_vector) > np.floor(160 * length_segment / 60)):
        std = 1000
    else:
        std = np.std(distance_vector)
    return std


def ICA(observations, num):
    demixing_matrix = jadeR(observations, m=num, verbose=False)
    return np.matmul(demixing_matrix, observations)


def butter_bandpass_filter(noisy_signal, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered_signal = signal.sosfilt(sos, noisy_signal)
    return filtered_signal


class Process(object):
    def __init__(self):
        self.sampling_rate = 30  # Frame rate of the video input
        self.order = 10  # order of butterworth filter
        self.length = 10
        self.buffer_size = self.sampling_rate * self.length
        self.NIR_signal_buffer = []
        self.RGB_signal_buffer = np.empty((3, 1))
        self.fps = 0.  # represent the performance of the computer, doesn't have any meaning besides that
        self.PSD = []
        self.noise_signal = np.zeros((2, self.buffer_size))
        self.desire_signal = []
        self.fd = FaceDetection()
        self.hr_freq = 0
        self.fs = FaceSegmentation()
        self.bpms = []
        self.NIR_PSD = []
        self.test1 = []
        self.test2 = []
        self.test3 = []
        self.test4 = []
        self.mode = 0
        self.t = []
        self.count = 0  # The second condition to stop the app
        self.FREQUENCY = []
        self.with_segment = True
        self.output_mode = 2
        self.frame_count = 1
        self.signal_process_t = []
        self.tracking_process_t = []
        self.without_wd = False

    def set_process_mode(self, mode):
        self.mode = mode

    def set_length(self, length):
        self.length = length
        self.buffer_size = self.sampling_rate * self.length
        self.noise_signal = np.zeros((2, self.buffer_size))

    def set_segment(self, segment):
        self.with_segment = segment

    def run(self, rgb_frame, nir_frame=None):
        t0 = time.time()
        bpm = 0.
        rgb_black_point = nir_black_point = 0

        # first frame detect
        if self.count == 0:
            #color_face = self.fd.face_detect(rgb_frame)
            color_face, color_bbox = self.fd.face_detect(rgb_frame)
            if self.with_segment:
                color_face, rgb_black_point = self.fs.face_segment(color_face)
                #color_face, rgb_black_point = self.fd.face_patch(rgb_frame, color_bbox)
            if nir_frame is not None:
                #nir_face = self.fd.face_detect(nir_frame)
                nir_face, nir_bbox = self.fd.face_detect(nir_frame)
                if self.with_segment:
                    #nir_face, nir_black_point = self.fs.face_segment(nir_face)
                    nir_face = self.fd.face_patch(nir_frame, nir_bbox)
        # other frame tracking
        else:
            track_start = time.time()
            #color_face = self.fd.face_track_rgb(rgb_frame)
            color_face, color_bbox = self.fd.face_track_rgb(rgb_frame)
            #color_face = self.fd.face_detect(rgb_frame)
            if self.with_segment:
                color_face, rgb_black_point = self.fs.face_segment(color_face)
                #color_face, rgb_black_point = self.fd.face_patch(rgb_frame, color_bbox)
            if nir_frame is not None:
                #nir_face = self.fd.face_track_gray(nir_frame)
                nir_face, nir_bbox = self.fd.face_track_gray(nir_frame)
                #nir_face = self.fd.face_detect_gray(nir_frame)
                #nir_face = self.fd.face_detect(nir_frame)
                if self.with_segment:
                    #nir_face, nir_black_point = self.fs.face_segment(nir_face)
                    nir_face = self.fd.face_patch(nir_frame, nir_bbox)
            track_end = time.time()
            self.tracking_process_t.append(track_end-track_start)
        #print(' tracking time : ', (track_end - track_start))

        #cv2.imshow("rgb ", rgb_frame)
        #cv2.imshow("rgb face", color_face)
        #cv2.waitKey(1)
        # extract signal
        if color_face is None:
            ROI_mean = self.RGB_signal_buffer[:, -1].reshape((3, 1))
        else:
            ROI_mean = extractRGB(color_face, rgb_black_point)
        # if face segment failed
        if ROI_mean is None:
            ROI_mean = self.RGB_signal_buffer[:, -1].reshape((3, 1))
        self.count += 1
        self.t.append(time.time() - t0)  # add time instance to the times meter

        self.RGB_signal_buffer = np.append(self.RGB_signal_buffer, ROI_mean, axis=1)

        if nir_frame is not None:
            roi_nir = extractNIR(nir_face, nir_black_point)
            self.NIR_signal_buffer = np.append(self.NIR_signal_buffer, roi_nir)


        if self.count == 3:
            self.RGB_signal_buffer = self.RGB_signal_buffer[:, 2:]
            if nir_frame is not None:
                self.NIR_signal_buffer = self.NIR_signal_buffer[1:]

        # Process in a fixed-size buffer = self.length
        length = self.RGB_signal_buffer.shape[1]
        if length > self.buffer_size:
            self.RGB_signal_buffer = self.RGB_signal_buffer[:, -self.buffer_size:]
            if nir_frame is not None:
                self.NIR_signal_buffer = self.NIR_signal_buffer[-self.buffer_size:]
            length = self.buffer_size  # set this to do it again

        #print("frame count in process : ", self.frame_count)
        if self.output_mode == 2:
            #if self.frame_count % (self.buffer_size/2) == 0:
            if self.frame_count % 6 == 0 or self.frame_count % 6 == 1:
            #if self.frame_count % 6 == 0:
                need_run = True
            else:
                need_run = False
        else:
            need_run = True
        #need_run = False

        # Start calculating at buffer_size and overlapping by buffer_size - 1
        if length >= self.buffer_size and need_run:
            print("frame count {} estimate : ".format(self.frame_count))
            #print("all signal mean : " + str(float("{:.3f}".format(np.mean(self.RGB_signal_buffer)))))
            bpm_t0 = time.time()
            self.fps = float(length) / (self.t[-1] - self.t[0])
            if nir_frame is not None:
                inputs = np.concatenate((self.RGB_signal_buffer, self.NIR_signal_buffer.reshape(1, self.buffer_size)),
                                        axis=0)
            else:
                inputs = self.RGB_signal_buffer.copy()
            # Amplitude Selective Filtering:
            if self.length == 10:
                inputs = amplitudeSelectiveFiltering(inputs, nir_max=0.005)
            elif self.length == 20:
                inputs = amplitudeSelectiveFiltering(inputs, nir_max=0.003)
            elif self.length == 30:
                inputs = amplitudeSelectiveFiltering(inputs, nir_max=0.002)
            else:
                inputs = amplitudeSelectiveFiltering(inputs)

            if self.mode == 2:
                inputs = RGB_2_Lab(inputs)
                inputs = inputs[1:, :]
                print("input shape : ", inputs.shape)
            # if nir_frame is not None:
            #     inputs = inputs[1:]

            if not self.without_wd:
                #print(self.without_wd)
                inputs = wavelet_decompose(inputs)

            self.test1 = inputs[0].copy()
            self.test2 = inputs[1].copy()
            self.test3 = inputs[2].copy()
            if self.mode == 1:
                self.test4 = inputs[3].copy()

            for i in np.arange(inputs.shape[0]):
                inputs[i] = detrend(inputs[i])
                inputs[i] = (inputs[i] - np.mean(inputs[i])) / np.std(inputs[i])

            #self.test1 = np.abs(scipy.fft.rfft(inputs[0].copy(), n=5*len(inputs[0])))/len(inputs[0])
            #self.test2 = np.abs(scipy.fft.rfft(inputs[1].copy(), n=5*len(inputs[1])))/len(inputs[1])
            #self.test3 = np.abs(scipy.fft.rfft(inputs[2].copy(), n=5*len(inputs[2])))/len(inputs[2])
            #if self.mode == 1:
            #    self.test4 = np.abs(scipy.fft.rfft(inputs[3].copy(), n=5*len(inputs[3])))/len(inputs[3])

            #print("shape of input which is before ica : ", inputs.shape)
            # ICA = JADE method
            try:
                if self.mode == 1:
                    inputs = ICA(inputs, num=4)
                else:
                    inputs = ICA(inputs, num=3)
            except np.linalg.LinAlgError:
                if nir_frame is not None:
                    return 0, color_face, nir_face
                else:
                    return 0, color_face

            #self.test1 = np.abs(scipy.fft.rfft(inputs[0].copy(), n=5*self.buffer_size))
            #self.test2 = np.abs(scipy.fft.rfft(inputs[1].copy(), n=5*self.buffer_size))
            #self.test3 = np.abs(scipy.fft.rfft(inputs[2].copy(), n=5*self.buffer_size))
            #if self.mode == 1:
            #    self.test4 = np.abs(scipy.fft.rfft(inputs[3].copy(), n=5*self.buffer_size))


            # ASC
            stat1 = asc(inputs[0], self.length)
            stat2 = asc(inputs[1], self.length)
            stat3 = asc(inputs[2], self.length)
            if self.mode == 1:
                stat4 = asc(inputs[3], self.length)

            # select ppg signal
            if self.mode == 1:
                min_stat = min(stat1, stat2, stat3, stat4)
            else:
                min_stat = min(stat1, stat2, stat3)

            if min_stat == stat1:
                color_ppg = inputs[0]
                #print('select r channel')
            elif min_stat == stat2:
                color_ppg = inputs[1]
                #print('select g channel')
            elif min_stat == stat3:
                color_ppg = inputs[2]
                #print('select b channel')
            elif nir_frame is not None:
                color_ppg = inputs[3]

            #color_ppg = inputs[1]
            color_ppg = butter_bandpass_filter(color_ppg, 0.67, 2.4, self.sampling_rate, self.order)

            # Power Spectrum Analysis
            self.PSD = np.abs(scipy.fft.rfft(color_ppg, n=5*len(color_ppg))) / len(color_ppg)
            self.FREQUENCY = scipy.fft.rfftfreq(n=5*self.buffer_size, d=(1 / self.sampling_rate))
            self.FREQUENCY *= 60
            #print('len of frequency : ', len(self.FREQUENCY))
            idx = np.argmax(self.PSD)

            #print("idx : ", idx)
            #if len(self.bpms) == 0:
            #    S = np.sum(self.PSD[idx-5:idx+6])
            #    self.PSD[idx-5:idx+6] = 0
            #    print('SNR: ' + str(S/np.sum(self.PSD)))

            if not np.isnan(bpm):
                bpm = self.limit_bpm(self.PSD, self.FREQUENCY, 144)
                #bpm = self.FREQUENCY[idx]
            else:
                print("nan value")


            self.bpms.append(bpm)
            self.signal_process_t.append(time.time()-bpm_t0)
            #print('bpm process time is: ' + str((time.time() - bpm_t0)))
            #print('bpm : ' + str(bpm))
            #print('self bpm : ', self.bpms)
        self.frame_count += 1
        if nir_frame is not None:
            return bpm, color_face, nir_face
        else:
            return bpm, color_face

    def reset(self):
        self.NIR_signal_buffer = []
        self.RGB_signal_buffer = np.empty((3, 1))
        self.noise_signal = np.zeros((2, self.buffer_size))
        self.fps = 0.  # represent the performance of the computer, doesn't have any meaning besides that
        self.PSD = []
        self.FREQUENCY = []
        self.SNR = []
        self.fd = FaceDetection()
        self.fs = FaceSegmentation()
        self.desire_signal = []
        self.bpms = []
        self.test1 = []
        self.test2 = []
        self.test3 = []
        self.test4 = []
        self.t = []
        self.count = 0
        self.hr_freq = 0
        self.frame_count = 1

    def limit_bpm(self, PSD, Frequency, limit):
        idx = np.argmax(PSD)
        #print("limit idx : ", idx)
        bpm = Frequency[idx]
        #print("select bpm : ", bpm)
        if bpm > limit:
            wo_max_PSD = PSD.copy()
            wo_max_PSD[idx] = np.min(PSD)
            bpm = self.limit_bpm(wo_max_PSD, Frequency, limit)
        return bpm

    def DFT_matrix(self, lower_freq, upper_freq, sampling_rate, points_num):
        lower_transition_band = 0.1*points_num/sampling_rate - 1
        upper_transition_band = 0.1*points_num/sampling_rate
        lower_bound_index = int(lower_freq*points_num/sampling_rate - lower_transition_band)
        upper_bound_index = int(upper_freq*points_num/sampling_rate + upper_transition_band)
        Phi = np.zeros((self.N, points_num), dtype=complex)
        for i in range(self.N):
            for j in np.arange(lower_bound_index, upper_bound_index + 1):
                Phi[i, j] = np.exp(2j * math.pi * i * j / points_num)
            for j in np.arange(points_num-upper_bound_index, points_num-lower_bound_index + 1):
                Phi[i, j] = np.exp(2j * math.pi * i * j / points_num)
        return Phi
