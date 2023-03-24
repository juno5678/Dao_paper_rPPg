import numpy as np
import matplotlib.pyplot as plt
import scipy


def amplitudeSelectiveFiltering(C_input, sampling_rate=30, red_max=0.0025, nir_max=0.0025, delta=0.0001):
    '''
    Input: Raw RGB signals with dimensions NxL, where the R channel is column 0
    Output:
    C = Filtered RGB-signals with added global mean,
    raw = Filtered RGB signals
    '''

    N = C_input.shape[0]
    L = C_input.shape[1]
    C_mean = np.tile((np.mean(C_input, axis=1) - 1).reshape(C_input.shape[0], 1), L)
    C = C_input/C_mean - 1

    # line 2
    F = scipy.fft.rfft(C, n=L, axis=1)
    F_mag = np.abs(F) / L
    F_phase = np.exp(1.0j * np.angle(F))
    freqs = 60*scipy.fft.rfftfreq(L, 1/sampling_rate)

    # max_bound = np.where(freqs <= 240)[0][-1]
    # min_bound = np.where(freqs >= 40)[0][0]
    # line 3
    W = np.ones((1, len(freqs)))


    #font = {'family': 'sans-serif',
    #        'weight': 'bold',
    #        'size': 22}
    #plt.rc('font', **font)
    #plt.plot(freqs, F_mag[0], 'r', label='red')
    #plt.plot(freqs, F_mag[2], 'b', label='blue')
    #plt.plot(freqs, F_mag[1], 'g', label='green')
    ##plt.plot(freqs, F_mag[3], 'k', label='NIR')
    #plt.legend(fontsize='x-large')
    #plt.ylim(bottom=0, top=0.005)
    #plt.xlim(left=36, right=210)
    #plt.show()
    #plt.waitforbuttonpress()
    # line 4
    for i in range(1, int(L/2) + 1):
        if np.sum(F_mag[2][i]) >= 1.25 * (F_mag[0][i] + F_mag[1][i]):
            W[0][i] = delta * F_mag[0][i]
            # W[0][i] = 0
        if F_mag.shape[0] > 3:
            if F_mag[0][i] >= red_max or F_mag[3][i] >= nir_max:
                #print("in masf")
                W[0][i] = delta * F_mag[0][i]
        else:
            if F_mag[0][i] >= red_max:
                W[0][i] = delta * F_mag[0][i]
        # if F_mag[0][i] <= 0.00025:
        #     W[0][i] = delta * F_mag[0][i]
        #     W[0, -i] = delta/F[0, -i]  # For fft function

    W = np.tile(W, (N, 1))
    F_mag = np.multiply(F_mag, W)

    #for i in range(1, int(L/2) + 1):
    #    if (F_mag[0][i] + F_mag[1][i] + F_mag[2][i]) <= 0.0005:
    #        F_mag[0][i] = 0
    #        F_mag[1][i] = 0
    #        F_mag[2][i] = 0
    #        #F_mag[3][i] = 0

    #font = {'family': 'sans-serif',
    #        'weight': 'bold',
    #        'size': 22}
    #plt.rc('font', **font)
    #plt.plot(freqs, F_mag[0], 'r', label='red')
    #plt.plot(freqs, F_mag[2], 'b', label='blue')
    #plt.plot(freqs, F_mag[1], 'g', label='green')
    ##plt.plot(freqs, F_mag[3], 'k', label='NIR')
    #plt.legend(fontsize='x-large')
    #plt.ylim(bottom=0, top=0.005)
    #plt.xlim(left=36, right=210)
    #plt.show()
    #plt.waitforbuttonpress()

    # line 6
    F_mag = F_mag * L
    F = np.multiply(F_mag, F_phase)
    C = scipy.fft.irfft(F, n=L, axis=1) + 1
    C = C*C_mean

    del F_mag, F, W
    return C
