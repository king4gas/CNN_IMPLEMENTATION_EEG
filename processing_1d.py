import os
import math
import numpy as np
import scipy.io as sio
from scipy.signal import butter, lfilter

# ---------------------------- FILTER
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

# ---------------------------- NORMALISASI & DE
def normalization(signal):
    return (signal - signal.mean()) / signal.std()

def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2

# ---------------------------- DEKOMPOSISI DATA
def decompose(data):
    frequency = 100
    decomposed_de = np.empty((0, 4), dtype=np.float64)
    base_DE = np.empty((0, 64), dtype=np.float64)
    data_seconds_list = np.empty((0,), dtype=np.float64)

    for trial in range(12):
        temp_base_theta_DE, temp_base_alpha_DE = np.empty(0), np.empty(0)
        temp_base_beta_DE, temp_base_gamma_DE = np.empty(0), np.empty(0)

        trial_data = data[:, trial][0]
        data_seconds = trial_data.shape[0] // frequency
        temp_de = np.empty((0, data_seconds), dtype=np.float64)

        for channel in range(16):
            trial_signal = trial_data[:, channel]
            base_signal = trial_data[:300, channel]

            remainder = trial_signal.shape[0] % frequency
            if remainder != 0:
                trial_signal = np.append(trial_signal, np.zeros(frequency - remainder))

            base_signal = normalization(base_signal)
            trial_signal = normalization(trial_signal)

            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)

            temp_base_theta_DE = np.append(temp_base_theta_DE, np.mean([
                compute_DE(base_theta[:100]),
                compute_DE(base_theta[100:200]),
                compute_DE(base_theta[200:])
            ]))
            temp_base_alpha_DE = np.append(temp_base_alpha_DE, np.mean([
                compute_DE(base_alpha[:100]),
                compute_DE(base_alpha[100:200]),
                compute_DE(base_alpha[200:])
            ]))
            temp_base_beta_DE = np.append(temp_base_beta_DE, np.mean([
                compute_DE(base_beta[:100]),
                compute_DE(base_beta[100:200]),
                compute_DE(base_beta[200:])
            ]))
            temp_base_gamma_DE = np.append(temp_base_gamma_DE, np.mean([
                compute_DE(base_gamma[:100]),
                compute_DE(base_gamma[100:200]),
                compute_DE(base_gamma[200:])
            ]))

            for band_filter in [(4, 8), (8, 14), (14, 31), (31, 45)]:
                band = butter_bandpass_filter(trial_signal, *band_filter, frequency, order=3)
                band_de = [
                    compute_DE(band[i * frequency:(i + 1) * frequency])
                    for i in range(data_seconds)
                ]
                temp_de = np.vstack([temp_de, np.array(band_de)])

        temp_trial_de = temp_de.reshape(-1, 4)
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

        temp_base_DE = np.concatenate([
            temp_base_theta_DE, temp_base_alpha_DE,
            temp_base_beta_DE, temp_base_gamma_DE
        ])
        base_DE = np.vstack([base_DE, temp_base_DE])
        data_seconds_list = np.append(data_seconds_list, float(data_seconds))

    decomposed_de = decomposed_de.reshape(-1, 16, 4).transpose(0, 2, 1).reshape(-1, 4, 16).reshape(-1, 64)
    return base_DE, decomposed_de, data_seconds_list

# ---------------------------- LABEL
def get_labels(data_labels, data_trial):
    valence, arousal, dominance = np.empty(0), np.empty(0), np.empty(0)

    for trial in range(12):
        label = data_labels[:, trial][0][0]
        seconds = data_trial[:300, trial][0].shape[0] // 100

        valence = np.append(valence, [label[1] == 1] * seconds)
        arousal = np.append(arousal, [label[0] == 1] * seconds)
        dominance = np.append(dominance, [label[2] == 1] * seconds)

    return valence.astype(np.float64), arousal.astype(np.float64), dominance.astype(np.float64)

# ---------------------------- UTAMA UNTUK BACKEND
def process_1d_from_mat(mat_data):
    joined_data = mat_data["joined_data"]
    labels = mat_data["labels_selfassessments"]

    base, data, seconds = decompose(joined_data.copy())
    val, aro, dom = get_labels(labels.copy(), joined_data.copy())

    return {
        "base_data": base,
        "data": data,
        "label_1": val,
        "label_2": aro,
        "label_3": dom,
        "data_seconds_list": seconds
    }
