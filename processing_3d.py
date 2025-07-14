import numpy as np
import scipy.io as sio
from sklearn import preprocessing

def read_file(file):
    trial_data = file['data']
    base_data = file["base_data"]
    data_seconds_list = file['data_seconds_list']
    return trial_data, base_data, file["label_1"], file["label_2"], file['label_3'], data_seconds_list

def get_vector_deviation(vector1, vector2):
    return vector1 / vector2

def get_dataset_deviation(trial_data, base_data, data_seconds_list):
    new_dataset = np.empty([0, 64])
    second_now = 0
    for i, seconds in enumerate(data_seconds_list[0]):
        for j in range(int(seconds)):
            new_record = get_vector_deviation(trial_data[j + second_now], base_data[i]).reshape(1, 64)
            new_dataset = np.vstack([new_dataset, new_record])
        second_now += int(seconds)
    return new_dataset

def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros((Y, X))  # Tambahkan ini

    data_2D[0] = (0, 0, 0, data[0], 0, data[1], 0, 0, 0)
    data_2D[1] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[2] = (data[8], 0, data[2], 0, 0, 0, data[3], 0, data[9])
    data_2D[3] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[4] = (data[10], 0, data[4], 0, 0, 0, data[5], 0, data[11])
    data_2D[5] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[6] = (data[12], 0, data[6], 0, 0, 0, data[7], 0, data[13])
    data_2D[7] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[8] = (0, 0, 0, data[14], 0, data[15], 0, 0, 0)

    return data_2D

def preprocess_to_3D(mat_data, use_baseline="yes"):
    sub_vector_len = 16
    data_3D = np.empty([0, 9, 9])
    
    trial_data, base_data, valence, arousal, dominance, seconds = read_file(mat_data)
    
    if use_baseline == "yes":
        data = get_dataset_deviation(trial_data, base_data, seconds)
    else:
        data = trial_data
    data = preprocessing.scale(data, axis=1)

    for vector in data:
        for band in range(4):
            slice = vector[band * sub_vector_len: (band + 1) * sub_vector_len]
            reshaped = data_1Dto2D(slice).reshape(1, 9, 9)
            data_3D = np.vstack([data_3D, reshaped])

    data_3D = data_3D.reshape(-1, 4, 9, 9)
    return {
        "data": data_3D,
        "label_1": valence,
        "label_2": arousal,
        "label_3": dominance
    }
