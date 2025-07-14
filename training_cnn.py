# train_logic.py
import os
import zipfile
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import Callback, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from flask import render_template
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.regularizers import l2

MODEL_FOLDER = 'static/models/'

def preprocess_two_labels(label_a, label_b, label_c):
    if label_a == 0 and label_b == 0 and label_c == 0:
        return '000'
    elif label_a == 0 and label_b == 0 and label_c == 1:
        return '001'
    elif label_a == 0 and label_b == 1 and label_c == 0:
        return '010'
    elif label_a == 0 and label_b == 1 and label_c == 1:
        return '011'
    elif label_a == 1 and label_b == 0 and label_c == 0:
        return '100'
    elif label_a == 1 and label_b == 1 and label_c == 0:
        return '110'
    else:
        return 'invalid'

def train_model(file_path, return_html=True):
    data = sio.loadmat(file_path)
    X = data["data"]
    v = data["label_1"][0]
    a = data["label_2"][0]
    d = data["label_3"][0]

    def preprocess_three_labels(val, ars, dom):
        mapping = {
            (0, 0, 0): '000',
            (0, 0, 1): '001',
            (0, 1, 0): '010',
            (0, 1, 1): '011',
            (1, 0, 0): '100',
            (1, 1, 0): '110',
        }
        return mapping.get((val, ars, dom), 'invalid')

    raw_labels = list(map(preprocess_three_labels, v, a, d))
    valid_idx = [i for i, lbl in enumerate(raw_labels) if lbl != 'invalid']
    filtered_data = X[valid_idx]
    filtered_labels = [raw_labels[i] for i in valid_idx]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(filtered_labels)
    y = to_categorical(y_encoded)

    filtered_data = filtered_data.transpose(0, 2, 3, 1)
    filtered_data = filtered_data[:, :, :, [0, 1, 2, 3]]

    labels_argmax = np.argmax(y, axis=1)

    initial_weights_path = os.path.join(MODEL_FOLDER, 'initial.weights.h5')

    def build_model(input_shape, num_classes):
        model = Sequential([
            Conv2D(64, kernel_size=2, padding='same', activation='relu', input_shape=input_shape),
            Conv2D(128, kernel_size=2, padding='same', activation='relu'),
            Conv2D(256, kernel_size=2, padding='same', activation='relu'),
            Conv2D(64, kernel_size=1, padding='same', activation='relu'),
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax', activity_regularizer=l2(0.5))
        ])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])
        return model

    model_template = build_model(filtered_data.shape[1:], y.shape[1])
    model_template.save_weights(initial_weights_path)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy_list, precision_list, recall_list, f1_list, epoch_list, model_files = [], [], [], [], [], []

    total_epochs = 50 * 10
    progress_counter = [0]

    class ProgressCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_counter[0] += 1
            percent = int((progress_counter[0] / total_epochs) * 100)
            with open("progress.txt", "w") as f:
                f.write(str(percent))

    for fold, (train_val_idx, _) in enumerate(kf.split(filtered_data, labels_argmax), start=1):
        model = build_model(filtered_data.shape[1:], y.shape[1])
        model.load_weights(initial_weights_path)

        X_train, X_val, y_train, y_val = train_test_split(
            filtered_data[train_val_idx], y[train_val_idx],
            test_size=0.1, stratify=labels_argmax[train_val_idx],
            random_state=fold
        )

        early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            batch_size=100, epochs=50,
            validation_data=(X_val, y_val),
            callbacks=[early_stop, ProgressCallback()],
            verbose=0
        )

        # Evaluation
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        y_true = np.argmax(y_val, axis=1)

        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        accuracy_list.append(round(acc * 100, 2))
        precision_list.append(round(precision * 100, 2))
        recall_list.append(round(recall * 100, 2))
        f1_list.append(round(f1 * 100, 2))
        epoch_list.append(len(history.epoch))

        model_name = f'model_fold_{fold}.h5'
        model.save(os.path.join(MODEL_FOLDER, model_name))
        model_files.append(model_name)

        # Save loss plot
        plt.figure()
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(f'Loss Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(MODEL_FOLDER, f'fold_{fold}_loss.png'))
        plt.close()

        # Save training history
        pd.DataFrame({
            'epoch': history.epoch,
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        }).to_excel(os.path.join(MODEL_FOLDER, f'fold_{fold}.xlsx'), index=False)

    with open("progress.txt", "w") as f:
        f.write("100")

    results_df = pd.DataFrame({
        'Fold': list(range(1, 11)),
        'Accuracy': accuracy_list,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1-Score': f1_list,
        'Epochs': epoch_list
    })
    results_df.to_excel(os.path.join(MODEL_FOLDER, "metrics.xlsx"), index=False)

    average_metrics = {
        'Accuracy': round(np.mean(accuracy_list), 2),
        'Precision': round(np.mean(precision_list), 2),
        'Recall': round(np.mean(recall_list), 2),
        'F1-Score': round(np.mean(f1_list), 2),
        'Epoch': round(np.mean(epoch_list), 2)
    }

    zip_path = os.path.join(MODEL_FOLDER, "all_models_and_metrics.rar")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in model_files:
            zipf.write(os.path.join(MODEL_FOLDER, file), arcname=file)
        zipf.write(os.path.join(MODEL_FOLDER, "metrics.xlsx"), arcname="metrics.xlsx")
        for i in range(1, 11):
            zipf.write(os.path.join(MODEL_FOLDER, f'fold_{i}_loss.png'), arcname=f'fold_{i}_loss.png')
            zipf.write(os.path.join(MODEL_FOLDER, f'fold_{i}.xlsx'), arcname=f'fold_{i}.xlsx')

    if return_html:
        return render_template('train_result.html',
                               model_files=model_files,
                               metrics_df=results_df.to_dict(orient='records'),
                               averages=average_metrics)
    else:
        return {
            'model_files': model_files,
            'metrics_df': results_df,
            'averages': average_metrics
        }

def get_training_results():
    model_files = sorted([f for f in os.listdir(MODEL_FOLDER) if f.endswith('.h5')])
    excel_path = os.path.join(MODEL_FOLDER, "metrics.xlsx")

    if os.path.exists(excel_path):
        results_df = pd.read_excel(excel_path)
        averages = {
            'Accuracy': round(results_df['Accuracy'].mean(), 2),
            'Precision': round(results_df['Precision'].mean(), 2),
            'Recall': round(results_df['Recall'].mean(), 2),
            'F1-Score': round(results_df['F1-Score'].mean(), 2)
        }
    else:
        results_df = pd.DataFrame([{
            'Fold': i + 1, 'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-Score': 0
        } for i in range(len(model_files))])
        averages = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-Score': 0}

    return model_files, results_df, averages
