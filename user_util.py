import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, average_precision_score, accuracy_score, f1_score, matthews_corrcoef

from tensorflow.keras import layers
from custom_layer import SpatialAttention, TemporalAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K


def make_windows(arr, win_size, step_size):
    w_list = list()
    n_records = arr.shape[0]
    remainder = (n_records - win_size) % step_size 
    num_windows = 1 + int((n_records - win_size - remainder) / step_size)
    for k in range(num_windows):
        w_list.append(arr[k*step_size:win_size-1+k*step_size+1])
    return np.array(w_list)
    
    
def load_data(X_path, y_path,
              window_size, step_size):
    data_X  = np.load(X_path)
    data_y  = np.load(y_path)
    data_X = data_X[:, 1:]

    train_X, val_X, test_X = data_X[:131746, :], data_X[131746:158095, :], data_X[158095:, :] # 0.625, 0.125, 0.25
    train_y, val_y, test_y = data_y[:131746], data_y[131746:158095], data_y[158095:]

    scaler = StandardScaler().fit(train_X)
    train_X, val_X, test_X = scaler.transform(train_X), scaler.transform(val_X), scaler.transform(test_X)

    window_s = window_size
    step_s   = step_size
    train_X, val_X, test_X = make_windows(train_X, window_s, step_s),\
    make_windows(val_X, window_s, step_s),\
    make_windows(test_X, window_s, step_s)
    train_y, val_y, test_y = make_windows(np.expand_dims(train_y, axis=-1), window_s, step_s),\
    make_windows(np.expand_dims(val_y, axis=-1), window_s, step_s),\
    make_windows(np.expand_dims(test_y, axis=-1), window_s, step_s)
    return (train_X, train_y), (val_X, val_y), (test_X, test_y)

    
def build_model(window_size, input_dim,
                spatial_dim, temporal_dim,
                lstm_dim, out_dim,
                lr=0.001):
    inputs = layers.Input(shape=(window_size, input_dim))

    spatial_emb, spatial_attnw = SpatialAttention(units=spatial_dim, dropout=0.3, return_attention=True, name='spatial_attn')(inputs)
    temporal_emb, temporal_attnw = TemporalAttention(units=temporal_dim, return_attention=True, name='temporal_attn')(inputs)
    attn = layers.Concatenate()([spatial_emb, temporal_emb])

    lstm = layers.LSTM(lstm_dim, activation='tanh', return_sequences=True, name='lstm')(attn)
    norm = layers.LayerNormalization(name='lstm_norm')(lstm)
    out = layers.Dense(out_dim, activation='elu')(norm)
    out = layers.Dense(1, activation='sigmoid', name='out')(out)

    model = Model(inputs=inputs, outputs=out, name='model')
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy', AUC(name='auc')]
    )
    return model


def get_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    J = tpr - fpr
    return thresholds[np.argmax(J)]


def evaluate(y_true, y_prob, threshold=None):
    if threshold is None:
        threshold = get_threshold(y_true, y_prob)
    else:
        threshold = 0.5
    y_pred = np.where(y_prob > threshold, 1, 0)
    
    print(f'AUPRC: %.4f' % average_precision_score(y_true, y_pred))
    print(f'Accuracy: %.4f' % accuracy_score(y_true, y_pred))
    print(f'F1: %.4f' % f1_score(y_true, y_pred))
    print(f'MCC: %.4f' % matthews_corrcoef(y_true, y_pred))