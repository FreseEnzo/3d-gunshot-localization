#!/usr/bin/env python3
"""
wav_distance_cnn.py

Processa arquivos .wav multicanal de disparos e treina uma CNN
de regressão da distância 3D (considerando offset vertical dos microfones)
usando espectrogramas Mel janelados.
Os metadados são embutidos diretamente no código.

Uso:
  Treinamento:
    $ python wav_distance_cnn.py --data_dir /data/wavs

  Teste de um arquivo .wav (usa os 4 mics ao mesmo tempo):
    $ python wav_distance_cnn.py --data_dir /data/wavs --test_file shot29.wav

Dependências:
  pip install numpy>=1.20 scipy>=1.5 scikit-learn>=0.24 tensorflow>=2.6 joblib>=1.0 soundfile librosa
"""
import os
import argparse
import numpy as np
import soundfile as sf
import librosa
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES ---
SR_TARGET = 48000        # Hz após resampling
DURATION = 0.1           # segundos de muzzle blast
N_FFT = 512
HOP = 256
N_MELS = 16
WIN_COLS = 16            # colunas Mel por janela
STRIDE = 8               # deslocamento entre janelas
BATCH_SIZE = 8
EPOCHS = 100
VERT_OFFSET = 3.2        # deslocamento vertical em metros dos mics em relação ao cano

# --- METADADOS (shot, distância horizontal) ---
METADATA = [
    ('shot1','5,2 m'),('shot2','5,2 m'),('shot3','5,2 m'),('shot4','5,2 m'),
    ('shot5','5,7 m'),('shot6','5,7 m'),('shot7','5,7 m'),('shot8','5,7 m'),
    ('shot9','6,2 m'),('shot10','6,2 m'),('shot11','6,2 m'),('shot12','6,2 m'),
    ('shot13','6,7 m'),('shot14','6,7 m'),('shot15','6,7 m'),('shot16','6,7 m'),
    ('shot17','7,2 m'),('shot18','7,2 m'),('shot19','7,2 m'),('shot20','7,2 m'),
    ('shot21','7,7 m'),('shot22','7,7 m'),('shot23','7,7 m'),('shot24','7,7 m'),
    ('shot25','8,2 m'),('shot26','8,2 m'),('shot27','8,2 m'),('shot28','8,2 m'),
    ('shot29','5,2 m'),('shot30','5,2 m'),('shot31','5,2 m'),('shot32','5,2 m'),
    ('shot33','5,7 m'),('shot34','5,7 m'),('shot35','5,7 m'),('shot36','5,7 m'),
    ('shot37','6,2 m'),('shot38','6,2 m'),('shot39','6,2 m'),('shot40','6,2 m'),
    ('shot41','6,7 m'),('shot42','6,7 m'),('shot43','6,7 m'),('shot44','6,7 m'),
    ('shot45','7,2 m'),('shot46','7,2 m'),('shot47','7,2 m'),('shot48','7,2 m'),
    ('shot49','7,7 m'),('shot50','7,7 m'),('shot51','7,7 m'),('shot52','7,7 m'),
    ('shot53','8,2 m'),('shot54','8,2 m'),('shot55','8,2 m'),('shot56','8,2 m')
]

def parse_metadata():
    """Retorna lista de (filename, distancia_3d)"""
    entries = []
    for shot, dist_str in METADATA:
        fname = f"{shot}.wav"
        # converte string '5,2 m' em float
        dh = float(dist_str.replace('m','').replace(',','.').strip())
        # calcula distância 3D
        d3 = np.sqrt(dh**2 + VERT_OFFSET**2)
        entries.append((fname, d3))
    return entries

# --- EXTRAÇÃO DE JANELAS MEL MULTICANAL ---
def extract_windows_multichannel(signal, orig_sr):
    n_samples = int(orig_sr * DURATION)
    sig_crop = signal[:n_samples]
    # resample cada canal
    sig_ds = np.stack([
        librosa.resample(sig_crop[:, ch], orig_sr=orig_sr, target_sr=SR_TARGET)
        for ch in range(sig_crop.shape[1])
    ], axis=1)
    mel_basis = librosa.filters.mel(sr=SR_TARGET, n_fft=N_FFT, n_mels=N_MELS)
    specs = []
    for ch in range(sig_ds.shape[1]):
        S = np.abs(librosa.stft(sig_ds[:, ch], n_fft=N_FFT, hop_length=HOP))**2
        mel = mel_basis.dot(S)
        specs.append(librosa.power_to_db(mel, ref=np.max))
    # janelamento no tempo
    T = specs[0].shape[1]
    n_wins = max(1, (T - WIN_COLS)//STRIDE + 1)
    windows = []
    for i in range(n_wins):
        s, e = i*STRIDE, i*STRIDE + WIN_COLS
        win = np.stack([specs[ch][:, s:e].T for ch in range(len(specs))], axis=-1)
        windows.append(win)
    return np.stack(windows, axis=0)

# --- LOAD DATA E SPLIT POR DISTÂNCIA GRUPO ---
def load_data_and_train(data_dir):
    entries = parse_metadata()
    # agrupa 4 arquivos por mesma distância 3D
    dist_groups = defaultdict(list)
    for fname, dist in entries:
        dist_groups[dist].append((fname, dist))
    train_files = []
    val_files = []
    test_files = []
    np.random.seed(42)
    for dist, files in dist_groups.items():
        perm = np.random.permutation(len(files))
        # 2 treino, 1 val, 1 teste
        train_idx, val_idx, test_idx = perm[:2], perm[2:3], perm[3:4]
        for idx in train_idx: train_files.append(files[idx])
        val_files.extend([files[idx] for idx in val_idx])
        test_files.extend([files[idx] for idx in test_idx])
    def build_dataset(file_list):
        Xs, ys = [], []
        for fname, dist in file_list:
            path = os.path.join(data_dir, fname)
            if not os.path.isfile(path):
                print(f"Aviso: {path} não encontrado")
                continue
            sig, sr = sf.read(path)
            if sig.ndim==1:
                sig = sig[:, None]
            Xw = extract_windows_multichannel(sig, sr)
            Xs.append(Xw)
            ys.append(np.full((Xw.shape[0],), dist, dtype=np.float32))
        X = np.vstack(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.float32)
        return X, y
    X_tr, y_tr = build_dataset(train_files)
    X_val, y_val = build_dataset(val_files)
    X_te, y_te = build_dataset(test_files)
    # normalização global (treino)
    mean = X_tr.mean(axis=(0,1,2), keepdims=True)
    std = X_tr.std(axis=(0,1,2), keepdims=True) + 1e-8
    for arr in (X_tr, X_val, X_te):
        arr[...] = (arr - mean) / std
    np.save('mean.npy', mean)
    np.save('std.npy', std)
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)

# --- MODEL ---
def build_cnn(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, name='dist_output')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# --- TESTE MULTI-MIC ---
def test_single(data_dir, test_file):
    entries = parse_metadata()
    # distância 3D verdadeira do arquivo
    dist_true = next(d for f,d in entries if f==test_file)
    # agrupa os 4 arquivos da mesma distância
    group = [f for f,d in entries if d==dist_true]
    preds_group = []
    model = load_model('best_cnn.wav.h5', compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    mean_arr, std_arr = np.load('mean.npy'), np.load('std.npy')
    for fname in group:
        path = os.path.join(data_dir, fname)
        sig, sr = sf.read(path)
        if sig.ndim==1:
            sig = sig[:, None]
        Xw = extract_windows_multichannel(sig, sr)
        Xn = ((Xw - mean_arr) / std_arr).astype(np.float32)
        preds_group.append(model.predict(Xn, verbose=0).mean())
    final_pred = float(np.mean(preds_group))
    print(f"Predição multi-mic (3D): {final_pred:.2f} m | Erro: {abs(final_pred-dist_true):.2f} m")

# --- MAIN ---
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='pasta com .wav')
    parser.add_argument('--test_file', help='arquivo para teste (um dos 4 wav)')
    args = parser.parse_args()
    if args.test_file:
        test_single(args.data_dir, args.test_file)
    else:
        (X_tr,y_tr),(X_val,y_val),(X_te,y_te) = load_data_and_train(args.data_dir)
        print(f"Treino: {X_tr.shape}, Val: {X_val.shape}, Teste: {X_te.shape}")
        model = build_cnn(X_tr.shape[1:])
        cp = callbacks.ModelCheckpoint('best_cnn.wav.h5', save_best_only=True, monitor='val_loss')
        model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[cp])
        loss, mae = model.evaluate(X_te, y_te)
        print(f"Teste MAE: {mae:.3f} m")
        model.save('final_cnn.wav.h5')
