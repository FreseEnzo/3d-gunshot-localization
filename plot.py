#!/usr/bin/env python3
"""
test_all_wavs.py

Extensão do script wav_distance_cnn.py para testar todas as amostras .wav
e gerar um gráfico 2D comparando distâncias reais com predições.

Uso:
  $ python test_all_wavs.py --data_dir /data/wavs --model_path best_cnn.wav.h5

Dependências:
  pip install numpy scipy scikit-learn tensorflow joblib soundfile librosa matplotlib
"""
import os
import argparse
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.keras.models import load_model
from matplotlib.patches import Patch

# --- CONFIGURAÇÕES ---
SR_TARGET = 48000        # Hz após resampling
DURATION = 0.1           # segundos de muzzle blast
N_FFT = 512
HOP = 256
N_MELS = 16
WIN_COLS = 16            # colunas Mel por janela
STRIDE = 8               # deslocamento entre janelas
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

def predict_distance(model, signal, sr, mean_arr, std_arr):
    """Prediz a distância para um único arquivo de áudio"""
    if signal.ndim == 1:
        signal = signal[:, None]
    Xw = extract_windows_multichannel(signal, sr)
    Xn = ((Xw - mean_arr) / std_arr).astype(np.float32)
    # Média das predições de todas as janelas
    return float(model.predict(Xn, verbose=0).mean())

def test_all_wavs(data_dir, model_path, show_individual_points=True):
    """Testa todas as amostras .wav e gera um gráfico"""
    # Carregar modelo e parâmetros de normalização
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    mean_arr = np.load('mean.npy')
    std_arr = np.load('std.npy')
    
    # Obter metadados
    entries = parse_metadata()
    
    # Dicionário para agrupar resultados por distância real
    results_by_dist = defaultdict(list)
    all_preds = []
    all_truths = []
    
    # Processar todos os arquivos
    for fname, dist_true in entries:
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            print(f"Aviso: {path} não encontrado")
            continue
            
        sig, sr = sf.read(path)
        pred_dist = predict_distance(model, sig, sr, mean_arr, std_arr)
        
        # Armazenar resultados
        results_by_dist[dist_true].append((fname, pred_dist))
        all_preds.append(pred_dist)
        all_truths.append(dist_true)
        
        print(f"{fname}: Real = {dist_true:.2f}m, Predição = {pred_dist:.2f}m, "
              f"Erro = {abs(pred_dist-dist_true):.2f}m")
    
    # Calcular métricas
    errors = np.array(all_preds) - np.array(all_truths)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # Média por grupo de distância
    dist_groups = sorted(results_by_dist.keys())
    avg_preds = [np.mean([p for _, p in results_by_dist[d]]) for d in dist_groups]
    
    # Criar gráfico melhorado
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid')  # Estilo mais atraente
    
    # 1. Linha ideal (y=x)
    min_val = min(min(all_truths), min(all_preds)) - 0.2
    max_val = max(max(all_truths), max(all_preds)) + 0.2
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ideal (y=x)')
    
    # 2. Pontos individuais (opcional)
    if show_individual_points:
        # Definir cores diferentes para cada grupo de distância
        colors = plt.cm.viridis(np.linspace(0, 1, len(dist_groups)))
        for i, dist in enumerate(dist_groups):
            points = results_by_dist[dist]
            x_vals = [dist] * len(points)
            y_vals = [p for _, p in points]
            plt.scatter(x_vals, y_vals, color=colors[i], alpha=0.4, s=40, 
                        label=f'Pontos individuais ({dist:.1f}m)' if i == 0 else "")
    
    # 3. Média por grupo (distância real) - conectados por linha
    plt.plot(dist_groups, avg_preds, 'ro-', linewidth=2, markersize=10, 
            label='Média por distância')
    
    # 4. Pontos de valores reais para comparação
    plt.plot(dist_groups, dist_groups, 'bo-', linewidth=2, markersize=10, 
            label='Valores reais')
    
    # 5. Adicionar anotações para os pontos médios
    for i, d in enumerate(dist_groups):
        plt.annotate(f'{avg_preds[i]:.2f}m', 
                    (dist_groups[i], avg_preds[i]),
                    textcoords="offset points",
                    xytext=(5, -15), 
                    ha='center',
                    fontsize=9,
                    color='darkred')
    
    # 6. Configurações do gráfico
    plt.title(f'Distância Real vs. Predita (MAE={mae:.3f}m, RMSE={rmse:.3f}m)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Distância Real (m)', fontsize=12)
    plt.ylabel('Distância Predita (m)', fontsize=12)
    
    # 7. Melhorar a legenda
    if show_individual_points:
        # Criar legenda personalizada para não mostrar cada grupo de pontos individuais
        handles, labels = plt.gca().get_legend_handles_labels()
        # Alterar o primeiro handle de pontos individuais
        custom_handles = [
            handles[0],  # Linha ideal
            handles[1],  # Pontos individuais (primeiro grupo)
            handles[-2], # Linha média
            handles[-1]  # Linha real
        ]
        custom_labels = [
            'Ideal (y=x)',
            'Pontos individuais',
            'Médias das predições',
            'Valores reais'
        ]
        plt.legend(custom_handles, custom_labels, loc='best', fontsize=10)
    else:
        plt.legend(loc='best', fontsize=10)
    
    # 8. Adicionar informações de erro no gráfico
    info_text = (
        f"MAE: {mae:.3f}m\n"
        f"RMSE: {rmse:.3f}m\n"
        f"Amostras: {len(all_preds)}"
    )
    plt.annotate(info_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                va='top', ha='left')
    
    # 9. Igualar eixos e melhorar o layout
    plt.axis('equal')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # 10. Adicionar linha de erro zero
    for i, d in enumerate(dist_groups):
        plt.plot([d, d], [d, avg_preds[i]], 'r-', alpha=0.3, linewidth=1.5)
    
    # 11. Salvar com qualidade e mostrar
    plt.savefig('distance_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Resumo das métricas
    print("\n=== Métricas de Desempenho ===")
    print(f"MAE: {mae:.3f} metros")
    print(f"RMSE: {rmse:.3f} metros")
    
    # Mostrar erro médio por grupo de distância
    print("\n=== Erro Médio por Distância ===")
    for d in dist_groups:
        preds = [p for _, p in results_by_dist[d]]
        err = np.mean(np.abs(np.array(preds) - d))
        print(f"Distância {d:.2f}m: Erro médio = {err:.3f}m")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='pasta com arquivos .wav')
    parser.add_argument('--model_path', default='best_cnn.wav.h5', help='caminho para o modelo treinado')
    parser.add_argument('--show_individual', action='store_true', help='mostrar pontos individuais no gráfico')
    args = parser.parse_args()
    
    test_all_wavs(args.data_dir, args.model_path, show_individual_points=args.show_individual)