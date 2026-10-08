import json
import csv
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt 

CASOS = ["caso_1", "caso_2", "caso_3"]
RUTA = "../pruebas"
INICIO = 16000 
N = 32768


def cargar_wav(ruta):
    fs, x = wavfile.read(ruta)
    return fs, x.astype(np.float64) / 32768.0   # int16 -> [-1, 1]


def snr_db(limpia, contaminada):
    error = contaminada - limpia
    return 10 * np.log10(np.sum(limpia**2) / np.sum(error**2))

def espectro(fs, signal): 
    w = np.hanning(len(signal))
    X = np.fft.fft(signal * w)[:len(signal)//2]
    f = np.arange(len(signal)//2) * fs / len(signal)
    return f, X


filas = []
TD_limpia = []
TD_sucia = []
TD_ruido = [] 

for caso in CASOS:
    with open(f"{RUTA}/{caso}/metadatos_prueba.json", encoding="utf-8") as f:
        meta = json.load(f)

    fs, limpia = cargar_wav(f"{RUTA}/{caso}/01_referencia_limpia.wav")
    _, ruido = cargar_wav(f"{RUTA}/{caso}/02_ruido.wav")
    _, sucia = cargar_wav(f"{RUTA}/{caso}/03_entrada_contaminada.wav")

    snr_wav = snr_db(limpia, sucia)
    snr_obj = meta["snr_objetivo_dB"]

    filas.append({
        "caso": caso,
        "semilla": meta["semilla"],
        "fs_Hz": fs,
        "N_muestras": len(limpia),
        "duracion_s": meta["duracion_s"],
        "frecuencias_utiles_Hz": meta["frecuencias_utiles_Hz"],
        "tipo_ruido": meta["tipo_ruido"],
        "snr_objetivo_dB": snr_obj,
        "snr_obtenida_wav_dB": round(snr_wav, 4),
        "diferencia_dB": round(snr_wav - snr_obj, 4),
    })

    TD_limpia.append([fs, limpia])
    TD_sucia.append(sucia)
    TD_ruido.append(ruido) 

for i in range(len(CASOS)): 
    caso = CASOS[i] 

    #magnitud 
    f, X_limpio = espectro(TD_limpia[i][0], TD_limpia[i][1][INICIO: INICIO + N])
    _, X_sucia = espectro(f, TD_sucia[i][INICIO: INICIO + N])
    _, X_ruido = espectro(f, TD_ruido[i][INICIO: INICIO + N])
    fs = TD_limpia[i][0]

    #calculo de la energia sobre el espectro X: 
    j = 0 
    E_limpio = []
    E_ruido = [] 
    sum_limpio = 0 
    sum_ruido = 0 
    for k in range(len(X_limpio)):
        push = [] 
        if ((j * 500 <= f[k]) and (f[k]) < 500 * (j + 1)): 
            sum_limpio += np.abs(X_limpio[k])**2 
            sum_ruido += np.abs(X_ruido[k])**2
        else: 
            j += 1 
            E_limpio.append(sum_limpio) 
            E_ruido.append(sum_ruido)
            sum_limpio = np.abs(X_limpio[k])**2 
            sum_ruido = np.abs(X_ruido[k])**2 
    E_limpio.append(sum_limpio)
    E_ruido.append(sum_ruido)   

    dB_limpio = 10 * np.log10(np.array(E_limpio) + 1e-12)
    dB_ruido = 10 * np.log10(np.array(E_ruido) + 1e-12)
    num_intervalos = len(dB_limpio)
    indices = np.arange(num_intervalos)
    intervalos_hz = [f"{i*1/2}-{(i+1)*1/2}" for i in range(num_intervalos)]


    #fase: 
    # la fase solo tiene sentido donde hay energia, en el resto es basura numerica
    umbral = 0.01 
    fase_limpio = np.degrees(np.angle(X_limpio))
    fase_sucia = np.degrees(np.angle(X_sucia))
    fase_limpio[np.abs(X_limpio) < umbral * np.max(np.abs(X_limpio))] = np.nan 
    fase_sucia[np.abs(X_sucia) < umbral * np.max(np.abs(X_sucia))] = np.nan 

    plt.figure() 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,4)) 
    ax1.plot(f, fase_limpio, 'o', markersize = 3, label = "Limpia", color = 'blue')
    ax2.plot(f, fase_sucia, 'o', markersize = 3, label = "Sucia", color = 'red')

    ax1.set_xlabel("frecuencia [Hz]")
    ax1.set_ylabel("fase [grados]")
    ax1.set_xlim(0, 2500)
    ax1.set_ylim(-180, 180)
    ax1.grid(True) 
    ax1.legend() 

    ax2.set_xlabel("frecuencia [Hz]")
    ax2.set_ylabel("fase [grados]")
    ax2.set_xlim(0, 2500)
    ax2.set_ylim(-180, 180)
    ax2.grid(True) 
    ax2.legend()

    plt.suptitle(caso + ": Fase del Espectro")
    plt.savefig(f"../resultados/figuras/{caso}_fase.png")


    #SNR por banda: dB_limpio - dB_ruido es lo mismo que 10*log10(E_limpio / E_ruido)
    SNR_banda = dB_limpio - dB_ruido 

    plt.figure(figsize=(10, 6))
    plt.bar(indices, SNR_banda, 0.6, color = 'seagreen', edgecolor = 'black')
    plt.axhline(y = 0, color = "black")
    plt.xticks(indices, intervalos_hz)
    plt.title(f'SNR por banda de frecuencia {caso}')
    plt.xlabel('Intervalo de Frecuencia [Hz]')
    plt.ylabel('SNR de la banda [dB]')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"../resultados/figuras/{caso}_snr_banda.png")

    print(f"{caso}: SNR por banda [dB] (solo las 4 primeras bandas)")
    for b in range(4): 
        print(f"   {intervalos_hz[b]} kHz: {SNR_banda[b]:.1f}")


    #numeros para el punto d, sobre el espectro del ruido: 
    P = np.abs(X_ruido)**2 
    pico_mediana = 10 * np.log10(np.max(P) / np.median(P))
    planitud = np.exp(np.mean(np.log(P + 1e-30))) / np.mean(P)
    print(f"   pico/mediana = {pico_mediana:.1f} dB | planitud espectral = {planitud:.3f}")
    
    #grafica de energia por bandas
    ancho_barra = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(indices - ancho_barra/2, dB_limpio, ancho_barra, label='Limpio', color='skyblue', edgecolor='black')
    plt.bar(indices + ancho_barra/2, dB_ruido, ancho_barra, label='Ruido', color='salmon', edgecolor='black')
    plt.xticks(indices, intervalos_hz)
    plt.title(f'Comparación de Energía en dB por Intervalos de Frecuencia {caso}')
    plt.xlabel('Intervalo de Frecuencia [kHz]')
    plt.ylabel('Energía [dB]')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../resultados/figuras/{caso}_energia.png")
    plt.close()