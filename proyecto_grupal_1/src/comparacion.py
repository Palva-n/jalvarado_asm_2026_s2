import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile


CARPETA_SRC = Path(__file__).resolve().parent
RAIZ = CARPETA_SRC.parent                     
RUTA_PRUEBAS = RAIZ / "pruebas"
RUTA_RESULTADOS = RAIZ / "resultados"
RUTA_FIGURAS = RUTA_RESULTADOS / "figuras"

sys.path.insert(0, str(CARPETA_SRC))
import filtrado_espectral as fe  # noqa: E402

CASOS = ["caso_1", "caso_2", "caso_3"]

INICIO = 16000

F_NOTCH_HZ = 1750
Q_NOTCH = 10

ANCHO_HZ = 5

REPETICIONES = 5

EPS = 1e-12


def cargar_wav(ruta):
    fs, x = wavfile.read(ruta)
    return fs, x.astype(np.float64) / 32768.0


def cargar_caso(nombre):
    carpeta = RUTA_PRUEBAS / nombre
    with open(carpeta / "metadatos_prueba.json", encoding="utf-8") as f:
        meta = json.load(f)

    fs, limpia = cargar_wav(carpeta / "01_referencia_limpia.wav")
    _, ruido = cargar_wav(carpeta / "02_ruido.wav")
    _, contaminada = cargar_wav(carpeta / "03_entrada_contaminada.wav")

    if fs != meta["fs_Hz"]:
        raise ValueError(f"{nombre}: fs del WAV ({fs}) no coincide con el JSON ({meta['fs_Hz']})")
    return fs, limpia, ruido, contaminada, meta


def coeficientes_notch(fs):
    return signal.iirnotch(F_NOTCH_HZ, Q_NOTCH, fs)


def metodo_temporal(x, fs):
    b, a = coeficientes_notch(fs)
    return signal.lfilter(b, a, x)


def metodo_espectral(x, fs, nombre):
    cfg = fe.CASOS[nombre]
    return fe.procesar(x, fs, fe.N, fe.HOP,
                       cfg["frec_min_util"], cfg["frec_max_util"],
                       **fe.PARAMETROS)


def mse(s, s_est):
    return float(np.mean((s - s_est) ** 2))


def snr_db(s, s_est):
    return float(10 * np.log10(np.sum(s ** 2) / (np.sum((s - s_est) ** 2) + EPS)))


def espectro_potencia(x, fs):
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return f, np.abs(X) ** 2


def mascara_util(f, frecuencias_utiles):
    m = np.zeros(len(f), dtype=bool)
    for fu in frecuencias_utiles:
        m |= np.abs(f - fu) <= ANCHO_HZ
    return m


def mascara_ruido(f, meta):
    tipo = meta["tipo_ruido"]
    if tipo == "tonal":
        m = np.abs(f - meta["frecuencia_interferencia_Hz"]) <= ANCHO_HZ
    elif tipo == "banda":
        lo, hi = meta["banda_ruido_Hz"]
        m = (f >= lo) & (f <= hi)
    else:
        m = np.ones(len(f), dtype=bool)
    return m & ~mascara_util(f, meta["frecuencias_utiles_Hz"])


def energia_util_pct(s, s_est, fs, frecuencias_utiles):
    f, P_s = espectro_potencia(s, fs)
    _, P_y = espectro_potencia(s_est, fs)
    m = mascara_util(f, frecuencias_utiles)
    return float(100 * P_y[m].sum() / (P_s[m].sum() + EPS))


def atenuacion_ruido_db(x, s_est, fs, meta):
    f, P_x = espectro_potencia(x, fs)
    _, P_y = espectro_potencia(s_est, fs)
    m = mascara_ruido(f, meta)
    return float(10 * np.log10(P_x[m].sum() / (P_y[m].sum() + EPS)))


def tiempo_temporal_us(x, fs):
    b, a = coeficientes_notch(fs)
    bloques = [x[i:i + fe.HOP] for i in range(0, len(x) - fe.HOP + 1, fe.HOP)]
    mejor = np.inf
    for _ in range(REPETICIONES):
        zi = np.zeros(max(len(a), len(b)) - 1)
        t0 = time.perf_counter()
        for blq in bloques:
            _, zi = signal.lfilter(b, a, blq, zi=zi)
        mejor = min(mejor, (time.perf_counter() - t0) / len(bloques))
    return mejor * 1e6


def tiempo_espectral_us(x, fs, nombre):
    n_bloques = int(np.ceil(len(x) / fe.HOP)) + fe.N // fe.HOP - 1 
    mejor = np.inf
    for _ in range(REPETICIONES):
        t0 = time.perf_counter()
        metodo_espectral(x, fs, nombre)
        mejor = min(mejor, (time.perf_counter() - t0) / n_bloques)
    return mejor * 1e6


def memoria_valores():
    N, HOP = fe.N, fe.HOP
    n_bins = N // 2 + 1            # bins de una rfft de N puntos
    temporal = {
        "coeficientes b0 b1 b2 a1 a2": 5,   # a0 = 1, esta no se guarda
        "estados x[n-1] x[n-2] y[n-1] y[n-2]": 4,
    }
    espectral = {
        "bloque de entrada (N)": N,
        "ventana sqrt-Hann (N)": N,
        "espectro complejo (2 x bins)": 2 * n_bins,
        "ganancia G (bins)": n_bins,
        "ruido estimado (bins)": n_bins,
        "buffer de solapamiento de salida (N)": N,
    }
    return sum(temporal.values()), sum(espectral.values()), temporal, espectral



COLORES = {"temporal": "#2a78d6", "espectral": "#eb6834"}
ETIQUETAS = {"temporal": "Temporal (notch IIR)", "espectral": "Espectral (FFT/IFFT)"}


def graficar_comparacion(filas, ruta):
    """Dos paneles: delta SNR y energia util conservada, por caso y metodo."""
    casos = CASOS
    metodos = ["temporal", "espectral"]
    x = np.arange(len(casos))
    ancho = 0.36

    def valor(caso, metodo, clave):
        return next(f[clave] for f in filas if f["caso"] == caso and f["metodo"] == metodo)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2))
    paneles = [("delta_snr_db", "Mejora de SNR, ΔSNR (dB)", "{:.1f}"),
               ("energia_util_pct", "Energía útil conservada (%)", "{:.1f}")]

    for ax, (clave, titulo, fmt) in zip(ejes, paneles):
        for i, met in enumerate(metodos):
            vals = [valor(c, met, clave) for c in casos]
            pos = x + (i - 0.5) * (ancho + 0.02)
            barras = ax.bar(pos, vals, ancho, color=COLORES[met], label=ETIQUETAS[met])
            ax.bar_label(barras, labels=[fmt.format(v) for v in vals],
                         padding=2, fontsize=8, color="#333333")
        tipos = [next(f["tipo_ruido"] for f in filas if f["caso"] == c) for c in casos]
        ax.set_xticks(x, [f"{c}\n(ruido {t})" for c, t in zip(casos, tipos)])
        ax.set_title(titulo, fontsize=11)
        ax.grid(axis="y", color="#e5e5e5")
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)

    ejes[0].axhline(0, color="#888888", linewidth=0.8)
    ejes[1].set_ylim(0, 110)
    ejes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Comparación de métodos sobre el mismo segmento (desde t = 1 s)", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    n_temp, n_esp, det_temp, det_esp = memoria_valores()
    memoria = {"sin_filtrar": (None, None),
               "temporal": (n_temp * 4, n_temp * 2),
               "espectral": (n_esp * 4, n_esp * 2)}

    filas = []
    for nombre in CASOS:
        fs, limpia, ruido, contaminada, meta = cargar_caso(nombre)

        salidas = {
            "sin_filtrar": contaminada,
            "temporal": metodo_temporal(contaminada, fs),
            "espectral": metodo_espectral(contaminada, fs, nombre),
        }

        s = limpia[INICIO:]
        x = contaminada[INICIO:]
        snr_in = snr_db(s, x)
        mse_in = mse(s, x)

        tiempos = {"sin_filtrar": None,
                   "temporal": tiempo_temporal_us(contaminada, fs),
                   "espectral": tiempo_espectral_us(contaminada, fs, nombre)}

        for metodo, y_completa in salidas.items():
            y = y_completa[INICIO:]
            snr_out = snr_db(s, y)
            filas.append({
                "caso": nombre,
                "tipo_ruido": meta["tipo_ruido"],
                "metodo": metodo,
                "snr_entrada_db": round(snr_in, 3),
                "snr_salida_db": round(snr_out, 3),
                "delta_snr_db": round(snr_out - snr_in, 3),
                "mse_entrada": f"{mse_in:.4e}",
                "mse_salida": f"{mse(s, y):.4e}",
                "energia_util_pct": round(energia_util_pct(s, y, fs, meta["frecuencias_utiles_Hz"]), 2),
                "atenuacion_ruido_db": round(atenuacion_ruido_db(x, y, fs, meta), 2),
                "tiempo_us_por_bloque": None if tiempos[metodo] is None else round(tiempos[metodo], 2),
                "limite_tiempo_real_us": round(fe.HOP / fs * 1e6, 1),
                "memoria_bytes_float32": memoria[metodo][0],
                "memoria_bytes_q15": memoria[metodo][1],
            })

    # metricas.csv
    ruta_csv = RUTA_RESULTADOS / "metricas.csv"
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    # tabla en terminal
    print(f"Segmento evaluado: muestras {INICIO} en adelante | bloque = {fe.HOP} muestras nuevas "
          f"(limite de tiempo real {fe.HOP / 16000 * 1e3:.0f} ms a 16 kHz)\n")
    enc = f"{'caso':7s} {'ruido':7s} {'metodo':12s} {'SNR':>7s} {'dSNR':>7s} {'MSE':>10s} {'E.util%':>8s} {'Aten.dB':>8s} {'t[us]':>8s}"
    print(enc)
    print("-" * len(enc))
    for r in filas:
        t = "-" if r["tiempo_us_por_bloque"] is None else f"{r['tiempo_us_por_bloque']:.1f}"
        print(f"{r['caso']:7s} {r['tipo_ruido']:7s} {r['metodo']:12s} {r['snr_salida_db']:7.2f} "
              f"{r['delta_snr_db']:7.2f} {r['mse_salida']:>10s} {r['energia_util_pct']:8.1f} "
              f"{r['atenuacion_ruido_db']:8.2f} {t:>8s}")

    print("\nMemoria aproximada (valores almacenados):")
    print(f"  temporal : {n_temp} valores -> {n_temp*4} B en float32, {n_temp*2} B en Q15")
    for k, v in det_temp.items():
        print(f"      {k}: {v}")
    print(f"  espectral: {n_esp} valores -> {n_esp*4} B en float32, {n_esp*2} B en Q15")
    for k, v in det_esp.items():
        print(f"      {k}: {v}")

    # figura
    ruta_fig = RUTA_FIGURAS / "comparacion_metodos.png"
    graficar_comparacion([r for r in filas if r["metodo"] != "sin_filtrar"], ruta_fig)
    print(f"\nGuardado: {ruta_csv}\nGuardado: {ruta_fig}")


if __name__ == "__main__":
    main()
