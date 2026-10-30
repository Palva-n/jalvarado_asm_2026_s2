import csv
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent   # tarea1_grupo03/
RUTA = RAIZ / "pruebas"
RUTA_RESULTADOS = RAIZ / "resultados"
N = 1024
HOP = N // 2
EPS = 1e-12   # evita divisiones por cero

# AJUSTA: banda útil de cada caso (mírala en el espectro de cada uno)
CASOS = {
    "caso_1": {"frec_min_util": 300, "frec_max_util": 1500},
    "caso_2": {"frec_min_util": 300, "frec_max_util": 1500},
    "caso_3": {"frec_min_util": 300, "frec_max_util": 1500},
}

ARCHIVO_LIMPIA = "01_referencia_limpia.wav"
ARCHIVO_CONTAMINADA = "03_entrada_contaminada.wav"

PARAMETROS = {
    "suavizado": 0.9,  # promedia la estimacion del ruido entre bloques

    "agresividad": 1.5, # exigencia sobre la frecuencia para saber que es ruido y que no

    "piso_ganancia": 0.1, # valor minimo de la ganancia de wiener

    "prominencia": 10, # picos > 10 veces la magnitud típica del ruido

    "ancho_notch": 3, # bins a cada lado del pico

    "piso_notch": 0.001, # piso de ganan cia aplicado a notch
}


def cargar_wav(ruta):
    fs, x = wavfile.read(ruta)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    return fs, x


def snr_db(limpia, estimada):
    error = limpia - estimada
    return 10 * np.log10(np.sum(limpia**2) / (np.sum(error**2) + EPS))


def mse(limpia, estimada):
    return np.mean((limpia - estimada) ** 2)

def estimar_ruido(potencia, ruido_previo, suavizado, ancho_med=31):
    # mediana a lo largo de la frecuencia: un valor de ruido por bin
    # (los picos aislados no suben la estimación)
    ruido_actual = median_filter(potencia, size=ancho_med, mode="nearest") / np.log(2)
    if ruido_previo is None:
        return ruido_actual
    return suavizado * ruido_previo + (1 - suavizado) * ruido_actual

def ganancia_wiener(potencia, ruido_est, agresividad, piso_ganancia):
    snr = np.maximum(potencia / (agresividad * ruido_est + EPS) - 1, 0)
    ganancia = snr / (snr + 1)
    return np.maximum(ganancia, piso_ganancia)


# ---------------- 3. Ganancia del notch ----------------
def ganancia_notch(magnitud, frecuencias, frec_min_util, frec_max_util, prominencia, ancho_notch, piso_notch, ruido_est):
    ganancia = np.ones_like(magnitud)

    # magnitud típica del ruido = sqrt(potencia de ruido)
    umbral = prominencia * np.sqrt(np.median(ruido_est))
    picos, _ = signal.find_peaks(magnitud, prominence=umbral)
    fuera = (frecuencias[picos] < frec_min_util) | (frecuencias[picos] > frec_max_util)
    for p in picos[fuera]:
        ini = max(0, p - ancho_notch)
        fin = min(len(magnitud), p + ancho_notch + 1)
        ganancia[ini:fin] = piso_notch
    return ganancia


# ---------------- 4. Ganancia total ----------------
def calcular_ganancia(X, ruido_est, frecuencias, frec_min_util, frec_max_util, agresividad, piso_ganancia, prominencia, ancho_notch, piso_notch):
    magnitud = np.abs(X)
    g_wiener = ganancia_wiener(magnitud**2, ruido_est, agresividad, piso_ganancia)
    g_notch = ganancia_notch(magnitud, frecuencias, frec_min_util, frec_max_util, prominencia, ancho_notch, piso_notch, ruido_est)
    return g_wiener * g_notch


# ---------------- 5. Bucle principal ----------------
def procesar(x, fs, N, hop, frec_min_util, frec_max_util, suavizado, agresividad, piso_ganancia, prominencia, ancho_notch, piso_notch, diagnostico=None):

    ventana = np.sqrt(signal.get_window('hann', N))
    frecuencias = np.fft.rfftfreq(N, 1 / fs)

    L = len(x)
    n_bloques = int(np.ceil(L / hop)) + N // hop - 1
    largo_pad = (n_bloques - 1) * hop + N
    x_pad = np.zeros(largo_pad)
    x_pad[N - hop : N - hop + L] = x

    salida = np.zeros(largo_pad)

    # estimación inicial con un bloque completo de señal real
    # (el primer bloque del bucle tiene la mitad en ceros por el padding)
    inicio = np.zeros(N)
    inicio[:min(N, L)] = x[:N]
    ruido_est = estimar_ruido(np.abs(np.fft.rfft(inicio * ventana)) ** 2, None, suavizado)

    for m in range(n_bloques):
        ini = m * hop
        bloque = x_pad[ini:ini + N] * ventana
        X = np.fft.rfft(bloque)

        ruido_est = estimar_ruido(np.abs(X) ** 2, ruido_est, suavizado)
        G = calcular_ganancia(X, ruido_est, frecuencias, frec_min_util, frec_max_util, agresividad, piso_ganancia, prominencia, ancho_notch, piso_notch)

        if diagnostico is not None:
            diagnostico["G"].append(G)
            diagnostico["potencia"].append(np.abs(X) ** 2)
            diagnostico["ruido"].append(ruido_est)

        y = np.fft.irfft(G * X, n=N) * ventana
        salida[ini:ini + N] += y

    return salida[N - hop : N - hop + L]


def espectro_db(x, fs):
    ventana = signal.get_window('hann', len(x))
    X = np.fft.rfft(x * ventana)
    frecuencias = np.fft.rfftfreq(len(x), 1 / fs)
    magnitud_db = 20 * np.log10(np.abs(X) / len(x) + EPS)
    return frecuencias, magnitud_db


def graficar_caso(nombre, fs, limpia, contaminada, salida, duracion_zoom=0.05):
    t = np.arange(len(limpia)) / fs
    zoom = t < duracion_zoom

    fig, ejes = plt.subplots(2, 1, figsize=(10, 8))

    ejes[0].plot(t[zoom], contaminada[zoom], label="Contaminada", alpha=0.5)
    ejes[0].plot(t[zoom], salida[zoom], label="Salida filtrada")
    ejes[0].plot(t[zoom], limpia[zoom], "--", label="Limpia")
    ejes[0].set_xlabel("Tiempo (s)")
    ejes[0].set_ylabel("Amplitud (normalizada)")
    ejes[0].set_title(f"{nombre}: señal en el tiempo (primeros {duracion_zoom} s)")
    ejes[0].legend()
    ejes[0].grid(True)

    for datos, etiqueta, estilo in [(contaminada, "Contaminada", {"alpha": 0.5}),
                                    (salida, "Salida filtrada", {}),
                                    (limpia, "Limpia", {"linestyle": "--"})]:
        f, mag = espectro_db(datos, fs)
        ejes[1].plot(f, mag, label=etiqueta, **estilo)

    ejes[1].set_xlabel("Frecuencia (Hz)")
    ejes[1].set_ylabel("Magnitud (dB)")
    ejes[1].set_title(f"{nombre}: espectro de magnitud")
    ejes[1].legend()
    ejes[1].grid(True)

    fig.tight_layout()
    return fig


def graficar_diagnostico(nombre, fs, limpia, contaminada, salida, diag, t_zoom=3.0, dur_zoom=0.05, f_zoom=2500):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    t = np.arange(len(limpia)) / fs

    # 1. Limpia vs filtrada en el tiempo (tramo estable, lejos del arranque)
    seg = (t >= t_zoom) & (t < t_zoom + dur_zoom)
    ax[0, 0].plot(t[seg], limpia[seg], "--", label="Limpia", linewidth=2)
    ax[0, 0].plot(t[seg], salida[seg], label="Salida filtrada")
    ax[0, 0].set_title(f"{nombre}: limpia vs filtrada")
    ax[0, 0].set_xlabel("Tiempo (s)")
    ax[0, 0].set_ylabel("Amplitud (normalizada)")
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # 2. Espectro limpia vs filtrada (zoom a la banda útil)
    for datos, etiqueta, estilo in [(limpia, "Limpia", {"linestyle": "--", "linewidth": 2}),
                                    (salida, "Salida filtrada", {"alpha": 0.8})]:
        f, mag = espectro_db(datos, fs)
        ax[0, 1].plot(f, mag, label=etiqueta, **estilo)
    ax[0, 1].set_xlim(0, f_zoom)
    ax[0, 1].set_title("Espectro limpia vs filtrada")
    ax[0, 1].set_xlabel("Frecuencia (Hz)")
    ax[0, 1].set_ylabel("Magnitud (dB)")
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # 3. Mapa de ganancia G(frecuencia, bloque): qué está atenuando el filtro
    G = np.array(diag["G"]).T
    im = ax[1, 0].imshow(G, origin="lower", aspect="auto", vmin=0, vmax=1,
                         extent=[0, len(limpia) / fs, 0, fs / 2], cmap="viridis")
    fig.colorbar(im, ax=ax[1, 0], label="Ganancia")
    ax[1, 0].set_title("Ganancia total G")
    ax[1, 0].set_xlabel("Tiempo (s)")
    ax[1, 0].set_ylabel("Frecuencia (Hz)")

    # 4. Potencia del bloque vs ruido estimado (un bloque del medio)
    m = len(diag["potencia"]) // 2
    f = np.fft.rfftfreq(N, 1 / fs)
    ax[1, 1].semilogy(f, diag["potencia"][m], label="Potencia del bloque", alpha=0.6)
    ax[1, 1].semilogy(f, diag["ruido"][m], label="Ruido estimado", linewidth=2)
    ax[1, 1].set_title(f"Estimación de ruido (bloque {m})")
    ax[1, 1].set_xlabel("Frecuencia (Hz)")
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    fig.tight_layout()
    return fig


def graficar_barrido(nombre, fs, limpia, contaminada, cfg, parametro, valores):
    # Corre el mismo caso variando un parámetro y grafica delta SNR
    snr_ent = snr_db(limpia, contaminada)
    deltas = []
    for v in valores:
        p = {**PARAMETROS, parametro: v}
        s = procesar(contaminada, fs, N, HOP, cfg["frec_min_util"],
                     cfg["frec_max_util"], **p)
        deltas.append(snr_db(limpia, s) - snr_ent)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(valores, deltas, "o-")
    if max(valores) / max(min(valores), EPS) >= 10:
        ax.set_xscale("log")
    ax.set_xlabel(parametro)
    ax.set_ylabel("Delta SNR (dB)")
    ax.set_title(f"{nombre}: efecto de {parametro}")
    ax.grid(True)
    fig.tight_layout()
    return fig


# ---------------- Punto f): ruido eliminado vs daño a la señal útil ----------------
def perdida_senal_util_db(limpia, salida):
    # Compara la altura de los picos de la señal limpia con los de la salida.
    # Valores negativos = se atenuaron los tonos útiles.
    w = np.hanning(len(limpia))
    Lm = np.abs(np.fft.rfft(limpia * w))
    Sm = np.abs(np.fft.rfft(salida * w))
    picos, _ = signal.find_peaks(Lm, height=0.1 * Lm.max())
    return 20 * np.log10(Sm[picos].sum() / Lm[picos].sum())


def graficar_compromiso(nombre, fs, limpia, contaminada, cfg, parametro, valores):
    snr_ent = snr_db(limpia, contaminada)
    filas = []
    for v in valores:
        p = {**PARAMETROS, parametro: v}
        s = procesar(contaminada, fs, N, HOP, cfg["frec_min_util"],
                     cfg["frec_max_util"], **p)
        filas.append({
            "caso": nombre,
            "parametro": parametro,
            "valor": v,
            "energia_eliminada_db": 10 * np.log10(np.sum(contaminada**2)
                                                  / (np.sum(s**2) + EPS)),
            "delta_snr_db": snr_db(limpia, s) - snr_ent,
            "perdida_senal_util_db": perdida_senal_util_db(limpia, s),
        })

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(valores, [f["energia_eliminada_db"] for f in filas], "o-",
             label="Energía eliminada (dB)")
    ax1.plot(valores, [f["delta_snr_db"] for f in filas], "s-", label="Delta SNR (dB)")
    ax1.set_xlabel(parametro)
    ax1.set_ylabel("Ruido eliminado / mejora (dB)")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(valores, [f["perdida_senal_util_db"] for f in filas], "^--", color="tab:red", label="Pérdida en tonos útiles (dB)")
    ax2.set_ylabel("Pérdida de señal útil (dB)", color="tab:red")

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc="best")
    ax1.set_title(f"{nombre}: compromiso al variar {parametro}")
    fig.tight_layout()
    return fig, filas


def main():
    RUTA_RESULTADOS.mkdir(exist_ok=True)

    # Prueba de sanidad: con ganancia = 1 la salida debe igualar la entrada
    fs, x = cargar_wav(RUTA / "caso_1" / ARCHIVO_CONTAMINADA)
    prueba = procesar(x, fs, N, HOP, 0, fs / 2, suavizado=0.9, agresividad=1.0, piso_ganancia=1.0, prominencia=10, ancho_notch=3, piso_notch=1.0)
    print("Error máximo con G = 1:", np.max(np.abs(x - prueba)))

    resultados = []
    compromiso = []
    for nombre, cfg in CASOS.items():
        fs, limpia = cargar_wav(RUTA / nombre / ARCHIVO_LIMPIA)
        _, contaminada = cargar_wav(RUTA / nombre / ARCHIVO_CONTAMINADA)

        diag = {"G": [], "potencia": [], "ruido": []}
        salida = procesar(contaminada, fs, N, HOP, cfg["frec_min_util"], cfg["frec_max_util"], **PARAMETROS, diagnostico=diag)

        snr_entrada = snr_db(limpia, contaminada)
        snr_salida = snr_db(limpia, salida)
        resultados.append({
            "caso": nombre,
            "mse_entrada": mse(limpia, contaminada),
            "mse_salida": mse(limpia, salida),
            "snr_entrada_db": snr_entrada,
            "snr_salida_db": snr_salida,
            "delta_snr_db": snr_salida - snr_entrada,
        })
        print(resultados[-1])

        fig = graficar_caso(nombre, fs, limpia, contaminada, salida)
        fig.savefig(RUTA_RESULTADOS / "figuras" / f"{nombre}_resultado_espec.png", dpi=150)

        fig_d = graficar_diagnostico(nombre, fs, limpia, contaminada, salida, diag)
        fig_d.savefig(RUTA_RESULTADOS / "figuras" / f"{nombre}_diagnostico_espec.png", dpi=150)

        for param, valores in [("piso_ganancia", [0.2, 0.1, 0.05, 0.02, 0.01]), ("suavizado", [0.7, 0.9, 0.95]), ("piso_notch", [0.1, 0.01, 0.001])]:
            fig_b = graficar_barrido(nombre, fs, limpia, contaminada, cfg, param, valores)
            fig_b.savefig(RUTA_RESULTADOS / "figuras" / f"{nombre}_barrido_{param}_espec.png", dpi=150)

        # Punto f): más ruido eliminado vs daño a la señal útil
        fig_c, filas = graficar_compromiso(nombre, fs, limpia, contaminada, cfg, "agresividad", [1, 1.5, 2, 3, 5, 8])
        fig_c.savefig(RUTA_RESULTADOS / "figuras" / f"{nombre}_compromiso_espec.png", dpi=150)
        compromiso.extend(filas)

    plt.show()


if __name__ == "__main__":
    main()