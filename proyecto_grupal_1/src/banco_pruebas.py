import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

CARPETA_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA_SRC))
import simulacion_sistema as sim

RUTA_RESULTADOS = CARPETA_SRC.parent / "resultados"
RUTA_FIGURAS = RUTA_RESULTADOS / "figuras"
RUTA_AUDIOS = RUTA_RESULTADOS / "audios"

FS = sim.FS            # 16000 Hz
N = sim.N
HOP = sim.HOP
DURACION = 6.0         # segundos por prueba
REPETICIONES = 3
SNR_ENTRADA_DB = 5.0
PICO = 0.8
EPS = 1e-12



# 1. Generadores de senal util (cada uno devuelve la senal limpia s[n])

def util_tono(rng, t):
    # Un tono de 1000 Hz con fase aleatoria
    return np.sin(2 * np.pi * 1000 * t + rng.uniform(-np.pi, np.pi))


def util_tres_tonos(rng, t):
    # Tres tonos con amplitudes decrecientes (los mismos de la parte 1)
    s = np.zeros(len(t))
    for f, a in zip([440, 880, 1320], [1.0, 0.55, 0.30]):
        s += a * np.sin(2 * np.pi * f * t + rng.uniform(-np.pi, np.pi))
    return s


def util_audio(rng, t):
    # "Voz" sintetica: fundamental que oscila entre 120 y 200 Hz con 12
    # armonicos que decaen, y modulacion de amplitud tipo silabas (4 Hz)
    f0 = 160 + 40 * np.sin(2 * np.pi * 0.5 * t)
    fase = 2 * np.pi * np.cumsum(f0) / FS
    s = np.zeros(len(t))
    for k in range(1, 13):
        s += (1 / k) * np.sin(k * fase + rng.uniform(-np.pi, np.pi))
    envolvente = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return s * envolvente


def util_transitoria(rng, t):
    # Rafagas de un seno de 800 Hz con decaimiento exponencial cada 0.5 s
    s = np.zeros(len(t))
    periodo = int(0.5 * FS)
    for ini in range(int(0.2 * FS), len(t), periodo):
        n = np.arange(min(periodo, len(t) - ini))
        s[ini:ini + len(n)] += np.exp(-n / (0.05 * FS)) * np.sin(2 * np.pi * 800 * n / FS)
    return s


def util_tres_tonos_solapados(rng, t):
    # Igual que la P2; el ruido de la P5 se solapa con el tono de 1320 Hz
    return util_tres_tonos(rng, t)



# 2. Generadores de ruido (devuelven ruido sin escalar)

def ruido_sinusoidal(rng, t):
    # Interferencia sinusoidal separada: 2200 Hz
    return np.sin(2 * np.pi * 2200 * t + rng.uniform(-np.pi, np.pi))


def ruido_banda(rng, t, f_lo, f_hi):
    # Ruido blanco pasado por un Butterworth pasa-banda de orden 6 (en SOS)
    sos = signal.butter(6, [f_lo, f_hi], btype="bandpass", fs=FS, output="sos")
    return signal.sosfilt(sos, rng.standard_normal(len(t)))


def ruido_alta_frecuencia(rng, t):
    # Ruido de banda limitada fuera de la banda util: 3 kHz a 7 kHz
    return ruido_banda(rng, t, 3000, 7000)


def ruido_blanco(rng, t):
    # Ruido de banda ancha
    return rng.standard_normal(len(t))


def ruido_impulsivo(rng, t):
    # Impulsos aislados: ~20 por segundo, signo aleatorio, amplitud grande
    r = np.zeros(len(t))
    pos = rng.random(len(t)) < 20 / FS
    r[pos] = rng.choice([-1.0, 1.0], size=int(pos.sum())) * rng.uniform(3, 6, size=int(pos.sum()))
    return r


def ruido_solapado(rng, t):
    # Banda limitada que se solapa con el tono de 1320 Hz: 1150 a 1600 Hz
    return ruido_banda(rng, t, 1150, 1600)



# 3. Definicion de las pruebas (Tabla 1 del enunciado)

BANDA_UTIL_DEFECTO = (300, 1500)   # Hz, la del sistema con tonos

PRUEBAS = {
    "P1": {"util": util_tono, "ruido": ruido_sinusoidal,
           "descr": "Tono + interferencia sinusoidal separada (2200 Hz)"},
    "P2": {"util": util_tres_tonos, "ruido": ruido_alta_frecuencia,
           "descr": "Tres tonos + ruido de alta frecuencia (3-7 kHz)"},
    "P3": {"util": util_audio, "ruido": ruido_blanco,
           "descr": "Audio sintetico + ruido de banda ancha",
           "banda_util": (80, 3000)},   # la voz tiene armonicos fuera de 300-1500 Hz
    "P4": {"util": util_transitoria, "ruido": ruido_impulsivo,
           "descr": "Senal transitoria + ruido impulsivo"},
    "P5": {"util": util_tres_tonos_solapados, "ruido": ruido_solapado,
           "descr": "Tres tonos + banda limitada solapada con 1320 Hz"},
}


def generar_prueba(nombre, repeticion):
    # La semilla depende de la prueba y de la repeticion, asi se puede repetir
    rng = np.random.default_rng(1000 * int(nombre[1]) + repeticion)
    t = np.arange(int(DURACION * FS)) / FS

    s = PRUEBAS[nombre]["util"](rng, t)
    r = PRUEBAS[nombre]["ruido"](rng, t)

    # Escalar el ruido para tener la SNR de entrada deseada
    p_s = np.mean(s ** 2)
    p_r = np.mean(r ** 2)
    r = r * np.sqrt(p_s / (p_r * 10 ** (SNR_ENTRADA_DB / 10)))

    x = s + r
    escala = PICO / np.max(np.abs(x))    # evitar que sature el ADC
    return s * escala, x * escala



# 4. Pipeline de tres nodos sobre una senal arbitraria

MARGEN_PASABAJOS_HZ = 300   # el pasabajos FIR corta 300 Hz por encima de la banda util


def correr_pipeline(contaminada, banda_util=BANDA_UTIL_DEFECTO):
    # La banda util es un parametro ajustable: los nodos 1 y 3 la leen del modulo
    sim.F_MIN_UTIL, sim.F_MAX_UTIL = banda_util
    n_bloques = len(contaminada) // HOP
    # Tres rutas: (a) solo notch IIR, (b) notch IIR + pasabajos FIR, (c) espectral
    fc = banda_util[1] + MARGEN_PASABAJOS_HZ
    nodo1, nodo3 = sim.Nodo1(), sim.Nodo3()
    nodo2_notch, nodo2 = sim.Nodo2(), sim.Nodo2(fc_pasabajos=fc)
    salida_notch = np.full(n_bloques * HOP, np.nan)
    salida_temporal = np.full(n_bloques * HOP, np.nan)
    salida_espectral = np.full(n_bloques * HOP, np.nan)

    for m in range(n_bloques):
        codigos = sim.adc_12bits(contaminada[m * HOP:(m + 1) * HOP])
        trama_1 = nodo1.procesar_bloque(codigos)

        r2n = nodo2_notch.procesar_bloque(trama_1)
        if r2n is not None:
            salida_notch[r2n[0] * HOP:(r2n[0] + 1) * HOP] = r2n[1]

        r2 = nodo2.procesar_bloque(trama_1)
        if r2 is not None:
            n, salida, _ = r2
            salida_temporal[n * HOP:(n + 1) * HOP] = salida

        r3 = nodo3.procesar_bloque(trama_1)
        if r3 is not None:
            n, salida = r3
            salida_espectral[n * HOP:(n + 1) * HOP] = salida

    # El FIR de fase lineal retrasa la salida (taps-1)/2 muestras; se alinea con la
    # referencia para que MSE y SNR no penalicen un retardo constante.
    d = nodo2.retardo_fir
    salida_temporal = np.concatenate([salida_temporal[d:], np.full(d, np.nan)])
    return salida_notch, salida_temporal, salida_espectral



# 5. Metricas (ecuaciones 2, 3 y 4 del enunciado, mas las pedidas en 3.5)

def mse(s, y):
    return float(np.mean((s - y) ** 2))


def snr_db(s, y):
    return float(10 * np.log10(np.sum(s ** 2) / (np.sum((s - y) ** 2) + EPS)))


def potencia_espectral(x):
    w = np.hanning(len(x))
    return np.abs(np.fft.rfft(x * w)) ** 2


def energia_util_pct(s, y):
    # Bins "utiles" = donde la senal limpia tiene al menos -30 dB respecto a su pico.
    # Porcentaje de la energia de la limpia en esos bins que conserva la salida.
    P_s, P_y = potencia_espectral(s), potencia_espectral(y)
    util = P_s > P_s.max() * 1e-3
    return float(100 * P_y[util].sum() / (P_s[util].sum() + EPS))


def atenuacion_interferencia_db(s, x, y):
    # En los bins donde el ruido domina sobre la senal, cuanto bajo la potencia.
    P_s = potencia_espectral(s)
    P_r = potencia_espectral(x - s)        # ruido de entrada
    P_e = potencia_espectral(y - s)        # error (ruido residual + distorsion) de salida
    dominado = P_r > P_s
    return float(10 * np.log10(P_r[dominado].sum() / (P_e[dominado].sum() + EPS)))


def calcular_metricas(s, x, y):
    # Se evalua en regimen estable: se descartan los primeros bloques (1 s)
    # y los bloques sin salida (nan).
    ini = sim.INICIO_METRICAS
    s, x, y = s[ini:len(y)], x[ini:len(y)], y[ini:]
    validas = ~np.isnan(y)
    s, x, y = s[validas], x[validas], y[validas]
    return {
        "mse": mse(s, y),
        "snr_entrada": snr_db(s, x),
        "snr_salida": snr_db(s, y),
        "delta_snr": snr_db(s, y) - snr_db(s, x),
        "energia_util_pct": energia_util_pct(s, y),
        "atenuacion_db": atenuacion_interferencia_db(s, x, y),
    }



# 6. Figuras

def graficar(nombre, s, x, y_t, y_e, y_n):
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)

    # Espectros (sin los nan)
    ini = sim.INICIO_METRICAS
    fin = len(y_t)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for etiqueta, v, color in [("Limpia", s, "#555555"), ("Contaminada", x, "#d62728"),
                               ("Notch IIR", y_n, "#9467bd"), ("Notch+FIR", y_t, "#2a78d6"),
                               ("Espectral", y_e, "#2ca02c")]:
        v = v[ini:fin]
        v = np.nan_to_num(v)
        P = potencia_espectral(v)
        f = np.fft.rfftfreq(len(v), 1 / FS)
        ax.plot(f, 10 * np.log10(P / P.max() + 1e-12), label=etiqueta, color=color, lw=0.9)
    ax.set(xlabel="Frecuencia [Hz]", ylabel="Potencia [dB rel.]", ylim=(-100, 5),
           title=f"{nombre}: {PRUEBAS[nombre]['descr']}")
    ax.grid(True, color="#e5e5e5")
    ax.legend(frameon=False, ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(RUTA_FIGURAS / f"{nombre}_espectros.png", dpi=150)
    plt.close(fig)

    # Zoom temporal de 60 ms
    a = int(3.0 * FS)
    b = a + int(0.06 * FS)
    t = np.arange(a, b) / FS
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, s[a:b], label="Limpia", color="#555555")
    ax.plot(t, x[a:b], label="Contaminada", color="#d62728", lw=0.6, alpha=0.7)
    ax.plot(t, y_n[a:b], label="Notch IIR", color="#9467bd", lw=0.9)
    ax.plot(t, y_t[a:b], label="Notch+FIR", color="#2a78d6", lw=0.9)
    ax.plot(t, y_e[a:b], label="Espectral", color="#2ca02c", lw=0.9)
    ax.set(xlabel="Tiempo [s]", ylabel="Amplitud", title=f"{nombre}: zoom temporal")
    ax.grid(True, color="#e5e5e5")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(RUTA_FIGURAS / f"{nombre}_temporal.png", dpi=150)
    plt.close(fig)


def guardar_wavs(nombre, s, x, y_t, y_e, y_n):
    # Para la revision cualitativa del sonido (prueba auditiva)
    RUTA_AUDIOS.mkdir(parents=True, exist_ok=True)
    for etiqueta, v in [("limpia", s), ("contaminada", x), ("notch", y_n), ("temporal", y_t), ("espectral", y_e)]:
        v = np.nan_to_num(v)
        wavfile.write(RUTA_AUDIOS / f"{nombre}_{etiqueta}.wav", FS, (v * 32767).astype(np.int16))



# 7. Programa principal

CAMPOS = ["mse", "snr_entrada", "snr_salida", "delta_snr", "energia_util_pct", "atenuacion_db"]


def main():
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    filas = []

    for nombre in PRUEBAS:
        for rep in range(1, REPETICIONES + 1):
            s, x = generar_prueba(nombre, rep)
            y_n, y_t, y_e = correr_pipeline(x, PRUEBAS[nombre].get("banda_util", BANDA_UTIL_DEFECTO))

            for metodo, y in [("notch", y_n), ("temporal", y_t), ("espectral", y_e)]:
                fila = {"prueba": nombre, "repeticion": rep, "metodo": metodo}
                fila.update(calcular_metricas(s, x, y))
                filas.append(fila)

            if rep == 1:
                graficar(nombre, s, x, y_t, y_e, y_n)
                guardar_wavs(nombre, s, x, y_t, y_e, y_n)
        print(f"{nombre} listo: {PRUEBAS[nombre]['descr']}")

    # CSV completo
    with open(RUTA_RESULTADOS / "pruebas_P1_P5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["prueba", "repeticion", "metodo"] + CAMPOS)
        w.writeheader()
        w.writerows(filas)

    # Resumen: promedio y desviacion estandar por prueba y metodo
    resumen = []
    print("\nPrueba Metodo     dSNR [dB]        E.util [%]      Atenuacion [dB]    MSE")
    for nombre in PRUEBAS:
        for metodo in ("notch", "temporal", "espectral"):
            sel = [f for f in filas if f["prueba"] == nombre and f["metodo"] == metodo]
            fila = {"prueba": nombre, "metodo": metodo}
            for c in CAMPOS:
                valores = np.array([f[c] for f in sel])
                fila[c + "_media"] = float(valores.mean())
                fila[c + "_std"] = float(valores.std(ddof=1))
            resumen.append(fila)
            print(f"{nombre:6s} {metodo:9s} {fila['delta_snr_media']:6.2f} +- {fila['delta_snr_std']:4.2f}   "
                  f"{fila['energia_util_pct_media']:6.1f} +- {fila['energia_util_pct_std']:4.1f}   "
                  f"{fila['atenuacion_db_media']:6.1f} +- {fila['atenuacion_db_std']:4.1f}   "
                  f"{fila['mse_media']:.2e}")

    with open(RUTA_RESULTADOS / "resumen_P1_P5.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        w.writerows(resumen)
    print(f"\nGuardado en {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
