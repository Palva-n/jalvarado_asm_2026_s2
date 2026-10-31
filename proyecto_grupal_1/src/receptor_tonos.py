from machine import *
import time
import gc
import math
from array import array
from ulab import numpy as np

# Modo PWM de la fuente de la Pico W (baja el ruido del ADC)
Pin("WL_GPIO1", Pin.OUT, value=1)

# ---------------- Parametros ----------------
N = 1024
FS_OBJETIVO = 16000
FS_REAL = 16220                      # Hz, fs efectiva medida del timer
FREC_ESPERADAS = [440, 880, 1320]    # Hz, las de la parte 1
TOLERANCIA_HZ = 30                   # un poco mas de 1 bin (fs/N ~ 15.8 Hz)
FACTOR_UMBRAL = 10                   # un pico debe ser 10 veces la mediana del espectro
MAX_PICOS = 5

# ---------------- Preparacion (una sola vez) ----------------
datos = array('H', (0 for _ in range(N)))
adc = ADC(Pin(26))

# Ventana de Hann: w[n] = 0.5 * (1 - cos(2 pi n / (N - 1)))
ventana_lista = []
for n in range(N):
    ventana_lista.append(0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))))
ventana = np.array(ventana_lista)
del ventana_lista

indice = 0
terminado = False


def registrar_muestra(timer):
    global indice, terminado
    if indice < N:
        datos[indice] = adc.read_u16()
        indice += 1
    else:
        timer.deinit()
        terminado = True


def capturar_bloque():
    global indice, terminado
    indice = 0
    terminado = False
    tim = Timer(-1)
    tim.init(freq=FS_OBJETIVO, mode=Timer.PERIODIC, callback=registrar_muestra)
    while not terminado:
        pass

def magnitud_espectro(bloque):
    # 1. Pasar a flotante y quitar el DC
    x = np.array(bloque, dtype=np.float)
    x = x - np.mean(x)
    # 2. Ventana de Hann
    x = x * ventana
    # 3. FFT
    resultado = np.fft.fft(x)
    if isinstance(resultado, tuple):
        re, im = resultado
    else:
        # ulab complejo: sacamos parte real e imaginaria
        re = np.real(resultado)
        im = np.imag(resultado)
    mag = np.sqrt(re * re + im * im)
    return mag


def buscar_picos(mag):
    mitad = N // 2
    umbral = FACTOR_UMBRAL * np.median(mag[:mitad])
    picos = []
    for k in range(1, mitad - 1):
        if mag[k] > umbral and mag[k] > mag[k - 1] and mag[k] >= mag[k + 1]:
            picos.append((mag[k], k))
    picos.sort(reverse=True)
    return picos[:MAX_PICOS]


# ---------------- Ciclo principal ----------------
while True:
    capturar_bloque()

    t0 = time.ticks_us()
    mag = magnitud_espectro(datos)
    picos = buscar_picos(mag)
    t_analisis = time.ticks_diff(time.ticks_us(), t0)

    print("--- Bloque (analisis: {} us) ---".format(t_analisis))

    detectadas = []
    for amplitud, k in picos:
        f = k * FS_REAL / N
        detectadas.append(f)
        print("  pico: {:.1f} Hz (bin {}), amplitud {:.0f}".format(f, k, amplitud))

    for esperada in FREC_ESPERADAS:
        encontrada = False
        for f in detectadas:
            if abs(f - esperada) <= TOLERANCIA_HZ:
                encontrada = True
        if encontrada:
            print("  {} Hz: DETECTADA".format(esperada))
        else:
            print("  {} Hz: no detectada".format(esperada))

    gc.collect()
    time.sleep_ms(500)