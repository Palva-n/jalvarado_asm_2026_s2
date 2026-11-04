from machine import *
import time
import gc
import math
from array import array
from ulab import numpy as np

N = 1024          # tamano de la FFT
HOP = 512         # muestras nuevas por bloque
PERIODO_MS = HOP / 16000 * 1000   # 32 ms

print("Bloque disponible:", PERIODO_MS, "ms")
gc.collect()
print("Memoria libre al inicio:", gc.mem_free(), "bytes")

#  Datos de prueba 
# Una senoidal de 1000 Hz muestreada a 16 kHz
x = np.array([math.sin(2 * math.pi * 1000 * n / 16000) for n in range(N)])
ventana = np.array([0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)])


#  1. FFT directa 
def fft_directa(v):
    r = np.fft.fft(v)
    if isinstance(r, tuple):        # ulab viejo: devuelve (real, imag)
        return r[0], r[1]
    return np.real(r), np.imag(r)   # ulab nuevo: devuelve complejo


t0 = time.ticks_us()
re, im = fft_directa(x * ventana)
t_fft = time.ticks_diff(time.ticks_us(), t0)
print("FFT 1024:", t_fft / 1000, "ms")


# 2. Magnitud y ganancia por bin 
t0 = time.ticks_us()
mag2 = re * re + im * im
ganancia = np.ones(N)               # ganancia de prueba (= no cambiar nada)
re2 = re * ganancia
im2 = im * ganancia
t_gan = time.ticks_diff(time.ticks_us(), t0)
print("Magnitud + ganancia:", t_gan / 1000, "ms")


#  3. FFT inversa 
t0 = time.ticks_us()
try:
    r = np.fft.ifft(re2, im2)       # ulab viejo
except TypeError:
    r = np.fft.ifft(re2 + 1j * im2)  # ulab nuevo
t_ifft = time.ticks_diff(time.ticks_us(), t0)
print("IFFT 1024:", t_ifft / 1000, "ms")


#  4. Filtro biquad en Python puro (notch del nodo 2) 
# Ecuacion de diferencias: y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]
b0, b1, b2, a1, a2 = 0.97, -1.0, 0.97, -1.0, 0.94
bloque = [int(1000 * math.sin(2 * math.pi * 1000 * n / 16000)) for n in range(HOP)]
salida = [0.0] * HOP

t0 = time.ticks_us()
x1 = 0.0
x2 = 0.0
y1 = 0.0
y2 = 0.0
for n in range(HOP):
    xn = bloque[n]
    yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    x2 = x1
    x1 = xn
    y2 = y1
    y1 = yn
    salida[n] = yn
t_biquad = time.ticks_diff(time.ticks_us(), t0)
print("Biquad en Python (512 muestras):", t_biquad / 1000, "ms")


# 5. Biquad con ulab.scipy (si existe) 
try:
    from ulab import scipy as spy
    sos = np.array([[b0, b1, b2, 1.0, a1, a2]])
    zi = np.zeros((1, 2))
    xb = np.array(bloque)
    t0 = time.ticks_us()
    yb, zf = spy.signal.sosfilt(sos, xb, zi=zi)
    t_sos = time.ticks_diff(time.ticks_us(), t0)
    print("sosfilt de ulab (512 muestras):", t_sos / 1000, "ms")
except Exception as e:
    print("ulab.scipy.signal.sosfilt NO disponible:", e)


#  6. Captura con el timer: cuantas muestras salen realmente 
datos = array('H', (0 for _ in range(N)))
adc = ADC(Pin(26))
indice = 0
terminado = False


def registrar(timer):
    global indice, terminado
    if indice < N:
        datos[indice] = adc.read_u16()
        indice += 1
    else:
        timer.deinit()
        terminado = True


tim = Timer(-1)
t0 = time.ticks_us()
tim.init(freq=16000, mode=Timer.PERIODIC, callback=registrar)
while not terminado:
    pass
t_cap = time.ticks_diff(time.ticks_us(), t0)
print("Captura de", N, "muestras pedidas a 16000 Hz:", t_cap / 1000, "ms")
print("fs efectiva =", N / (t_cap / 1e6), "Hz")


#  Resumen
total = t_fft + t_gan + t_ifft + t_biquad
print("----------------------------------------")
print("Suma FFT + ganancia + IFFT + biquad:", total / 1000, "ms de", PERIODO_MS, "ms")
gc.collect()
print("Memoria libre al final:", gc.mem_free(), "bytes")
