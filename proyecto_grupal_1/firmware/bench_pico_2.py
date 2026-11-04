from machine import *
import time
import gc
import math
from array import array
from ulab import numpy as np


def fft_directa(v):
    r = np.fft.fft(v)
    if isinstance(r, tuple):        # ulab viejo
        return r[0], r[1]
    return np.real(r), np.imag(r)   # ulab nuevo


def ifft_inversa(re, im):
    try:
        return np.fft.ifft(re, im)
    except TypeError:
        return np.fft.ifft(re + 1j * im)


def medir_fft_ifft(N, repeticiones=3):
    # Promedia FFT e IFFT de N puntos
    x = np.array([math.sin(2 * math.pi * 1000 * n / 16000) for n in range(N)])
    t_f = 0
    t_i = 0
    for _ in range(repeticiones):
        t0 = time.ticks_us()
        re, im = fft_directa(x)
        t_f += time.ticks_diff(time.ticks_us(), t0)
        t0 = time.ticks_us()
        ifft_inversa(re, im)
        t_i += time.ticks_diff(time.ticks_us(), t0)
    return t_f / repeticiones / 1000, t_i / repeticiones / 1000   # ms


# A y B: frecuencia de CPU y tamano de FFT 
for freq in (125_000_000, 250_000_000):
    machine.freq(freq)
    print("=== CPU a", machine.freq() / 1e6, "MHz ===")
    for N in (256, 512, 1024):
        gc.collect()
        f, i = medir_fft_ifft(N)
        print("N =", N, " FFT:", f, "ms   IFFT:", i, "ms")

# A partir de aqui se queda a 250 MHz. Si la Pico se reinicia sola o se cuelga, avisar.
machine.freq(250_000_000)

# C: costo del timer del ADC 
adc = ADC(Pin(26))
buf = array('H', (0 for _ in range(4096)))
cuenta = 0


def leer_adc(t):
    global cuenta
    buf[cuenta & 4095] = adc.read_u16()
    cuenta += 1


def fft_con_timer(fs, N=512):
    global cuenta
    cuenta = 0
    tim = Timer(-1)
    tim.init(freq=fs, mode=Timer.PERIODIC, callback=leer_adc, hard=True)
    f, i = medir_fft_ifft(N)
    tim.deinit()
    print("fs =", fs, "Hz, N =", N, " FFT:", f, "ms  IFFT:", i, "ms  (muestras tomadas:", cuenta, ")")


print("=== FFT con el timer del ADC corriendo (250 MHz) ===")
fft_con_timer(8000)
fft_con_timer(16000)

#  D: segundo nucleo 
print("=== Segundo nucleo (_thread) ===")
try:
    import _thread
    resultado = [0, 0]
    terminado = [False]

    def tarea_nucleo1():
        f, i = medir_fft_ifft(512, 5)
        resultado[0] = f
        resultado[1] = i
        terminado[0] = True

    _thread.start_new_thread(tarea_nucleo1, ())
    # Mientras tanto el nucleo 0 tambien hace FFTs
    f0, i0 = medir_fft_ifft(512, 5)
    while not terminado[0]:
        pass
    print("Nucleo 0 -> FFT:", f0, "ms IFFT:", i0, "ms")
    print("Nucleo 1 -> FFT:", resultado[0], "ms IFFT:", resultado[1], "ms")
except Exception as e:
    print("_thread no funciono:", e)

gc.collect()
print("Memoria libre al final:", gc.mem_free(), "bytes")
print("Listo")
