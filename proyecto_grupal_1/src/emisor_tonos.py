import numpy as np
from scipy.io import wavfile

FS = 16000                      # Hz
DURACION = 10                   # segundos
FRECUENCIAS = [440, 880, 1320]  # Hz, las de la parte 1
AMPLITUDES = [1.0, 0.55, 0.3]
FASES_GRADOS = [0, 35, -50]
PICO = 0.6                      # volumen: 1.0 es el maximo, baja si suena fuerte

# 1. Tiempo de cada muestra
t = np.arange(int(FS * DURACION)) / FS

# 2. Suma de senoidales: x[n] = sum A_i sin(2 pi f_i n / fs + fase_i)
x = np.zeros(len(t))
for f, A, fase in zip(FRECUENCIAS, AMPLITUDES, FASES_GRADOS):
    x += A * np.sin(2 * np.pi * f * t + np.deg2rad(fase))

# 3. Normalizar al volumen deseado
x = x / np.max(np.abs(x)) * PICO

# 4. Rampa corta al inicio y al final para evitar un chasquido
n_rampa = int(0.02 * FS)
rampa = np.sin(np.linspace(0, np.pi / 2, n_rampa)) ** 2
x[:n_rampa] *= rampa
x[-n_rampa:] *= rampa[::-1]

# 5. Guardar como wav de 16 bits
wavfile.write("tonos.wav", FS, (x * 32767).astype(np.int16))
print("Guardado tonos.wav:", FRECUENCIAS, "Hz,", DURACION, "s")