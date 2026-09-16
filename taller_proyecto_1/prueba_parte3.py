import numpy as np
import time
from Signal_Generator import SignalGenerator

sample_rate = 2000
dur = 0.5
gen = SignalGenerator(sample_rate=sample_rate, dur=dur, f_sonido=100, A=1.0)

x = gen.generate_chirp(f0=100, f1=500)

retardo_real = 33
y = gen.eco(x, retardo_real, atenuacion=0.6)

N = len(x)
max_lag = N

# --- Correlacion directa ---
t0 = time.time()
corr_dir = gen.correlacion(x, y)
t_directa = time.time() - t0

corr_directa_pura = gen.correlacion(x, x)
corr_dir_sin_leak = corr_dir - corr_directa_pura
retardo_dir = np.argmax(corr_dir_sin_leak) - max_lag

# --- Correlacion via FFT ---
t0 = time.time()
corr_fft = gen.correlacion_fft(x, y)
t_fft = time.time() - t0

corr_fft_pura = gen.correlacion_fft(x, x)
corr_fft_sin_leak = corr_fft - corr_fft_pura
retardo_fft = np.argmax(corr_fft_sin_leak) - max_lag

print(f"Directa: retardo={retardo_dir}, tiempo={t_directa:.4f}s")
print(f"FFT:     retardo={retardo_fft}, tiempo={t_fft:.6f}s")
print(f"Esperado: {retardo_real}")
print(f"Speedup: {t_directa/t_fft:.1f}x")
print(f"\nMax diferencia entre metodos: {np.max(np.abs(corr_dir - corr_fft))}")