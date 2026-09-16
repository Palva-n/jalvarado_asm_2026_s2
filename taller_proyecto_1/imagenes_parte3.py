import matplotlib.pyplot as plt
import os
from Signal_Generator import SignalGenerator

os.makedirs("imagenes", exist_ok=True)

gen_cos = SignalGenerator(sample_rate=2000, dur=0.5, f_sonido=100, A=1.0)

# --- Coseno puro (f0=f1), SIN restar leakage ---
x_cos = gen_cos.generate_chirp(f0=100, f1=100)
y_cos = gen_cos.eco(x_cos, 33, atenuacion=0.6)
corr_cos = gen_cos.correlacion_fft(x_cos, y_cos)

plt.figure()
plt.plot(corr_cos)
plt.title("Correlacion con senoidal pura (ambigua)")
plt.xlabel("lag")
plt.ylabel("correlacion")
plt.grid(True)
plt.savefig("imagenes/correlacion_senoidal_pura.png")
plt.close()

# --- Chirp, CON leakage restado ---
x_chirp = gen_cos.generate_chirp(f0=100, f1=500)
y_chirp = gen_cos.eco(x_chirp, 33, atenuacion=0.6)
corr_chirp = gen_cos.correlacion_fft(x_chirp, y_chirp)
corr_chirp_directa = gen_cos.correlacion_fft(x_chirp, x_chirp)
corr_chirp_sin_leakage = corr_chirp - corr_chirp_directa

plt.figure()
plt.plot(corr_chirp_sin_leakage)
plt.title("Correlacion con chirp, leakage removido (pico limpio)")
plt.xlabel("lag")
plt.ylabel("correlacion")
plt.grid(True)
plt.savefig("imagenes/correlacion_chirp_sin_leakage.png")
plt.close()

print("listo")