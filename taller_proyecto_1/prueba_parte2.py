import numpy as np
import matplotlib.pyplot as plt
from Signal_Analyzer import SignalAnalyzer
from Signal_Generator import SignalGenerator 

def known_signal(t):
    return float(np.cos(2*np.pi*300*(t-1)) + 9*np.sin(2*np.pi*400*(t+2)) - np.cos(2*np.pi*500*t))

fs = 2000
resultados = []

# --- Parte b: comparacion de tiempos DFT vs FFT ---
for i in range(6, 13):
    N = 2**i
    dur = N/fs
    xvals = np.linspace(0, dur, N)
    xs = np.array([known_signal(x) for x in xvals])

    analyzer = SignalAnalyzer(xs, fs)
    analyzer.DFT()
    analyzer.FFT()

    name = f"{i}th_power"
    analyzer.ImageGenerateDFT(name)
    analyzer.ImageGenerateFFT(name)

    t_dft, t_fft = analyzer.compare_times()
    resultados.append((N, t_dft, t_fft))
    print(f"N={N}: DFT={t_dft:.4f}s, FFT={t_fft:.6f}s")

# Grafica de tiempos
Ns = [r[0] for r in resultados]
t_dfts = [r[1] for r in resultados]
t_ffts = [r[2] for r in resultados]

plt.figure()
plt.plot(Ns, t_dfts, marker='o', label='DFT')
plt.plot(Ns, t_ffts, marker='o', label='FFT')
plt.xlabel('N')
plt.ylabel('tiempo (s)')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.savefig('imagenes/comparacion_tiempos.png')
plt.close()

# --- Parte c y d: repertorio de senales, incluyendo leakage ---
N_fijo = 2048
t_fijo = np.arange(N_fijo) / fs

señales_prueba = [
    ("senoidal_pura", lambda t: np.cos(2*np.pi*300*t)),
    ("dos_armonicos", lambda t: np.cos(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*600*t)),
    ("con_ruido", lambda t: np.cos(2*np.pi*300*t) + 0.3*np.random.randn(len(t))),
    ("leakage", lambda t: np.cos(2*np.pi*137*t)),
    ("sin_leakage", lambda t: np.cos(2*np.pi*125*t)),
]

for nombre, f_signal in señales_prueba:
    x = f_signal(t_fijo)
    analyzer = SignalAnalyzer(x, fs)
    analyzer.DFT()
    analyzer.FFT()
    analyzer.ImageGenerateDFT(f"sig_{nombre}")
    analyzer.ImageGenerateFFT(f"sig_{nombre}")

print("Listo, revisa la carpeta imagenes/")


#parte 3: Aniadir eco y ruido a una senial. 

gen = SignalGenerator(sample_rate=2000, dur=0.5, f_sonido=100, A=1.0)
x = gen.generate_chirp(f0=100, f1=500)   

retardo_real = 33  
x_con_eco = gen.eco(x, retardo_real, atenuacion=0.6)

corr = gen.correlacion(x, x_con_eco)
corr_directa = gen.correlacion(x, x)
corr_sin_leakage = corr - corr_directa

retardo_detectado = np.argmax(corr_sin_leakage) - len(x)
print(f"retardo detectado: {retardo_detectado}, esperado: {retardo_real}")