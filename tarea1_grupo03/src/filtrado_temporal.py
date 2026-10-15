import numpy as np 
from scipy.io import wavfile
from scipy import signal 
import matplotlib.pyplot as plt 



#se escoge la senal del caso 1 y un filtro notch. 

CASOS = ["caso_1", "caso_2", "caso_3"]
RUTA = "../pruebas"
INICIO = 16000 
N = 32768


def cargar_wav(ruta):
    fs, x = wavfile.read(ruta)
    return fs, x.astype(np.float64) / 32768.0   # int16 -> [-1, 1]


def snr_db(limpia, contaminada):
    error = contaminada - limpia
    return 10 * np.log10(np.sum(limpia**2) / np.sum(error**2))

def mse(limpia, salida): 
    return np.mean((limpia - salida)**2)

def espectro(fs, signal): 
    w = np.hanning(len(signal))
    X = np.fft.fft(signal * w)[:len(signal)//2]
    f = np.arange(len(signal)//2) * fs / len(signal)
    return f, X

fs , x_clean = cargar_wav(f"{RUTA}/caso_1/01_referencia_limpia.wav")
_, x_d = cargar_wav(f"{RUTA}/caso_1/03_entrada_contaminada.wav")


def IIR_filter(f0, Q, fs): 
    b,a  = signal.iirnotch(f0, Q, fs)
    return b,a


b_, a_  = IIR_filter(1750, 10, fs) 

print(f"coeficientes a: {a_[0]}, {a_[1]}, {a_[2]}")
print(f"coeficientes b: {b_[0]}, {b_[1]}, {b_[2]}")


def H(z): 
    return ((b_[0] * (z**2)  + b_[1] * (z) + b_[2] )/(a_[0] * (z ** 2) + a_[1] * z + a_[2]))

polos = np.roots(a_) 
zeroes = np.roots(b_) 
z0 = 0 + 0j 
r = 1 
theta = np.linspace(0, 2 * np.pi, 200)

z = z0 + r * np.exp(1j * theta)
plt.figure() 
plt.plot(z.real, z.imag, label = f"|z - {z0}| = {r}", color = 'blue')
plt.plot(polos.real, polos.imag, 'o', label = "polos", color = 'red')
plt.plot(zeroes.real, zeroes.imag, 'o', label = "ceros", color = 'purple')
plt.axhline(y = 0, color = 'black')
plt.axvline(x = 0, color = 'black')
plt.axis('equal')
plt.grid(True) 
plt.title("puntos criticos del filtro")
plt.legend() 
plt.savefig("../resultados/figuras/filtro_puntos.png")

print("|polos| =", np.abs(polos))
print("estable:", np.all(np.abs(polos) < 1))



#respuesta en frecuencia evaluando H(z) sobre el circulo unitario
f0 = 1750
f = np.linspace(0, fs/2, 8000, endpoint = False)   # frecuencias a evaluar [Hz]
omega = 2 * np.pi * f / fs                         # frecuencia digital [rad/muestra]
z = np.exp(1j * omega)                             # puntos del circulo unitario
Hf = H(z)                                          # numero complejo por cada frecuencia

mag = 20 * np.log10(np.maximum(np.abs(Hf), 1e-12)) # magnitud en dB
fase = np.unwrap(np.angle(Hf))                     # fase en radianes
retardo = -np.gradient(fase, omega)                # retardo de grupo en muestras
retardo_ms = retardo / fs * 1000
retardo_ms[np.abs(f - f0) < 2] = np.nan            # en f0 el cero hace que no este definido

fig, ax = plt.subplots(3, 1, figsize = (8, 9), sharex = True)
ax[0].plot(f, mag, color = 'blue')
ax[0].set_ylabel("magnitud [dB]")
ax[0].set_ylim(-60, 5)
ax[1].plot(f, np.degrees(fase), color = 'red')
ax[1].set_ylabel("fase [grados]")
ax[2].plot(f, retardo_ms, color = 'green')
ax[2].set_ylabel("retardo de grupo [ms]")
ax[2].set_xlabel("frecuencia [Hz]")
for axi in ax: 
    axi.grid(True)
plt.savefig("../resultados/figuras/filtro_respuesta.png")
plt.close()


def ecuacion_diferencias(b, a, x): 
    y = np.zeros(len(x))
    for n in range(len(x)): 
        y[n] = b[0] * x[n]
        if n >= 1: 
            y[n] += b[1] * x[n-1] - a[1] * y[n-1]
        if n >= 2: 
            y[n] += b[2] * x[n-2] - a[2] * y[n-2]
    return y / a[0]

y_propia = ecuacion_diferencias(b_, a_, x_d)
y_lib = signal.lfilter(b_, a_, x_d)
print("error maximo propia vs lfilter:", np.max(np.abs(y_propia - y_lib)))
print("SNR entrada:", snr_db(x_clean[INICIO:], x_d[INICIO:]))
print("SNR salida :", snr_db(x_clean[INICIO:], y_propia[INICIO:]))


#------------------ (g) aplicar el filtro a los tres casos ------------------
print("caso      SNR_in   SNR_out   dSNR     MSE_in       MSE_out")
for caso in CASOS: 
    _, limpia = cargar_wav(f"{RUTA}/{caso}/01_referencia_limpia.wav")
    _, sucia = cargar_wav(f"{RUTA}/{caso}/03_entrada_contaminada.wav")
    filtrada = ecuacion_diferencias(b_, a_, sucia)

    # se descarta el primer segundo para no medir el transitorio inicial
    snr_in = snr_db(limpia[INICIO:], sucia[INICIO:])
    snr_out = snr_db(limpia[INICIO:], filtrada[INICIO:])
    mse_in = mse(limpia[INICIO:], sucia[INICIO:])
    mse_out = mse(limpia[INICIO:], filtrada[INICIO:])
    print(f"{caso}   {snr_in:6.2f}   {snr_out:6.2f}   {snr_out - snr_in:6.2f}   {mse_in:.2e}   {mse_out:.2e}")

    # espectro antes y despues
    f, X_sucia = espectro(fs, sucia[INICIO: INICIO + N])
    _, X_filtrada = espectro(fs, filtrada[INICIO: INICIO + N])
    plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
    ax1.plot(f, np.abs(X_sucia), color = 'red', label = "Sucia")
    ax2.plot(f, np.abs(X_filtrada), color = 'green', label = "Filtrada")
    for axi in (ax1, ax2): 
        axi.set_xlabel("frecuencia [Hz]")
        axi.set_ylabel("magnitud")
        axi.grid(True)
        axi.legend()
    plt.suptitle(caso + ": espectro antes y despues del notch")
    plt.savefig(f"../resultados/figuras/{caso}_antes_despues.png")
    plt.close()

#------------------ (h) cuantizacion de los coeficientes ------------------
def cuantizar(coef, bits_frac): 
    # punto fijo: se redondea al multiplo mas cercano de 2^(-bits_frac)
    paso = 2 ** (-bits_frac)
    return np.round(coef / paso) * paso

def ganancia_dB(b, a, f_hz): 
    # H(z) evaluada en z = e^{j w}, con w = 2 pi f / fs
    z = np.exp(1j * 2 * np.pi * f_hz / fs)
    H_z = (b[0] * z**2 + b[1] * z + b[2]) / (a[0] * z**2 + a[1] * z + a[2])
    return 20 * np.log10(np.maximum(np.abs(H_z), 1e-12))

_, limpia1 = cargar_wav(f"{RUTA}/caso_1/01_referencia_limpia.wav")
_, sucia1 = cargar_wav(f"{RUTA}/caso_1/03_entrada_contaminada.wav")

print()
print("formato    max|p|    estable   gan.1750Hz   gan.1320Hz   SNR_out(caso 1)")
formatos = [("float64", None), ("float32", "f32"), ("Q2.14", 14), ("Q2.12", 12), ("Q2.10", 10), ("Q2.8", 8)]
for nombre, bits in formatos: 
    if bits is None: 
        bq, aq = b_, a_
    elif bits == "f32": 
        bq, aq = b_.astype(np.float32).astype(np.float64), a_.astype(np.float32).astype(np.float64)
    else: 
        bq, aq = cuantizar(b_, bits), cuantizar(a_, bits)
    polos_q = np.roots(aq)
    salida_q = ecuacion_diferencias(bq, aq, sucia1)
    print(f"{nombre:8s}   {np.max(np.abs(polos_q)):.5f}   {str(np.all(np.abs(polos_q) < 1)):7s}   {ganancia_dB(bq, aq, 1750):8.1f} dB   {ganancia_dB(bq, aq, 1320):8.3f} dB   {snr_db(limpia1[INICIO:], salida_q[INICIO:]):6.2f}")


