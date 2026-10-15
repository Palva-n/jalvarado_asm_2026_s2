import numpy as np
from DFT import FFT 
import matplotlib.pyplot as plt 
import random 


Nvals = [256, 512, 1024, 4096]
fs = 16000.0 

for N in Nvals: 
    memo = N * 8
    if (N * 8 > 1024): 
        memo = memo/1024 
    print("=============================")
    print(f"Muestras: {N}")
    print(f"df: {fs/N}")
    print(f"duracion: {N/fs}")
    print(f"memoria: {memo}") #4 bytes y x2 ya que el resultado es complejo 
    print("============================")

def signal(t, f_selected, phi): 
    return np.cos(2 * np.pi * f_selected * t + phi)
#sin uso de ventanas de Hann: 
t = np.linspace(0, Nvals[2]/fs, Nvals[2], endpoint = False)


casos = [1000, 1100] #uno que sea multiplo de fs/N y otro que no 

frecs = np.array([i * fs/len(t) for i in range(len(t))], dtype = float) 

for f_selected in casos: 
    x = signal(t, f_selected, 0) 
    X = FFT(x) 
    magX = [np.abs(i) for i in X]
    magX = np.array(magX)[:len(t)//2]
    magX = 20 * np.log10(np.maximum(magX, 1e-12))
    plt.figure() 
    plt.plot(frecs[:len(t)//2], magX, color = 'blue')
    plt.xlabel("frecuencia [Hz]")
    plt.ylabel("magnitud [dB]")
    plt.axhline(y=0, color="black")
    plt.xlim(800, 1300)
    plt.ylim(-130, 60)
    plt.grid(True)
    plt.savefig(f"../resultados/figuras/{f_selected}_mag_FFT.png")
    plt.close() 

#con uso de ventanas de hann: 

for f_selected in casos: 
    x = signal(t, f_selected, 0) 
    w = np.hanning(len(t))
    X = FFT(x * w) 
    magX = [np.abs(i) for i in X]
    magX = np.array(magX)[:len(t)//2]
    magX = 20 * np.log10(np.maximum(magX, 1e-12))
    plt.figure() 
    plt.plot(frecs[:len(t)//2], magX, color = 'blue')
    plt.xlabel("frecuencia [Hz]")
    plt.ylabel("magnitud [dB]")
    plt.axhline(y=0, color="black")
    plt.xlim(800, 1300)
    plt.ylim(-130, 60)
    plt.grid(True)
    plt.savefig(f"../resultados/figuras/{f_selected}_hann_mag_FFT.png")
    plt.close() 

#confirmar y destruir simetria: 

#confirmar: 
sig = signal(t, 1000, 0)
X = FFT(sig)
print("maximo error entre k y N - k para todo k = 0, 1, ... , N:")
mx = 0
for i in range(len(t)//2): 
    mx = max(np.abs(X[i] - np.conj(X[(len(t) - i)%len(t)])), mx)
print(mx) 


#destruir: toma un k arbitrario y cambialo 
print("destruccion de simetria: ")
X[64] = 1 + 1j
x_supposed = np.fft.ifft(X) 
print("maximo valor imaginario de la inversa: ")
print(np.max(np.abs(np.imag(x_supposed))))


#cambiar espejo y confirmar que la parte imaginaria cae: 
print("arreglo de simetria")
X[len(t) - 64] = 1 - 1j  #ya que el espejo tiene que ser conjugado 
x_supposed = np.fft.ifft(X) 
print("maximo valor imaginario de la inversa: ")
print(np.max(np.abs(np.imag(x_supposed))))


#magnitud y fase juntas, con desfase conocido:
phi = np.radians(35)   # desfase de prueba, igual al del notebook del profe

for f_selected in casos:
    x = signal(t, f_selected, phi)
    X = FFT(x)
    N = len(t)
    mag = np.abs(X)[:N//2]
    fase = np.degrees(np.angle(X))[:N//2]

    # la fase solo vale donde hay señal; en el resto es basura numerica
    fase[mag < 0.01 * np.max(mag)] = np.nan

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(frecs[:N//2], 20*np.log10(np.maximum(mag, 1e-12)), color='blue')
    ax[0].set_ylabel("magnitud [dB]")
    ax[0].set_ylim(-130, 60)
    ax[0].grid(True)
    ax[1].plot(frecs[:N//2], fase, 'o', color='red', markersize=4)
    ax[1].set_ylabel("fase [grados]")
    ax[1].set_xlabel("frecuencia [Hz]")
    ax[1].set_ylim(-180, 180)
    ax[1].grid(True)
    plt.xlim(800, 1300)
    plt.savefig(f"../resultados/figuras/{f_selected}_mag_fase.png")
    plt.close()

# chequeo: la fase en el bin 64 debe dar 35 grados
X = FFT(signal(t, 1000, phi))
print("fase en el bin 64 [grados]:", np.degrees(np.angle(X[64])))