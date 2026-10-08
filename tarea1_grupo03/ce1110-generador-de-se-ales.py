#!/usr/bin/env python
# coding: utf-8

# # CE 1110 - Generador de señales para la defensa
# 
# Genera señales limpias y contaminadas para evaluar el **sistema distribuido de reducción adaptativa de ruido**. El notebook permite configurar casos conocidos o producir una prueba desconocida reproducible mediante una semilla. Exporta WAV de la señal limpia, el ruido y la mezcla, además de metadatos JSON y un ZIP.
# 
# > Seguridad: inicie la reproducción con volumen bajo. La salida se normaliza a un pico máximo configurable para evitar saturación digital, pero el nivel eléctrico final depende del equipo de reproducción.

# In[1]:


import json, shutil, zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, chirp, get_window
from IPython.display import Audio, display

plt.style.use('seaborn-v0_8-whitegrid')
print('Entorno listo')


# ## 1. Configuración
# Cambie solamente esta celda para preparar una prueba. Use `MODO = 'manual'` para fijar todos los parámetros o `MODO = 'defensa_aleatoria'` para sortear un caso reproducible.

# In[ ]:


MODO = 'manual'                 # 'manual' o 'defensa_aleatoria'
SEMILLA = 2026106         # guardar para reproducir exactamente la prueba
NOMBRE_PRUEBA = 'caso_3'
FS = 16_000                     # Hz
DURACION = 6.0                  # s
PICO_MAXIMO = 0.80              # amplitud digital final
FADE_MS = 20                    # entrada/salida suave

# Señal útil: 'tono', 'multitono','chirp', 'armonicos' o 'transitoria'
TIPO_SENAL = 'multitono'
FRECUENCIAS_UTILES = [440, 880, 1320]
AMPLITUDES_UTILES = [1.0, 0.55, 0.30]
FASES_UTILES_GRADOS = [0, 35, -50]

# Ruido: 'tonal', 'red_60hz', 'blanco', 'banda', 'impulsivo' o 'mixto'
TIPO_RUIDO = 'banda'
SNR_OBJETIVO_DB = 3.0
FRECUENCIA_INTERFERENCIA = 1750  #  Hz, para tonal o mixto
BANDA_RUIDO = (500, 1500)       # Hz, debe quedar bajo FS/2
NUM_IMPULSOS = 10

CARPETA_SALIDA = Path('/kaggle/working/tonos_defensa') if Path('/kaggle/working').exists() else Path('tonos_defensa')
CARPETA_SALIDA


# ## 2. Funciones de generación y medición

# In[3]:


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))

def snr_db(clean, observed):
    error = observed - clean
    return 10*np.log10((np.sum(clean**2)+1e-15)/(np.sum(error**2)+1e-15))

def fade(x, fs, ms=20):
    n = min(int(fs*ms/1000), len(x)//2)
    if n > 0:
        ramp = np.sin(np.linspace(0, np.pi/2, n))**2
        x = x.copy(); x[:n] *= ramp; x[-n:] *= ramp[::-1]
    return x

def normalizar_conjunto(*signals, peak=0.8):
    maximum = max(np.max(np.abs(x)) for x in signals)
    gain = peak/maximum if maximum > 0 else 1.0
    return [x*gain for x in signals], gain

def generar_limpia(kind, t, freqs, amps, phases_deg, rng):
    if kind == 'tono':
        return amps[0]*np.sin(2*np.pi*freqs[0]*t + np.deg2rad(phases_deg[0]))
    if kind == 'multitono':
        return sum(a*np.sin(2*np.pi*f*t + np.deg2rad(p)) for f,a,p in zip(freqs,amps,phases_deg))
    if kind == 'chirp':
        return chirp(t, f0=freqs[0], f1=freqs[-1], t1=t[-1], method='linear')
    if kind == 'armonicos':
        f0 = freqs[0]
        return sum((1/k)*np.sin(2*np.pi*k*f0*t + rng.uniform(-np.pi,np.pi)) for k in range(1,6))
    if kind == 'transitoria':
        env = np.exp(-5*(t % 1.0))*(np.sin(2*np.pi*2*t)>0)
        return env*np.sin(2*np.pi*freqs[0]*t)
    raise ValueError(f'Tipo de señal no reconocido: {kind}')

def ruido_banda(n, fs, band, rng):
    low, high = band
    if not (0 < low < high < fs/2):
        raise ValueError(f'BANDA_RUIDO debe cumplir 0 < low < high < {fs/2}')
    sos = butter(6, [low, high], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, rng.standard_normal(n))

def generar_ruido(kind, t, fs, f_int, band, impulses, rng):
    n = len(t)
    tonal = np.sin(2*np.pi*f_int*t + rng.uniform(-np.pi,np.pi))
    red = sum((1/k)*np.sin(2*np.pi*60*k*t + rng.uniform(-np.pi,np.pi)) for k in range(1,5))
    white = rng.standard_normal(n)
    band_noise = ruido_banda(n, fs, band, rng)
    impulse = np.zeros(n)
    locations = rng.choice(np.arange(int(.1*fs), n-int(.1*fs)), size=min(impulses,max(1,n//100)), replace=False)
    for loc in locations:
        width = max(2, int(rng.uniform(.0005,.004)*fs))
        stop = min(n, loc+width)
        impulse[loc:stop] += rng.choice([-1,1])*np.hanning(2*(stop-loc))[:stop-loc]
    options = {'tonal':tonal, 'red_60hz':red, 'blanco':white, 'banda':band_noise, 'impulsivo':impulse}
    if kind == 'mixto':
        return 0.55*tonal + 0.30*band_noise + 0.15*impulse
    if kind not in options: raise ValueError(f'Tipo de ruido no reconocido: {kind}')
    return options[kind]

def escalar_ruido_para_snr(clean, noise, target_db):
    target_noise_rms = rms(clean)/(10**(target_db/20))
    return noise*(target_noise_rms/(rms(noise)+1e-15))

def espectro(x, fs):
    w = get_window('hann', len(x), fftbins=True)
    X = np.fft.rfft(x*w)
    f = np.fft.rfftfreq(len(x), 1/fs)
    mag = 20*np.log10(np.maximum(np.abs(X)/(np.sum(w)/2), 1e-12))
    return f, mag


# ## 3. Sorteo opcional de la prueba desconocida
# El sorteo evita casos imposibles para el hardware: todas las frecuencias permanecen por debajo de Nyquist y la banda de ruido se construye dentro del margen disponible. La configuración final siempre se muestra y se guarda.

# In[4]:


rng = np.random.default_rng(SEMILLA)
if MODO == 'defensa_aleatoria':
    TIPO_SENAL = rng.choice(['tono','multitono','armonicos','transitoria']).item()
    TIPO_RUIDO = rng.choice(['tonal','red_60hz','blanco','banda','impulsivo','mixto']).item()
    base = int(rng.choice([220, 260, 330, 390, 440, 520, 660]))
    FRECUENCIAS_UTILES = [base, min(2*base, int(.28*FS)), min(3*base, int(.38*FS))]
    AMPLITUDES_UTILES = [1.0, float(rng.uniform(.35,.7)), float(rng.uniform(.15,.4))]
    FASES_UTILES_GRADOS = rng.integers(-180,181,3).tolist()
    FRECUENCIA_INTERFERENCIA = int(rng.uniform(.18*FS, .40*FS))
    low = int(rng.uniform(.20*FS,.30*FS)); high = int(min(low+rng.uniform(.07*FS,.14*FS),.46*FS))
    BANDA_RUIDO = (low, high)
    SNR_OBJETIVO_DB = float(rng.choice([-5,-2,0,3,6]))

print({'modo':MODO,'semilla':SEMILLA,'señal':TIPO_SENAL,'ruido':TIPO_RUIDO,
       'frecuencias_utiles_Hz':FRECUENCIAS_UTILES,'interferencia_Hz':FRECUENCIA_INTERFERENCIA,
       'banda_ruido_Hz':BANDA_RUIDO,'snr_objetivo_dB':SNR_OBJETIVO_DB})


# ## 4. Generación, comprobación y gráficas

# In[5]:


if FS < 2*max(FRECUENCIAS_UTILES + [FRECUENCIA_INTERFERENCIA, BANDA_RUIDO[1]]):
    raise ValueError('La configuración viola Nyquist. Aumente FS o reduzca las frecuencias.')

t = np.arange(int(FS*DURACION))/FS
clean = generar_limpia(TIPO_SENAL,t,FRECUENCIAS_UTILES,AMPLITUDES_UTILES,FASES_UTILES_GRADOS,rng)
clean = fade(clean,FS,FADE_MS)
noise_raw = generar_ruido(TIPO_RUIDO,t,FS,FRECUENCIA_INTERFERENCIA,BANDA_RUIDO,NUM_IMPULSOS,rng)
noise = escalar_ruido_para_snr(clean,noise_raw,SNR_OBJETIVO_DB)
noisy = clean + noise
[clean, noise, noisy], gain = normalizar_conjunto(clean,noise,noisy,peak=PICO_MAXIMO)
snr_real = snr_db(clean,noisy)
print(f'Muestras: {len(t):,} | SNR obtenida: {snr_real:.3f} dB | Ganancia común: {gain:.5f}')

view = min(len(t),int(.05*FS))
f1,m1 = espectro(clean,FS); f2,m2 = espectro(noisy,FS)
fig,ax = plt.subplots(2,1,figsize=(14,8))
ax[0].plot(t[:view]*1000,clean[:view],label='Limpia',lw=1.5)
ax[0].plot(t[:view]*1000,noisy[:view],label='Contaminada',alpha=.7,lw=1)
ax[0].set(xlabel='Tiempo (ms)',ylabel='Amplitud',title='Primeros 50 ms'); ax[0].legend()
ax[1].plot(f1,m1,label='Limpia'); ax[1].plot(f2,m2,label='Contaminada',alpha=.75)
ax[1].set_xlim(0,FS/2); ax[1].set_ylim(max(-120,np.max(m2)-100),np.max(m2)+5)
ax[1].set(xlabel='Frecuencia (Hz)',ylabel='Magnitud (dB)',title='Espectro con ventana Hann'); ax[1].legend()
plt.tight_layout(); plt.show()


# ## 5. Escucha
# La referencia limpia se incluye para evaluación docente. Para la defensa puede reproducirse únicamente la señal contaminada hacia el ADC del sistema.

# In[6]:


print('Referencia limpia')
display(Audio(clean,rate=FS,normalize=False))
print('Entrada contaminada para el sistema')
display(Audio(noisy,rate=FS,normalize=False))


# ## 6. Exportación
# Crea archivos PCM mono de 16 bits. Todos usan la misma ganancia, por lo que las relaciones de amplitud y la SNR se conservan. El JSON permite repetir y auditar la prueba.

# In[7]:


if CARPETA_SALIDA.exists(): shutil.rmtree(CARPETA_SALIDA)
CARPETA_SALIDA.mkdir(parents=True)
def pcm16(x): return np.int16(np.clip(x,-1,1)*32767)
wavfile.write(CARPETA_SALIDA/'01_referencia_limpia.wav',FS,pcm16(clean))
wavfile.write(CARPETA_SALIDA/'02_ruido.wav',FS,pcm16(noise))
wavfile.write(CARPETA_SALIDA/'03_entrada_contaminada.wav',FS,pcm16(noisy))
np.savetxt(CARPETA_SALIDA/'muestras_referencia.csv',np.column_stack([t,clean,noise,noisy]),
           delimiter=',',header='tiempo_s,limpia,ruido,contaminada',comments='')
metadata = {
 'nombre_prueba':NOMBRE_PRUEBA,'modo':MODO,'semilla':int(SEMILLA),'fs_Hz':int(FS),
 'duracion_s':float(DURACION),'tipo_senal':TIPO_SENAL,'frecuencias_utiles_Hz':[float(v) for v in FRECUENCIAS_UTILES],
 'amplitudes_utiles':[float(v) for v in AMPLITUDES_UTILES],'fases_grados':[float(v) for v in FASES_UTILES_GRADOS],
 'tipo_ruido':TIPO_RUIDO,'snr_objetivo_dB':float(SNR_OBJETIVO_DB),'snr_obtenida_dB':float(snr_real),
 'frecuencia_interferencia_Hz':float(FRECUENCIA_INTERFERENCIA),'banda_ruido_Hz':[float(v) for v in BANDA_RUIDO],
 'numero_impulsos':int(NUM_IMPULSOS),'pico_maximo':float(PICO_MAXIMO),'ganancia_comun':float(gain)
}
with open(CARPETA_SALIDA/'metadatos_prueba.json','w',encoding='utf-8') as f: json.dump(metadata,f,indent=2,ensure_ascii=False)
with open(CARPETA_SALIDA/'LEAME.txt','w',encoding='utf-8') as f:
    f.write('03_entrada_contaminada.wav: señal que se aplica al sistema.\n01_referencia_limpia.wav: referencia reservada para calcular métricas.\n02_ruido.wav: componente de ruido aislada.\n')
zip_path = CARPETA_SALIDA.parent/f'{NOMBRE_PRUEBA}.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
    for p in CARPETA_SALIDA.iterdir(): z.write(p,arcname=p.name)
print(f'Archivos exportados en: {CARPETA_SALIDA}')
print(f'ZIP para descargar: {zip_path}')
display(metadata)


# ## Uso sugerido durante la defensa
# 1. Defina una semilla distinta por grupo y ejecute todo el notebook.
# 2. Conserve `01_referencia_limpia.wav` y `metadatos_prueba.json` como información docente.
# 3. Reproduzca `03_entrada_contaminada.wav` hacia el ADC, comenzando con volumen bajo.
# 4. Solicite al grupo diagnosticar el ruido, ajustar su sistema y mostrar ambas salidas.
# 5. Capture las salidas y compárelas contra la referencia con MSE, SNR, energía, atenuación y latencia.
# 6. Si desea exactamente el mismo caso otra vez, reutilice la semilla y la configuración guardada.
