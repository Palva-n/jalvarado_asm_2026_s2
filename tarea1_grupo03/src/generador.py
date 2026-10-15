
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, chirp

FADE_MS = 20 


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def snr_db(clean, observed):
    error = observed - clean
    return 10 * np.log10((np.sum(clean**2) + 1e-15) / (np.sum(error**2) + 1e-15))


def fade(x, fs, ms=20):
    n = min(int(fs * ms / 1000), len(x) // 2)
    if n > 0:
        ramp = np.sin(np.linspace(0, np.pi / 2, n))**2
        x = x.copy(); x[:n] *= ramp; x[-n:] *= ramp[::-1]
    return x


def normalizar_conjunto(*signals, peak=0.8):
    maximum = max(np.max(np.abs(x)) for x in signals)
    gain = peak / maximum if maximum > 0 else 1.0
    return [x * gain for x in signals], gain


def generar_limpia(kind, t, freqs, amps, phases_deg, rng):
    if kind == 'tono':
        return amps[0] * np.sin(2*np.pi*freqs[0]*t + np.deg2rad(phases_deg[0]))
    if kind == 'multitono':
        return sum(a*np.sin(2*np.pi*f*t + np.deg2rad(p)) for f, a, p in zip(freqs, amps, phases_deg))
    if kind == 'chirp':
        return chirp(t, f0=freqs[0], f1=freqs[-1], t1=t[-1], method='linear')
    if kind == 'armonicos':
        f0 = freqs[0]
        return sum((1/k)*np.sin(2*np.pi*k*f0*t + rng.uniform(-np.pi, np.pi)) for k in range(1, 6))
    if kind == 'transitoria':
        env = np.exp(-5*(t % 1.0))*(np.sin(2*np.pi*2*t) > 0)
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
    tonal = np.sin(2*np.pi*f_int*t + rng.uniform(-np.pi, np.pi))
    red = sum((1/k)*np.sin(2*np.pi*60*k*t + rng.uniform(-np.pi, np.pi)) for k in range(1, 5))
    white = rng.standard_normal(n)
    band_noise = ruido_banda(n, fs, band, rng)
    impulse = np.zeros(n)
    locations = rng.choice(np.arange(int(.1*fs), n-int(.1*fs)), size=min(impulses, max(1, n//100)), replace=False)
    for loc in locations:
        width = max(2, int(rng.uniform(.0005, .004)*fs))
        stop = min(n, loc+width)
        impulse[loc:stop] += rng.choice([-1, 1])*np.hanning(2*(stop-loc))[:stop-loc]
    options = {'tonal': tonal, 'red_60hz': red, 'blanco': white, 'banda': band_noise, 'impulsivo': impulse}
    if kind == 'mixto':
        return 0.55*tonal + 0.30*band_noise + 0.15*impulse
    if kind not in options:
        raise ValueError(f'Tipo de ruido no reconocido: {kind}')
    return options[kind]


def escalar_ruido_para_snr(clean, noise, target_db):
    target_noise_rms = rms(clean)/(10**(target_db/20))
    return noise*(target_noise_rms/(rms(noise)+1e-15))


def pcm16(x):
    return np.int16(np.clip(x, -1, 1)*32767)


def cargar_configuraciones(ruta):
    with open(ruta, encoding='utf-8') as f:
        return json.load(f)['casos']


def generar_caso(cfg):
    fs = cfg['fs_Hz']
    f_max = max(cfg['frecuencias_utiles_Hz'] + [cfg['frecuencia_interferencia_Hz'], cfg['banda_ruido_Hz'][1]])
    if fs < 2*f_max:
        raise ValueError(f"{cfg['nombre_prueba']}: la configuración viola Nyquist")

    rng = np.random.default_rng(cfg['semilla'])
    t = np.arange(int(fs*cfg['duracion_s']))/fs
    clean = generar_limpia(cfg['tipo_senal'], t, cfg['frecuencias_utiles_Hz'],
                           cfg['amplitudes_utiles'], cfg['fases_grados'], rng)
    clean = fade(clean, fs, FADE_MS)
    noise_raw = generar_ruido(cfg['tipo_ruido'], t, fs, cfg['frecuencia_interferencia_Hz'],
                              tuple(cfg['banda_ruido_Hz']), cfg['numero_impulsos'], rng)
    noise = escalar_ruido_para_snr(clean, noise_raw, cfg['snr_objetivo_dB'])
    noisy = clean + noise
    [clean, noise, noisy], gain = normalizar_conjunto(clean, noise, noisy, peak=cfg['pico_maximo'])
    return t, clean, noise, noisy, gain, snr_db(clean, noisy)


def exportar_caso(cfg, carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    fs = cfg['fs_Hz']
    t, clean, noise, noisy, gain, snr_real = generar_caso(cfg)

    wavfile.write(carpeta/'01_referencia_limpia.wav', fs, pcm16(clean))
    wavfile.write(carpeta/'02_ruido.wav', fs, pcm16(noise))
    wavfile.write(carpeta/'03_entrada_contaminada.wav', fs, pcm16(noisy))
    np.savetxt(carpeta/'muestras_referencia.csv', np.column_stack([t, clean, noise, noisy]),
               delimiter=',', header='tiempo_s,limpia,ruido,contaminada', comments='')

    metadata = {
        'nombre_prueba': cfg['nombre_prueba'], 'modo': cfg['modo'], 'semilla': int(cfg['semilla']),
        'fs_Hz': int(fs), 'duracion_s': float(cfg['duracion_s']), 'tipo_senal': cfg['tipo_senal'],
        'frecuencias_utiles_Hz': [float(v) for v in cfg['frecuencias_utiles_Hz']],
        'amplitudes_utiles': [float(v) for v in cfg['amplitudes_utiles']],
        'fases_grados': [float(v) for v in cfg['fases_grados']],
        'tipo_ruido': cfg['tipo_ruido'], 'snr_objetivo_dB': float(cfg['snr_objetivo_dB']),
        'snr_obtenida_dB': float(snr_real),
        'frecuencia_interferencia_Hz': float(cfg['frecuencia_interferencia_Hz']),
        'banda_ruido_Hz': [float(v) for v in cfg['banda_ruido_Hz']],
        'numero_impulsos': int(cfg['numero_impulsos']), 'pico_maximo': float(cfg['pico_maximo']),
        'ganancia_comun': float(gain),
    }
    with open(carpeta/'metadatos_prueba.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(carpeta/'LEAME.txt', 'w', encoding='utf-8') as f:
        f.write('03_entrada_contaminada.wav: señal que se aplica al sistema.\n'
                '01_referencia_limpia.wav: referencia reservada para calcular métricas.\n'
                '02_ruido.wav: componente de ruido aislada.\n')
    return metadata
