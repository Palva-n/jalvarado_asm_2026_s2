
from machine import Pin, ADC, Timer
import machine
import sys
import time
import gc
import math
import struct
from array import array
from ulab import numpy as np
from ulab import scipy as spy

#  Constantes fijas 
CPU_HZ = 125_000_000     # probar 250_000_000 si se necesita mas velocidad
FS_TIMER = 16000         # frecuencia que se le pide al timer
N = 1024                 # tamano de la FFT
HOP = N // 2             # muestras nuevas por bloque
EPS = 1e-12
SINCRONIA = 0xAA55
TIPO_ENTRADA = 0
TIPO_TEMPORAL = 1
TIPO_ESPECTRAL = 2
FORMATO_CABECERA = ">HBHHh"   # sincronia, tipo, n_bloque, frecuencia*8, nivel*10

#  Parametros ajustables desde la PC 
P = {
    "duracion": 1.0,          # s a grabar (maximo 2.0 por la memoria)
    "f_min_util": 300.0,      # Hz, banda util
    "f_max_util": 1500.0,
    "umbral_db": 20.0,        # nodo 1: prominencia minima para declarar interferencia
    "q_notch": 10.0,          # nodo 2
    "histeresis_hz": 0.5,
    "hp_activo": 1,           # nodo 2: 1 = pasaaltos IIR que quita el ruido grave
    "hp_factor": 0.5,         # el pasaaltos corta a f_min_util * hp_factor
    "fir_activo": 1,          # nodo 2: 1 = notch + FIR, 0 = solo notch
    "fir_taps": 63,
    "fir_margen_hz": 300.0,   # el FIR corta f_max_util + margen
    "suavizado": 0.9,         # nodo 3
    "agresividad": 1.5,
    "piso_ganancia": 0.1,
    "prominencia": 10.0,
    "ancho_notch": 3,
    "piso_notch": 0.001,
}

# Modo PWM de la fuente de la Pico W (baja el ruido del ADC), igual que en receptor_tonos.py
Pin("WL_GPIO1", Pin.OUT, value=1)

pin_adc = ADC(Pin(26))
led = Pin("LED", Pin.OUT)


# 
# CRC-16 (CCITT, polinomio 0x1021, valor inicial 0xFFFF): igual al de la simulacion
# 
def crear_tabla_crc():
    tabla = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        tabla.append(crc)
    return tabla


TABLA_CRC = crear_tabla_crc()


def crc16(datos):
    crc = 0xFFFF
    for b in datos:
        crc = ((crc << 8) & 0xFFFF) ^ TABLA_CRC[(crc >> 8) ^ b]
    return crc


def enviar_trama(tipo, n_bloque, frecuencia_hz, nivel_db, muestras):
    # muestras: ndarray float en [-1, 1) -> int16 little-endian
    s = np.clip(muestras * 32768, -32768, 32767)
    cuerpo = struct.pack(FORMATO_CABECERA, SINCRONIA, tipo, n_bloque % 65536,
                         int(round(frecuencia_hz * 8)),
                         int(max(-32768, min(32767, round(nivel_db * 10))))) \
        + np.array(s, dtype=np.int16).tobytes()
    sys.stdout.buffer.write(cuerpo + struct.pack(">H", crc16(cuerpo)))


#
# Captura del ADC con un timer. Es el mismo metodo de pruebas_microfono1.py
# (variables globales simples en el callback: mas rapido y con menos jitter).
# 
datos = array('H', (0 for _ in range(1)))
indice = 0
n_total = 0
terminado = False
t_inicio = 0
t_fin = 0


def registrar_muestra(timer):
    global indice, terminado, t_inicio, t_fin
    if indice < n_total:
        if indice == 0:
            t_inicio = time.ticks_us()
        datos[indice] = pin_adc.read_u16()
        indice += 1
        if indice == n_total:
            t_fin = time.ticks_us()
    else:
        timer.deinit()
        terminado = True


def grabar(n_muestras):
    global datos, indice, n_total, terminado
    datos = array('H', (0 for _ in range(n_muestras)))
    indice = 0
    n_total = n_muestras
    terminado = False
    tim = Timer(-1)
    tim.init(freq=FS_TIMER, mode=Timer.PERIODIC, callback=registrar_muestra)
    while not terminado:
        pass
    duracion = time.ticks_diff(t_fin, t_inicio) / 1e6
    fs_real = (n_muestras - 1) / duracion   # frecuencia de muestreo efectiva
    return datos, fs_real


# 
# FFT: esta version de ulab devuelve un arreglo complejo. Se libera memoria
# antes de cada FFT/IFFT porque necesitan bloques continuos de 8 KB.
# 
def fft(v):
    gc.collect()
    return np.fft.fft(v)


def ifft_real(X):
    gc.collect()
    return np.real(np.fft.ifft(X))


#
# Nodo 1: FFT del bloque y estimacion de la interferencia
#
class Nodo1:
    def __init__(self, fs):
        self.fs = fs
        self.anterior = np.zeros(HOP)
        self.ventana = np.array([0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)])

    def procesar(self, nuevas):
        bloque = np.concatenate((self.anterior, nuevas))
        self.anterior[:] = nuevas      # se copia en el mismo lugar (evita fragmentar la memoria)
        X = fft(bloque * self.ventana)
        re = np.real(X)
        im = np.imag(X)
        mag = np.sqrt(re * re + im * im)[:HOP + 1]   # solo frecuencias positivas
        del X, re, im

        # Se busca el pico mas grande por encima de la banda util
        k_min = int(math.ceil(P["f_max_util"] / (self.fs / N)))
        k = k_min + int(np.argmax(mag[k_min:HOP]))
        piso = float(np.median(mag)) + EPS
        nivel_db = 20 * math.log10((float(mag[k]) + EPS) / piso)
        if nivel_db < P["umbral_db"]:
            return 0.0, nivel_db

        # Interpolacion parabolica sobre el logaritmo de la magnitud (mejora la frecuencia)
        a = math.log(float(mag[k - 1]) + EPS)
        b = math.log(float(mag[k]) + EPS)
        c = math.log(float(mag[k + 1]) + EPS)
        den = a - 2 * b + c
        p = 0.0
        if den != 0:
            p = 0.5 * (a - c) / den
        return (k + p) * self.fs / N, nivel_db


# 
# Nodo 2: notch IIR adaptativo + pasabajos FIR (ecuacion de diferencias por bloques)
# 
def disenar_notch(f0, q, fs):
    w0 = 2 * math.pi * f0 / fs
    beta = math.tan(w0 / (2 * q))
    g = 1 / (1 + beta)
    b = [g, -2 * g * math.cos(w0), g]
    a = [1.0, -2 * g * math.cos(w0), 2 * g - 1]
    return b, a


def disenar_pasaaltos(fc, fs):
    # Biquad pasaaltos Butterworth de 2.o orden (transformada bilineal)
    k = math.tan(math.pi * fc / fs)
    q = 1 / math.sqrt(2)
    norma = 1 / (1 + k / q + k * k)
    b = [norma, -2 * norma, norma]
    a = [1.0, 2 * (k * k - 1) * norma, (1 - k / q + k * k) * norma]
    return b, a


def disenar_fir_pasabajos(fc, taps, fs):
    # Ventana de Hamming sobre un sinc ideal, normalizado a ganancia 1 en continua
    M = taps - 1
    h = []
    for n in range(taps):
        x = n - M / 2
        if x == 0:
            ideal = 2 * fc / fs
        else:
            ideal = math.sin(2 * math.pi * fc * x / fs) / (math.pi * x)
        ventana = 0.54 - 0.46 * math.cos(2 * math.pi * n / M)
        h.append(ideal * ventana)
    suma = sum(h)
    return np.array([v / suma for v in h])


class Nodo2:
    # Cadena: notch IIR adaptativo -> pasaaltos IIR -> pasabajos FIR
    # Las dos primeras etapas IIR van en un solo sosfilt de 2 secciones.
    def __init__(self, fs):
        self.fs = fs
        self.f_actual = 0.0
        self.fila_notch = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]    # identidad hasta que haya interferencia
        self.fila_hp = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        if P["hp_activo"] != 0:
            b, a = disenar_pasaaltos(P["f_min_util"] * P["hp_factor"], fs)
            self.fila_hp = [b[0], b[1], b[2], 1.0, a[1], a[2]]
        self.sos = np.array([self.fila_notch, self.fila_hp])
        self.zi = np.zeros((2, 2))
        self.usa_fir = P["fir_activo"] != 0
        if self.usa_fir:
            taps = int(P["fir_taps"])
            self.h = disenar_fir_pasabajos(P["f_max_util"] + P["fir_margen_hz"], taps, fs)
            self.historial = np.zeros(taps - 1)

    def actualizar_notch(self, f):
        if f == 0:
            if self.f_actual != 0:
                self.fila_notch = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                self.sos = np.array([self.fila_notch, self.fila_hp])
                self.f_actual = 0.0
            return
        if abs(f - self.f_actual) > P["histeresis_hz"]:
            b, a = disenar_notch(f, P["q_notch"], self.fs)
            self.fila_notch = [b[0], b[1], b[2], 1.0, a[1], a[2]]
            self.sos = np.array([self.fila_notch, self.fila_hp])
            self.f_actual = f

    def procesar(self, nuevas, f_estimada):
        self.actualizar_notch(f_estimada)
        y, self.zi = spy.signal.sosfilt(self.sos, nuevas, zi=self.zi)
        if self.usa_fir:
            # y[n] = suma h[k] x[n-k]: se concatena el final del bloque anterior
            ext = np.concatenate((self.historial, y))
            completo = np.convolve(ext, self.h)
            taps = len(self.h)
            self.historial[:] = ext[HOP:]   # en el mismo lugar
            y = completo[taps - 1:taps - 1 + HOP]
        return y


# 
# Nodo 3: filtrado espectral por bloques (ventana, FFT, ganancia, IFFT, overlap-add)
# 
class Nodo3:
    def __init__(self, fs):
        self.fs = fs
        self.anterior = np.zeros(HOP)
        self.cola = np.zeros(HOP)
        self.ruido = None
        # Ventana raiz de Hann: aplicada al analisis y a la sintesis suma 1 con 50 % de solape
        self.ventana = np.array([math.sqrt(0.5 * (1 - math.cos(2 * math.pi * n / N))) for n in range(N)])
        self.df = fs / N

    def estimar_ruido(self, potencia):
        # Mediana en frecuencia (31 bins) calculada cada 4 bins e interpolada: baja el costo.
        paso = 4
        centros = list(range(0, HOP + 1, paso))
        if centros[-1] != HOP:
            centros.append(HOP)
        valores = []
        for c in centros:
            ini = max(0, c - 15)
            fin = min(HOP + 1, c + 16)
            valores.append(float(np.median(potencia[ini:fin])) / math.log(2))
        actual = np.interp(np.arange(HOP + 1), np.array(centros), np.array(valores))
        if self.ruido is None:
            self.ruido = actual
        else:
            self.ruido[:] = P["suavizado"] * self.ruido + (1 - P["suavizado"]) * actual
        return self.ruido

    def ganancia(self, mag, potencia, ruido):
        # Wiener por bin
        snr = np.maximum(potencia / (P["agresividad"] * ruido + EPS) - 1, 0)
        g = np.maximum(snr / (snr + 1), P["piso_ganancia"])

        # Notch espectral: picos fuera de la banda util se anulan (+- ancho_notch bins)
        umbral = P["prominencia"] * math.sqrt(float(np.median(ruido)))
        k_util_min = int(P["f_min_util"] / self.df)
        k_util_max = int(P["f_max_util"] / self.df)
        m = mag.tolist()
        piso_notch = P["piso_notch"]
        ancho = int(P["ancho_notch"])
        g_notch = [1.0] * (HOP + 1)
        for k in range(1, HOP):
            if k_util_min <= k <= k_util_max:
                continue
            if m[k] > umbral and m[k] > m[k - 1] and m[k] >= m[k + 1]:
                for j in range(max(0, k - ancho), min(HOP + 1, k + ancho + 1)):
                    g_notch[j] = piso_notch
        return g * np.array(g_notch)

    def procesar(self, nuevas):
        X = fft(np.concatenate((self.anterior, nuevas)) * self.ventana)
        self.anterior[:] = nuevas        # en el mismo lugar
        re = np.real(X)
        im = np.imag(X)
        potencia = (re * re + im * im)[:HOP + 1]
        del re, im
        mag = np.sqrt(potencia)
        ruido = self.estimar_ruido(potencia)
        g = self.ganancia(mag, potencia, ruido)
        # Ganancia real y simetrica: conserva la simetria conjugada (senal real)
        g_completa = np.concatenate((g, np.flip(g[1:HOP])))
        y = ifft_real(X * g_completa) * self.ventana
        terminada = self.cola + y[:HOP]
        self.cola[:] = y[HOP:]           # en el mismo lugar
        return terminada


# 
# Comando REC: grabar y procesar
# 
def comando_rec():
    n_muestras = int(P["duracion"] * FS_TIMER)
    n_muestras = (n_muestras // HOP) * HOP
    gc.collect()
    led.value(1)
    datos, fs_real = grabar(n_muestras)
    led.value(0)

    # Continua (offset del MAX4466): media de toda la grabacion, calculada por trozos
    suma = 0.0
    for m in range(n_muestras // HOP):
        suma += float(np.sum(np.array(datos[m * HOP:(m + 1) * HOP], dtype=np.float)))
    media = suma / n_muestras
    gc.collect()

    n_bloques = n_muestras // HOP
    longitud_trama = struct.calcsize(FORMATO_CABECERA) + HOP * 2 + 2
    sys.stdout.write("BEGIN {} {} {} {:.3f} {}\n".format(n_bloques, HOP, N, fs_real, longitud_trama))

    nodo1, nodo2, nodo3 = Nodo1(fs_real), Nodo2(fs_real), Nodo3(fs_real)
    t1 = t2 = t3 = 0
    for m in range(n_bloques):
        # Bloque a float, sin continua y normalizado a +-1
        nuevas = (np.array(datos[m * HOP:(m + 1) * HOP], dtype=np.float) - media) / 32768.0

        a = time.ticks_us()
        f_est, nivel = nodo1.procesar(nuevas)
        b = time.ticks_us()
        y2 = nodo2.procesar(nuevas, f_est)
        c = time.ticks_us()
        y3 = nodo3.procesar(nuevas)
        d = time.ticks_us()
        t1 += time.ticks_diff(b, a)
        t2 += time.ticks_diff(c, b)
        t3 += time.ticks_diff(d, c)

        enviar_trama(TIPO_ENTRADA, m, f_est, nivel, nuevas)
        enviar_trama(TIPO_TEMPORAL, m, nodo2.f_actual, 0.0, y2)
        enviar_trama(TIPO_ESPECTRAL, m, 0.0, 0.0, y3)
        gc.collect()

    # Latencia por bloque de cada nodo (microsegundos) y memoria libre
    sys.stdout.write("FIN {} {} {} {}\n".format(t1 // n_bloques, t2 // n_bloques,
                                                t3 // n_bloques, gc.mem_free()))


def comando_set(partes):
    clave = partes[1]
    if clave not in P:
        sys.stdout.write("ERR clave desconocida: {}\n".format(clave))
        return
    P[clave] = float(partes[2]) if "." in partes[2] or "e" in partes[2] else int(partes[2])
    sys.stdout.write("OK {} = {}\n".format(clave, P[clave]))


# 
# Bucle principal: espera comandos de la PC
# 
if __name__ == "__main__":
    if CPU_HZ != 125_000_000:
        machine.freq(CPU_HZ)   # solo si se pide algo distinto al valor por defecto
    while True:
        linea = sys.stdin.readline().strip()
        if not linea:
            continue
        partes = linea.split()
        if partes[0] == "PING":
            sys.stdout.write("OK PICO {}\n".format(machine.freq()))
        elif partes[0] == "SET" and len(partes) == 3:
            comando_set(partes)
        elif partes[0] == "GET":
            sys.stdout.write("PARAMS " + " ".join("{}={}".format(k, P[k]) for k in P) + "\n")
        elif partes[0] == "REC":
            try:
                comando_rec()
            except Exception as e:
                # El error se manda como texto, empieza con ERR_PICO
                sys.stdout.write("\nERR_PICO mem libre {}\n".format(gc.mem_free()))
                sys.print_exception(e, sys.stdout)
                sys.stdout.write("\nFIN_ERR\n")
        else:
            sys.stdout.write("ERR comando desconocido\n")
