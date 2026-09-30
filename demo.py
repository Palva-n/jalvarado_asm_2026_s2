import time
import numpy as np
import sounddevice as sd
import serial

PORT = "COM3"          # el puerto de tu Pico
FS = 16000
F0, F1 = 1500, 6000
DUR = 0.010
NX = 512
PRE = 200
M = 40
REC_S = 0.6
NFR_BASE = 30          # cuadros del baseline (calibración)
NFR_MED = 10           # cuadros por medición
SCALE = 20000
UMBRAL = 4.0           # "x del ruido" mínimo para declarar un eco
C = 343.0
DEV_IN = None
DEV_OUT = None
ZA, ZB = 120, 420

din, dout = sd.default.device
sd.default.device = (DEV_IN if DEV_IN is not None else din,
                     DEV_OUT if DEV_OUT is not None else dout)

NT = int(DUR * FS)
i = np.arange(NT); t = i / FS
w = 0.5 - 0.5 * np.cos(2 * np.pi * i / (NT - 1))
chirp = np.sin(2 * np.pi * (F0 * t + (F1 - F0) * t**2 / (2 * DUR))) * w
tx = np.zeros(int(REC_S * FS), dtype=np.float32)
tx[:NT] = 0.8 * chirp

def grabar():
    rec = sd.playrec(tx, samplerate=FS, channels=1, blocking=True)
    x = rec[:, 0].astype(float)
    x -= np.median(x)
    x = np.concatenate([np.zeros(PRE + M), x])
    ruido = np.median(np.abs(x)) / 0.6745
    thr = max(8 * ruido, 0.01)
    idx = int(np.argmax(np.abs(x) > thr))
    if abs(x[idx]) <= thr:
        return None, "sin chirp"
    ini = idx - PRE - M
    if ini + NX + 2 * M > len(x):
        return None, "chirp demasiado tarde"
    return x[ini:ini + NX + 2 * M], "ok"

def mejor_lag(wide, ref):
    best = None; bl = 0
    for lag in range(-M, M + 1):
        seg = wide[M + lag + ZA: M + lag + ZB]
        s = float(np.dot(seg, ref[ZA:ZB]))
        if best is None or s > best:
            best = s; bl = lag
    return bl

def apilar(n):
    ref = None; acc = None; k = 0; intentos = 0; motivos = {}
    while k < n and intentos < n * 3:
        intentos += 1
        wd, mot = grabar()
        time.sleep(0.05)
        if wd is None:
            motivos[mot] = motivos.get(mot, 0) + 1
            continue
        if ref is None:
            ref = wd[M:M + NX].copy(); acc = ref.copy(); k = 1
            continue
        lag = mejor_lag(wd, ref)
        if abs(lag) >= M - 1:
            motivos["desalineado"] = motivos.get("desalineado", 0) + 1
            continue
        acc += wd[M + lag: M + lag + NX]
        k += 1
    if k < max(3, n // 2):
        print("  cuadros válidos: %d de %d intentos. Motivos de descarte: %s" % (k, intentos, motivos))
        return None
    return acc / k

def alinear(ref, x):
    wide = np.concatenate([np.zeros(M), x, np.zeros(M)])
    lag = mejor_lag(wide, ref)
    return wide[M + lag: M + lag + NX], lag

def enviar(v):
    q = np.clip(np.round(v * SCALE), -32000, 32000).astype(int)
    datos = ("M," + ",".join(str(k) for k in q) + "\n").encode()
    for j in range(0, len(datos), 64):
        ser.write(datos[j:j + 64]); ser.flush(); time.sleep(0.003)

def leer():
    info = None; P = []; errores = []
    while True:
        l = ser.readline().decode(errors="ignore").strip()
        if not l:
            raise SystemExit("La Pico no responde. ¿main.py está en la placa y Thonny cerrado?")
        if l == "END":
            return info, P, errores
        if l.startswith("I,"): info = l.split(",")[1:]
        elif l.startswith("P,"): P.append(l.split(",")[1:])
        elif l.startswith("ERR"): errores.append(l)

ser = serial.Serial(PORT, 115200, timeout=30)
ser.reset_input_buffer()

input("CALIBRACIÓN: sin el objeto delante (solo el cuarto), silencio total. Enter...")
print("Calibrando (~%d s)..." % int(NFR_BASE * (REC_S + 0.15)))
bas = apilar(NFR_BASE)
if bas is None:
    raise SystemExit("No se pudo calibrar (ver motivos arriba).")
print("Calibración lista.")
input("Pon el objeto frente a la laptop. Enter para empezar (Ctrl+C para terminar)...")

ciclo = 0
try:
    while True:
        tgt = apilar(NFR_MED)
        if tgt is None:
            print("[%d] sin señal del micrófono" % ciclo)
            continue
        tgt_al, lag = alinear(bas, tgt)
        alpha = float(np.dot(tgt_al[ZA:ZB], bas[ZA:ZB]) / np.dot(bas[ZA:ZB], bas[ZA:ZB]))
        dif = tgt_al - alpha * bas
        rel = np.sqrt(np.mean(dif**2)) / np.sqrt(np.mean(bas**2))
        enviar(np.concatenate([bas, dif]))
        info, P, errores = leer()
        if errores or not P:
            print("[%d] error de la Pico: %s" % (ciclo, errores[0] if errores else "sin picos"))
            continue
        n_, ms_, m_, rel_, xr = P[0]
        otros = " | otros: " + ", ".join("%.2f m (x%s)" % (float(p[2]), p[4]) for p in P[1:])
        aviso = "  [aviso: el entorno cambió respecto a la calibración]" if rel > 0.6 else ""
        if float(xr) >= UMBRAL:
            print("[%d] DISTANCIA ESTIMADA: %.2f m (eco en %s ms, x%s del ruido)%s%s"
                  % (ciclo, float(m_), ms_, xr, otros, aviso))
        else:
            print("[%d] sin eco claro (mejor candidato %.2f m, x%s)%s%s"
                  % (ciclo, float(m_), xr, otros, aviso))
        ciclo += 1
except KeyboardInterrupt:
    print("Fin.")