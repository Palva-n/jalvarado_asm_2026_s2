import sys, math, time

try:
    from ulab import numpy as np
except ImportError:
    print("Falta ulab: necesitas un firmware de MicroPython que la incluya.")
    raise SystemExit

FS = 16000
F0, F1 = 1500, 6000
DUR = 0.010
NT = int(DUR * FS)
NX = 512
L = 1024
C = 343.0
DIRECT_WIN = 320        # el sonido directo (baseline) debe caer aquí
MIN_SEP = 16            # ignora los primeros ~1 ms tras el directo

# Plantilla del chirp y su FFT (una sola vez, en la Pico)
n_ = np.array(list(range(NT)), dtype=np.float)
t_ = n_ / FS
w_ = -0.5 * np.cos(2 * math.pi * n_ / (NT - 1)) + 0.5
chirp = np.sin(2 * math.pi * (F0 * t_ + (F1 - F0) * t_ * t_ / (2 * DUR))) * w_
tp = np.zeros(L)
tp[:NT] = chirp
TC = np.fft.fft(tp)

def corr(xp):
    X = np.fft.fft(xp)
    if isinstance(X, tuple):
        ar, ai = X
        cr, ci = TC
        y, _ = np.fft.ifft(ar * cr + ai * ci, ai * cr - ar * ci)
        return y
    return np.real(np.fft.ifft(X * np.conjugate(TC)))

def env_de(seg):
    media = sum(seg) / len(seg)
    x = (np.array(seg, dtype=np.float) - media) / 1000
    xp = np.zeros(L)
    xp[:NX] = x
    y = corr(xp)
    env = [abs(v) for v in y[:NX]]
    env2 = [0.0] * NX
    for k in range(NX):
        lo = k - 3 if k > 3 else 0
        hi = k + 4 if k + 4 < NX else NX
        env2[k] = max(env[lo:hi])
    return env2

def procesar(m):
    t0 = time.ticks_ms()
    base = env_de(m[:NX])          # baseline: contiene el sonido directo
    dif = env_de(m[NX:])           # diferencia (con pared - baseline)
    idx0 = 0
    p0 = base[0]
    for k in range(DIRECT_WIN):
        if base[k] > p0:
            p0 = base[k]
            idx0 = k
    if p0 <= 0:
        p0 = 1e-9
    s = sorted(dif[NX // 2:])
    ruido = s[len(s) // 2]
    if ruido < 1e-9:
        ruido = 1e-9
    tmp = list(dif)
    for k in range(min(NX, idx0 + MIN_SEP)):
        tmp[k] = 0.0
    picos = []
    for _ in range(3):
        best = 0
        bv = tmp[0]
        for k in range(NX):
            if tmp[k] > bv:
                bv = tmp[k]
                best = k
        picos.append((best, bv))
        for k in range(max(0, best - 20), min(NX, best + 21)):
            tmp[k] = 0.0
    ms = time.ticks_diff(time.ticks_ms(), t0)
    print("I,%d,%d" % (idx0, ms))
    for n, (idx, v) in enumerate(picos):
        tau = (idx - idx0) / FS
        print("P,%d,%.2f,%.2f,%.4f,%.1f" % (n + 1, tau * 1000, C * tau / 2, v / p0, v / ruido))
    e = [int(10000 * max(dif[k:k + 4]) / p0) for k in range(0, NX, 4)]
    print("E," + ",".join(str(v) for v in e))
    print("END")

while True:
    linea = sys.stdin.readline()
    if not linea:
        continue
    linea = linea.strip()
    if not linea.startswith("M,"):
        continue
    try:
        m = [int(v) for v in linea[2:].split(",")]
        if len(m) != 2 * NX:
            print("ERR,largo %d (se esperaban %d)" % (len(m), 2 * NX))
            print("END")
            continue
        procesar(m)
    except Exception as ex:
        print("ERR,%s" % ex)
        print("END")
