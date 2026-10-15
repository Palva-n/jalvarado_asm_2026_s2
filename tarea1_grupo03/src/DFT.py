import time

import numpy as np


def DFT(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)                 # k como columna y n como fila
    W = np.exp(-2j * np.pi * k * n / N)  # matriz N x N de exponenciales
    return W @ x


def FFT(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)
    if N == 0 or N & (N - 1) != 0:
        raise ValueError(f"La FFT radix-2 requiere N potencia de 2 (N = {N})")
    if N == 1:
        return x.copy()

    E = FFT(x[0::2])                                     # DFT de las muestras pares
    O = FFT(x[1::2])                                     # DFT de las muestras impares
    W = np.exp(-2j * np.pi * np.arange(N // 2) / N)      # factores de giro
    return np.concatenate([E + W * O,                    # k = 0 ... N/2 - 1
                           E - W * O])                   # k = N/2 ... N - 1


def IFFT(X):
    X = np.asarray(X, dtype=complex)
    return np.conj(FFT(np.conj(X))) / len(X)


def _tiempo(funcion, x, repeticiones=5):
    mejor = np.inf
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        funcion(x)
        mejor = min(mejor, time.perf_counter() - t0)
    return mejor


def verificar(tamanos=(64, 256, 1024, 2048), semilla=2026):
    rng = np.random.default_rng(semilla)
    filas = []
    for N in tamanos:
        x = rng.standard_normal(N)
        ref = np.fft.fft(x)
        filas.append({
            "N": N,
            "error_DFT": float(np.max(np.abs(DFT(x) - ref))),
            "error_FFT": float(np.max(np.abs(FFT(x) - ref))),
            "error_IFFT": float(np.max(np.abs(IFFT(FFT(x)) - x))),
            "t_DFT_ms": _tiempo(DFT, x) * 1e3,
            "t_FFT_ms": _tiempo(FFT, x) * 1e3,
            "t_numpy_ms": _tiempo(np.fft.fft, x) * 1e3,
        })
    return filas


if __name__ == "__main__":
    print(f"{'N':>6s} {'err DFT':>10s} {'err FFT':>10s} {'err IFFT':>10s} "
          f"{'t DFT [ms]':>11s} {'t FFT [ms]':>11s} {'t numpy [ms]':>13s}")
    for f in verificar():
        print(f"{f['N']:>6d} {f['error_DFT']:>10.1e} {f['error_FFT']:>10.1e} {f['error_IFFT']:>10.1e} "
              f"{f['t_DFT_ms']:>11.3f} {f['t_FFT_ms']:>11.3f} {f['t_numpy_ms']:>13.4f}")