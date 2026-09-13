import numpy as np

class SignalGenerator:

    def __init__(self, sample_rate, dur, f_sonido, A):
        self.sample_rate = sample_rate
        self.dur = dur
        self.f_sonido = f_sonido
        self.Amplitude = A


    def generate_signal(self, type):
        t = np.linspace(0, self.dur, int(self.dur * self.sample_rate), endpoint=False)
        if type is None:
            raise ValueError("No se ingreso un tipo")
        if type == "Coseno":
            return self.Amplitude * np.cos(2*np.pi*self.f_sonido*t)
        if type == "Seno":
            return self.Amplitude * np.sin(2*np.pi*self.f_sonido*t)
        
    

    
    def generate_chirp(self, f0, f1):
        N = int(self.dur * self.sample_rate)
        t = np.linspace(0, self.dur, N, endpoint=False)
        k = (f1 - f0) / self.dur
        fase = 2*np.pi*(f0*t + 0.5*k*t**2)
        return self.Amplitude * np.cos(fase)

    def eco(self, senial, samples, atenuacion=0.5):
        N = len(senial)
        eco_signal = np.zeros(N)
        eco_signal[samples:] = senial[:N - samples] * atenuacion
        return senial + eco_signal

    def ruido(self, senial):
        num_samples = len(senial)
        white_noise = self.Amplitude * np.random.uniform(-1.0, 1.0, num_samples)
        return senial + white_noise

    def correlacion(self, x, y):
        N = len(x)
        M = len(y)
        results = []
        max_lag = N
        for k in range(-max_lag, max_lag):
            suma = 0
            for n in range(N):
                if 0 <= n + k < M:
                    suma += x[n] * y[n + k]
            results.append(suma)
        return np.array(results)

    

    def correlacion_fft(self, x, y):
        Nfft = 2*len(x)
        X = np.fft.fft(x, n=Nfft)
        Y = np.fft.fft(y, n=Nfft)
        corr_full = np.real(np.fft.ifft(np.conj(X) * Y))
        corr = np.concatenate([corr_full[-len(x):], corr_full[:len(x)]])
        return corr

    def encontrar_retardo(self, x, y, max_lag, metodo="directa"):
        if metodo == "directa":
            corr_total = self.correlacion(x, y)
            corr_directa = self.correlacion(x, x)
        elif metodo == "fft":
            corr_total = self.correlacion_fft(x, y)
            corr_directa = self.correlacion_fft(x, x)
        else:
            raise ValueError("metodo debe ser 'directa' o 'fft'")
        
        corr_sin_leakage = corr_total - corr_directa
        idx_max = np.argmax(corr_sin_leakage)
        return idx_max - max_lag