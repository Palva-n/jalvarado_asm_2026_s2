import numpy as np
import matplotlib.pyplot as plt 
import time

class SignalAnalyzer:
    

    def pow_check(self, N):
        count = 0
        while N > 0:
            if N & 1:
                count += 1
            N = N >> 1
        return count == 1

    def __init__(self, X, fs):
        

        if X is not None: 
            if not self.pow_check(len(X)):
                raise ValueError("el tamaño de la señal debe ser potencia de 2")
        
        self.x = X
        self.fs = fs 
        self.DFT_ans = None 
        self.FFT_ans = None 


    def assign_signal(self, NewSignal, fs):
        if NewSignal is not None: 
            if not self.pow_check(len(NewSignal)):
                raise ValueError("el tamaño de la señal debe ser potencia de 2")
        
        self.x = NewSignal
        self.fs = fs 
        self.DFT_ans = None 
        self.FFT_ans = None 



    def DFT(self):
        if self.x is not None: 
            self.DFT_ans =  self.__DFT(self.x)

    def FFT(self):
        if self.x is not None: 
            self.FFT_ans = self.__FFT(self.x)

    def __DFT(self, x):
        size = len(x)
        X_ = np.zeros(size, dtype=np.complex64)
        for k in range(size):
            for n in range(size):
                X_[k] += x[n] * np.exp(-2j*np.pi*k*n/size)
        return X_

    
    
    def __FFT(self, x):

        N = len(x)
        if N <= 32:
            return self.__DFT(x)
        else:
            even_vals = np.zeros(N//2, dtype=float)
            odd_vals = np.zeros(N//2, dtype=float)
            for i in range(N):
                if i % 2 == 0:
                    even_vals[i//2] = x[i]
                else:
                    odd_vals[i//2] = x[i]

            
            Xeven = self.__FFT(even_vals)
            Xodd = self.__FFT(odd_vals)



            factor = np.exp(-2j*np.pi*np.arange(N)/N)
            return np.concatenate([Xeven + factor[:N//2]*Xodd,
            Xeven - factor[:N//2]*Xodd])



    def ImageGenerateDFT(self, name): 
        if self.DFT_ans is  None:
            raise ValueError("No se ha ejecutado DFT")
        if name is None: 
            raise ValueError("Nombre vacio o invalido")
        ans = self.DFT_ans 
        mitad = len(self.x)//2 

        mag_X = [np.abs(x) for x in ans]
        phas_X = [np.angle(x) for x in ans]

        frecs = np.array([i * (self.fs / len(self.x)) for i in range(len(self.x))], dtype=float)

        f_plot = frecs[:mitad]

        mag_plot = np.array(mag_X)[:mitad]
        phas_plot = np.array(phas_X)[:mitad] 
        #plt.figure() reinicia el canvas a blanco
        plt.figure() 
        plt.plot(f_plot, mag_plot, color="blue")
        plt.xlabel("frecuencia")
        plt.ylabel("magnitud")
        plt.xscale('log')
        plt.axhline(y=0, color="black")
        plt.grid(True)
        plt.savefig(f"imagenes/{name}_mag_DFT.png")
        plt.close() 

        plt.figure() 
        plt.plot(f_plot, phas_plot, color = 'blue')
        plt.xlabel("frecuencia")
        plt.ylabel("fase")
        plt.xscale('log')
        plt.axhline(y = 0, color = 'black') 
        plt.grid(True) 
        plt.savefig(f"imagenes/{name}_phase_DFT.png")
        plt.close() 



    def ImageGenerateFFT(self, name):
        if self.FFT_ans is None:
            raise ValueError("No se ha ejecutado FFT")
        if name is None: 
            raise ValueError("Nombre vacio o invalido")
        
        ans = self.FFT_ans 
        mitad = len(self.x)//2 
    
        mag_X = [np.abs(x) for x in ans]
        phas_X = [np.angle(x) for x in ans]
    
        frecs = np.array([i * (self.fs / len(self.x)) for i in range(len(self.x))], dtype=float)
            
        f_plot = frecs[:mitad]
    
        mag_plot = np.array(mag_X)[:mitad]
        phas_plot = np.array(phas_X)[:mitad] 
        #plt.figure() reinicia el canvas a blanco
        plt.figure() 
        plt.plot(f_plot, mag_plot, color="blue")
        plt.xlabel("frecuencia")
        plt.ylabel("magnitud")
        plt.xscale('log')
        plt.axhline(y=0, color="black")
        plt.grid(True)
        plt.savefig(f"imagenes/{name}_mag_FFT.png")
        plt.close() 
    
        plt.figure() 
        plt.plot(f_plot, phas_plot, color = 'blue')
        plt.xlabel("frecuencia")
        plt.ylabel("fase")
        plt.xscale('log')
        plt.axhline(y = 0, color = 'black') 
        plt.grid(True) 
        plt.savefig(f"imagenes/{name}_phase_FFT.png")
        plt.close() 


    

    def compare_times(self):

        if self.x is None:
            raise ValueError("No hay señal asignada")
        N = len(self.x)

        t0 = time.time()
        self.__DFT(self.x)
        t_dft = time.time() - t0

        t0 = time.time()
        self.__FFT(self.x)
        t_fft = time.time() - t0

        return t_dft, t_fft


    
    




    
