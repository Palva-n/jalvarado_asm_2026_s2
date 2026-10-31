import binascii
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

CARPETA_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA_SRC))
import comparacion as cmp 
import filtrado_espectral as fe  

RUTA_FIGURAS = CARPETA_SRC.parent / "resultados" / "figuras"


FS = 16_000      
FS_REAL_PICO = 16_220  
                       
N = 1024              
HOP = N // 2          

BITS_ADC = 12
ADC_MEDIO = 2 ** (BITS_ADC - 1)


FRECUENCIAS_UTILES = [440, 880, 1320] 
AMPLITUDES_UTILES = [1.0, 0.55, 0.30]
F_MIN_UTIL = 300                           
F_MAX_UTIL = 1500                         

TRAMOS_INTERFERENCIA = [(0.0, 3.0, 1750), (3.0, 6.0, 2500)]
SNR_ENTRADA_DB = 5.0
PICO_MAXIMO = 0.8

Q_NOTCH = 10      
HISTERESIS_HZ = 0.5

UMBRAL_PROMINENCIA_DB = 20.0
EPS = 1e-12

SINCRONIA = 0xAA55
FORMATO_CABECERA = ">HHHh"
FORMATO_TRAMA = FORMATO_CABECERA + f"{HOP}h"
ESCALA_FREC = 8 
BYTES_TRAMA = struct.calcsize(FORMATO_TRAMA) + 2 

INICIO_METRICAS = FS

def generar_senal_prueba(duracion=6.0, fs_real=FS_REAL_PICO, semilla=2026):
    rng = np.random.default_rng(semilla)
    n = np.arange(int(fs_real * duracion))
    t_real = n / fs_real                  
    a_aparente = FS / fs_real                 

    limpia = sum(a * np.sin(2 * np.pi * f * t_real + rng.uniform(-np.pi, np.pi))
                 for f, a in zip(FRECUENCIAS_UTILES, AMPLITUDES_UTILES))

    f_real = np.zeros(len(n))
    tramos = []
    for ini, fin, f in TRAMOS_INTERFERENCIA:
        f_real[(t_real >= ini) & (t_real < fin)] = f
        tramos.append({"ini": int(ini * fs_real), "fin": int(fin * fs_real),
                       "f_real": f, "f_aparente": f * a_aparente})
    fase = 2 * np.pi * np.cumsum(f_real) / fs_real
    interferencia = np.sin(fase)

    p_limpia = np.mean(limpia ** 2)
    p_interf = np.mean(interferencia ** 2)
    interferencia *= np.sqrt(p_limpia / (p_interf * 10 ** (SNR_ENTRADA_DB / 10)))

    contaminada = limpia + interferencia
    escala = PICO_MAXIMO / np.max(np.abs(contaminada))
    return {
        "t": n / FS,                        
        "limpia": limpia * escala,
        "contaminada": contaminada * escala,
        "f_aparente": f_real * a_aparente,   
        "tramos": tramos,
        "frecuencias_utiles": [f * a_aparente for f in FRECUENCIAS_UTILES],
    }

def adc_12bits(x):
    codigos = np.round((x + 1) / 2 * (2 ** BITS_ADC - 1))
    return np.clip(codigos, 0, 2 ** BITS_ADC - 1).astype(np.uint16)


def cuentas_a_float(muestras_int16):
    return np.asarray(muestras_int16, dtype=np.float64) / ADC_MEDIO


def empaquetar_trama(n_bloque, frecuencia_hz, nivel_db, muestras_int16):
    frec_codificada = int(round(frecuencia_hz * ESCALA_FREC))
    nivel_codificado = int(np.clip(round(nivel_db * 10), -32768, 32767))
    cuerpo = struct.pack(FORMATO_TRAMA, SINCRONIA, n_bloque % 65536,
                         frec_codificada, nivel_codificado, *muestras_int16)
    crc = binascii.crc_hqx(cuerpo, 0xFFFF)
    return cuerpo + struct.pack(">H", crc)


def desempaquetar_trama(trama):
    if len(trama) != BYTES_TRAMA:
        return None
    cuerpo, crc_recibido = trama[:-2], struct.unpack(">H", trama[-2:])[0]
    if binascii.crc_hqx(cuerpo, 0xFFFF) != crc_recibido:
        return None
    campos = struct.unpack(FORMATO_TRAMA, cuerpo)
    if campos[0] != SINCRONIA:
        return None
    return {
        "n_bloque": campos[1],
        "frecuencia_hz": campos[2] / ESCALA_FREC,
        "nivel_db": campos[3] / 10,
        "muestras": np.array(campos[4:], dtype=np.int16),
    }


def canal(trama, prob_error=0.0, rng=None):
    if rng is not None and prob_error > 0 and rng.random() < prob_error:
        alterada = bytearray(trama)
        alterada[rng.integers(len(alterada))] ^= 0xFF
        return bytes(alterada)
    return trama



class Nodo1:
    def __init__(self, interpolar=True):
        self.anterior = np.zeros(HOP, dtype=np.int16) 
        self.ventana = np.hanning(N)
        self.interpolar = interpolar 
        self.n_bloque = 0
        self.historial_frecuencia = [] 
    def procesar_bloque(self, codigos_adc):
        nuevas = (codigos_adc.astype(np.int32) - ADC_MEDIO).astype(np.int16)

        bloque = np.concatenate([self.anterior, nuevas])
        self.anterior = nuevas

        magnitud = np.abs(np.fft.rfft(bloque * self.ventana))

        if self.n_bloque == 0:
            frecuencia_hz, nivel_db = 0.0, 0.0
        else:
            frecuencia_hz, nivel_db = self.estimar_interferencia(magnitud)

        self.historial_frecuencia.append(frecuencia_hz)
        trama = empaquetar_trama(self.n_bloque, frecuencia_hz, nivel_db, nuevas)
        self.n_bloque += 1
        return trama

    def estimar_interferencia(self, magnitud):
        k_min = int(np.ceil(F_MAX_UTIL / (FS / N)))
        k = k_min + int(np.argmax(magnitud[k_min:-1]))

        piso = np.median(magnitud) + EPS
        nivel_db = float(20 * np.log10(magnitud[k] / piso))
        if nivel_db < UMBRAL_PROMINENCIA_DB:
            return 0.0, nivel_db

        p = 0.0
        if self.interpolar:
            a, b, c = np.log(magnitud[k - 1:k + 2] + EPS)
            denominador = a - 2 * b + c
            if denominador != 0:
                p = 0.5 * (a - c) / denominador
        return float((k + p) * FS / N), nivel_db


def disenar_notch(f0, q=Q_NOTCH, fs=FS):
    w0 = 2 * np.pi * f0 / fs
    beta = np.tan(w0 / (2 * q))
    g = 1 / (1 + beta)
    b = g * np.array([1.0, -2 * np.cos(w0), 1.0])
    a = np.array([1.0, -2 * g * np.cos(w0), 2 * g - 1])
    return b, a


class Nodo2:
    def __init__(self):
        self.frecuencia_actual = 0.0
        self.b = np.array([1.0, 0.0, 0.0])
        self.a = np.array([1.0, 0.0, 0.0])
        self.estado = np.zeros(2)          
        self.esperado = 0 
        self.tramas_perdidas = 0
        self.tramas_con_error = 0
        self.recalculos = 0            

    def procesar_bloque(self, trama):
        datos = desempaquetar_trama(trama)
        if datos is None:
            self.tramas_con_error += 1
            return None
        if datos["n_bloque"] != self.esperado % 65536:
            self.tramas_perdidas += (datos["n_bloque"] - self.esperado) % 65536
        self.esperado = datos["n_bloque"] + 1

        self.actualizar_notch(datos["frecuencia_hz"])

        x = cuentas_a_float(datos["muestras"])
        salida, self.estado = signal.lfilter(self.b, self.a, x, zi=self.estado)

        salida_int16 = np.clip(np.round(salida * ADC_MEDIO), -32768, 32767).astype(np.int16)
        trama_salida = empaquetar_trama(datos["n_bloque"], self.frecuencia_actual, 0.0, salida_int16)
        return datos["n_bloque"], salida, trama_salida

    def actualizar_notch(self, frecuencia_hz):
        if frecuencia_hz == 0:
            if self.frecuencia_actual != 0:
                self.b = np.array([1.0, 0.0, 0.0])
                self.a = np.array([1.0, 0.0, 0.0])
                self.frecuencia_actual = 0.0
            return
        if abs(frecuencia_hz - self.frecuencia_actual) > HISTERESIS_HZ:
            self.b, self.a = disenar_notch(frecuencia_hz)
            self.frecuencia_actual = frecuencia_hz
            self.recalculos += 1


class Nodo3:
    def __init__(self, parametros=None):
        self.p = dict(fe.PARAMETROS) if parametros is None else parametros
        self.ventana = np.sqrt(signal.get_window("hann", N))
        self.frecuencias = np.fft.rfftfreq(N, 1 / FS)
        self.anterior = np.zeros(HOP)
        self.cola = np.zeros(HOP)
        self.ruido_est = None
        self.n_bloque_anterior = None
        self.salidas_temporales = {}   
        self.tramas_con_error = 0

    def recibir_de_nodo2(self, trama):
        datos = desempaquetar_trama(trama)
        if datos is None:
            self.tramas_con_error += 1
            return
        self.salidas_temporales[datos["n_bloque"]] = cuentas_a_float(datos["muestras"])

    def procesar_bloque(self, trama):
        datos = desempaquetar_trama(trama)
        if datos is None:
            self.tramas_con_error += 1
            return None
        x = cuentas_a_float(datos["muestras"])
        X = np.fft.rfft(np.concatenate([self.anterior, x]) * self.ventana)
        self.ruido_est = fe.estimar_ruido(np.abs(X) ** 2, self.ruido_est, self.p["suavizado"])
        G = fe.calcular_ganancia(X, self.ruido_est, self.frecuencias, F_MIN_UTIL, F_MAX_UTIL,
                                 self.p["agresividad"], self.p["piso_ganancia"],
                                 self.p["prominencia"], self.p["ancho_notch"], self.p["piso_notch"])

        y = np.fft.irfft(G * X, n=N) * self.ventana

        terminada = self.cola + y[:HOP]
        self.cola = y[HOP:].copy()

        self.anterior = x
        resultado = None
        if self.n_bloque_anterior is not None:
            resultado = (self.n_bloque_anterior, terminada)
        self.n_bloque_anterior = datos["n_bloque"]
        return resultado


def simular(prob_error=0.0, interpolar=True, fs_real=FS_REAL_PICO, semilla=2026):
    senal = generar_senal_prueba(fs_real=fs_real, semilla=semilla)
    contaminada = senal["contaminada"]
    rng = np.random.default_rng(semilla)
    n_bloques = len(contaminada) // HOP

    nodo1, nodo2, nodo3 = Nodo1(interpolar), Nodo2(), Nodo3()
    salida_temporal = np.full(n_bloques * HOP, np.nan)
    salida_espectral = np.full(n_bloques * HOP, np.nan)

    for m in range(n_bloques):                           
        codigos = adc_12bits(contaminada[m * HOP:(m + 1) * HOP])
        trama_1 = nodo1.procesar_bloque(codigos)

        r2 = nodo2.procesar_bloque(canal(trama_1, prob_error, rng))
        if r2 is not None:
            n, salida, trama_2 = r2
            salida_temporal[n * HOP:(n + 1) * HOP] = salida
            nodo3.recibir_de_nodo2(canal(trama_2, prob_error, rng))

        r3 = nodo3.procesar_bloque(canal(trama_1, prob_error, rng))
        if r3 is not None:
            n, salida = r3
            salida_espectral[n * HOP:(n + 1) * HOP] = salida

    largo = n_bloques * HOP
    return {
        "t": senal["t"][:largo], "limpia": senal["limpia"][:largo],
        "contaminada": contaminada[:largo], "f_aparente": senal["f_aparente"][:largo],
        "tramos": senal["tramos"], "frecuencias_utiles": senal["frecuencias_utiles"],
        "temporal": salida_temporal, "espectral": salida_espectral,
        "f_estimada": np.array(nodo1.historial_frecuencia),
        "recalculos_nodo2": nodo2.recalculos,
        "errores": {"nodo2": nodo2.tramas_con_error, "nodo3": nodo3.tramas_con_error,
                    "perdidas_nodo2": nodo2.tramas_perdidas},
    }


BLOQUES_ADAPTACION = 2


def evaluar(res):

    filas = []
    for tramo in res["tramos"]:
        a = max(tramo["ini"] + BLOQUES_ADAPTACION * HOP, INICIO_METRICAS)
        b = min(tramo["fin"], len(res["limpia"]))
        s, x = res["limpia"][a:b], res["contaminada"][a:b]
        for metodo in ("temporal", "espectral"):
            y = res[metodo][a:b]
            validas = ~np.isnan(y)
            snr_in = cmp.snr_db(s[validas], x[validas])
            snr_out = cmp.snr_db(s[validas], y[validas])
            filas.append({"tramo": f"{tramo['f_real']} Hz", "metodo": metodo,
                          "snr_entrada": snr_in, "snr_salida": snr_out,
                          "delta_snr": snr_out - snr_in,
                          "energia_util": cmp.energia_util_pct(s[validas], y[validas], FS,
                                                               res["frecuencias_utiles"])})
    return filas


def error_estimacion(res):
    f_est = res["f_estimada"]
    fin_ventana = (np.arange(len(f_est)) + 1) * HOP
    ini_ventana = fin_ventana - N
    filas = []
    for tramo in res["tramos"]:
        dentro = (ini_ventana >= tramo["ini"]) & (fin_ventana <= tramo["fin"])
        dentro[0] = False
        errores = np.abs(f_est[dentro] - tramo["f_aparente"])
        filas.append({"f_real": tramo["f_real"], "f_aparente": tramo["f_aparente"],
                      "bloques": int(dentro.sum()), "error_medio": float(errores.mean()),
                      "error_max": float(errores.max())})
    return filas


def latencia_adaptacion(res, metodo, factor=1.5):
    error = res[metodo] - res["limpia"]
    n_bloques = len(error) // HOP
    rms = np.full(n_bloques, np.nan)            # nan = bloque sin salida
    for m in range(n_bloques):
        bloque = error[m * HOP:(m + 1) * HOP]
        if not np.all(np.isnan(bloque)):
            rms[m] = np.sqrt(np.nanmean(bloque ** 2))
    m_salto = res["tramos"][1]["ini"] // HOP
    estable = np.nanmedian(rms[m_salto + 5:])   # nan = bloque sin salida
    bloques = 0
    while m_salto + bloques < n_bloques and rms[m_salto + bloques] > factor * estable:
        bloques += 1
    return bloques, bloques * HOP / FS * 1e3


def validar_nodo2():

    errores_coef = []
    for f0 in (1726.26, 2466.09, 1750, 2500):
        b, a = disenar_notch(f0)
        b_ref, a_ref = signal.iirnotch(f0, Q_NOTCH, FS)
        errores_coef.append(max(np.max(np.abs(b - b_ref)), np.max(np.abs(a - a_ref))))

    rng = np.random.default_rng(7)
    x_int16 = np.round(rng.uniform(-0.5, 0.5, 40 * HOP) * ADC_MEDIO).astype(np.int16)
    nodo2 = Nodo2()
    por_bloques = []
    for m in range(40):
        trama = empaquetar_trama(m, 1750, 60.0, x_int16[m * HOP:(m + 1) * HOP])
        por_bloques.append(nodo2.procesar_bloque(trama)[1])
    b, a = disenar_notch(1750)
    completa = signal.lfilter(b, a, cuentas_a_float(x_int16))
    return max(errores_coef), float(np.max(np.abs(np.concatenate(por_bloques) - completa)))


def validar_nodo3():
    senal = generar_senal_prueba()
    n_bloques = len(senal["contaminada"]) // HOP
    x_int16 = (adc_12bits(senal["contaminada"][:n_bloques * HOP]).astype(np.int32)
               - ADC_MEDIO).astype(np.int16)
    x = cuentas_a_float(x_int16)

    def correr(nodo):
        salida = np.full(len(x), np.nan)
        for m in range(n_bloques):
            r = nodo.procesar_bloque(empaquetar_trama(m, 0, 0, x_int16[m * HOP:(m + 1) * HOP]))
            if r is not None:
                salida[r[0] * HOP:(r[0] + 1) * HOP] = r[1]
        return salida

    transparente = dict(fe.PARAMETROS, agresividad=1.0, piso_ganancia=1.0, piso_notch=1.0)
    y1 = correr(Nodo3(transparente))
    validas = ~np.isnan(y1)
    err_g1 = float(np.max(np.abs(y1[validas] - x[validas])))

    y = correr(Nodo3())
    ref = fe.procesar(x, FS, N, HOP, F_MIN_UTIL, F_MAX_UTIL, **fe.PARAMETROS)
    dif = np.abs(y - ref)
    dif_por_segundo = [float(np.nanmax(dif[k * FS:(k + 1) * FS])) for k in range(len(dif) // FS)]
    return err_g1, dif_por_segundo


def graficar(res, ruta):
    t_bloques = (np.arange(len(res["f_estimada"])) + 1) * HOP / FS
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(res["t"], res["f_aparente"], color="#888888",
            label=f"Esperada (aparente con fs real = {FS_REAL_PICO} Hz)")
    detectada = res["f_estimada"] > 0       # 0 = "no hay interferencia clara"
    ax.plot(t_bloques[detectada], res["f_estimada"][detectada], "o", markersize=3,
            color="#2a78d6", label="Estimada por el Nodo 1")
    ax.set(xlabel="Tiempo [s]", ylabel="Frecuencia [Hz]",
           title="Seguimiento de la interferencia por bloque")
    ax.grid(True, color="#e5e5e5")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def main():
    print(f"Bloque: {HOP} muestras nuevas = {HOP / FS * 1e3:.0f} ms | "
          f"FFT de {N} puntos, df = {FS / N:.2f} Hz | trama = {BYTES_TRAMA} bytes")

    print(f"Senal muestreada a fs real = {FS_REAL_PICO} Hz; los nodos suponen {FS} Hz")

    res = simular()
    res_sin_interp = simular(interpolar=False)

    print("\nNodo 1 - estimacion de la interferencia (sin bloques de transicion):")
    print("f real   f aparente   bloques   error medio [Hz]        error max [Hz]")
    print("                                con / sin interp.       con / sin interp.")
    for con, sin in zip(error_estimacion(res), error_estimacion(res_sin_interp)):
        print(f"{con['f_real']:6d}   {con['f_aparente']:10.2f}   {con['bloques']:7d}   "
              f"{con['error_medio']:7.3f} / {sin['error_medio']:6.3f}       "
              f"{con['error_max']:7.3f} / {sin['error_max']:6.3f}")

    err_coef, err_bloques = validar_nodo2()
    print("\nNodo 2 - validacion:")
    print(f"  coeficientes propios vs scipy.iirnotch: error max = {err_coef:.1e}")
    print(f"  filtrado por bloques vs senal completa: error max = {err_bloques:.1e}")
    print(f"  recalculos del notch en la simulacion: {res['recalculos_nodo2']}")
    bloques, ms = latencia_adaptacion(res, "temporal")
    print(f"  latencia de adaptacion al salto de frecuencia: {bloques} bloques ({ms:.0f} ms)")

    err_g1, dif_por_segundo = validar_nodo3()
    print("\nNodo 3 - validacion:")
    print(f"  reconstruccion con G = 1 (overlap-add): error max = {err_g1:.1e}")
    print("  diferencia max vs metodo de la Tarea 1, por segundo:")
    print("    " + "  ".join(f"{k}-{k + 1} s: {d:.0e}" for k, d in enumerate(dif_por_segundo)))
    print(f"  latencia de procesamiento: 1 bloque ({HOP / FS * 1e3:.0f} ms) por el overlap-add")
    bloques, ms = latencia_adaptacion(res, "espectral")
    print(f"  latencia de adaptacion al salto de frecuencia: {bloques} bloques ({ms:.0f} ms)")

    print(f"\nMetricas en regimen estable (sin los {BLOQUES_ADAPTACION} primeros bloques de cada tramo):")
    print("Tramo      Metodo      SNR in   SNR out    dSNR   E.util %")
    for f in evaluar(res):
        print(f"{f['tramo']:9s}  {f['metodo']:10s} {f['snr_entrada']:7.2f}  {f['snr_salida']:7.2f}  "
              f"{f['delta_snr']:6.2f}  {f['energia_util']:8.1f}")

    res_err = simular(prob_error=0.05)
    print(f"\nPrueba con 5 % de tramas alteradas: {res_err['errores']}")

    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_FIGURAS / "simulacion_seguimiento_interferencia.png"
    graficar(res, ruta)
    print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()