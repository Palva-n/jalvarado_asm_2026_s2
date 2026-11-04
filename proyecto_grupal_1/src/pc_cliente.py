
import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

CARPETA = Path(__file__).resolve().parent
sys.path.insert(0, str(CARPETA))
import banco_pruebas as bp          # generadores de senales y metricas

FS_PC = 16000                        # frecuencia con que la PC reproduce
FORMATO_CABECERA = ">HBHHh"          # sincronia, tipo, n_bloque, frecuencia*8, nivel*10
SINCRONIA = 0xAA55
TAM_CABECERA = struct.calcsize(FORMATO_CABECERA)
NOMBRES = {0: "entrada", 1: "temporal", 2: "espectral"}



# CRC-16 CCITT (mismo que la Pico)

def crc16(datos):
    import binascii
    return binascii.crc_hqx(datos, 0xFFFF)



# Comunicacion con la Pico

def encontrar_puerto():
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == 0x2E8A or "MicroPython" in (p.description or ""):
            return p.device
    return None


def enviar(ser, linea):
    ser.write((linea + "\n").encode())


def leer_linea(ser):
    return ser.readline().decode(errors="replace").strip()


def aplicar_parametros(ser, parametros):
    # parametros: lista de "clave=valor"
    for texto in parametros:
        clave, valor = texto.split("=")
        enviar(ser, f"SET {clave} {valor}")
        print("  Pico:", leer_linea(ser))


def leer_resultado(ser):
    # Espera la linea BEGIN, lee las tramas y la linea FIN.
    while True:
        linea = leer_linea(ser)
        if linea.startswith("BEGIN"):
            break
        if linea:
            print("  Pico:", linea)
    _, n_bloques, hop, n, fs_real, long_trama = linea.split()
    n_bloques, hop, long_trama = int(n_bloques), int(hop), int(long_trama)
    fs_real = float(fs_real)

    senales = {0: np.zeros(n_bloques * hop), 1: np.zeros(n_bloques * hop), 2: np.zeros(n_bloques * hop)}
    frecuencias = np.zeros(n_bloques)
    recibidas = {0: set(), 1: set(), 2: set()}
    errores_crc = 0

    print(f"  BEGIN recibido: la Pico termino de grabar (fs real {fs_real:.1f} Hz). "
          f"Esperando {n_bloques} bloques...", flush=True)
    t_inicio = time.time()
    for i_trama in range(n_bloques * 3):
        if i_trama % 3 == 0:
            print(f"  bloque {i_trama // 3 + 1}/{n_bloques}  ({time.time() - t_inicio:.1f} s)", flush=True)
        inicio = ser.read(2)
        if inicio != b"\xaa\x55":
            # No es una trama: la Pico mando texto (casi seguro un error)
            time.sleep(1.0)
            texto = inicio + ser.read(ser.in_waiting)
            print("\n  LA PICO REPORTO UN ERROR (bloque", i_trama // 3 + 1, "):")
            print(texto.decode(errors="replace"))
            sys.exit(1)
        trama = inicio + ser.read(long_trama - 2)
        if len(trama) != long_trama:
            break                                    # trama incompleta: se cuenta como perdida
        if crc16(trama[:-2]) != struct.unpack(">H", trama[-2:])[0]:
            errores_crc += 1
            continue
        sinc, tipo, m, f8, nivel = struct.unpack(FORMATO_CABECERA, trama[:TAM_CABECERA])
        if sinc != SINCRONIA or tipo not in senales or m >= n_bloques:
            errores_crc += 1
            continue
        senales[tipo][m * hop:(m + 1) * hop] = np.frombuffer(trama[TAM_CABECERA:-2], dtype="<i2") / 32768.0
        recibidas[tipo].add(m)
        if tipo == 0:
            frecuencias[m] = f8 / 8

    resto = leer_linea(ser)                          # "FIN t1 t2 t3 memoria"
    tiempos = [int(v) for v in resto.split()[1:]] if resto.startswith("FIN") else [0, 0, 0, 0]
    perdidas = sum(n_bloques - len(r) for r in recibidas.values())
    return {"senales": senales, "frecuencias": frecuencias, "fs_real": fs_real, "hop": hop,
            "n_bloques": n_bloques, "errores_crc": errores_crc, "perdidas": perdidas,
            "t_nodo_us": tiempos[:3], "mem_libre": tiempos[3]}



# Alineacion contra la referencia

def remuestrear_a_fs_pico(x, fs_real):
    # La PC reproduce a 16000 Hz y la Pico muestrea a fs_real: se pasa x al reloj de la Pico
    t_pico = np.arange(int(len(x) / FS_PC * fs_real)) / fs_real
    return np.interp(t_pico * FS_PC, np.arange(len(x)), x)


def alinear(entrada_pico, contaminada_pc, limpia_pc, fs_real):
    # 1. Correlacion cruzada para hallar donde cae la grabacion dentro de lo reproducido
    cont = remuestrear_a_fs_pico(contaminada_pc, fs_real)
    lim = remuestrear_a_fs_pico(limpia_pc, fs_real)
    cont = cont - cont.mean()
    corr = signal.correlate(cont, entrada_pico, mode="valid", method="fft")
    inicio = int(np.argmax(corr))
    ref_cont = cont[inicio:inicio + len(entrada_pico)]
    ref_limpia = lim[inicio:inicio + len(entrada_pico)]
    ref_limpia = ref_limpia - ref_limpia.mean()
    # 2. Ganancia del camino parlante -> microfono -> ADC. Se ajusta por separado la
    #    ganancia de la senal util y la del ruido (minimos cuadrados):
    #        entrada = a * limpia + b * ruido + (ruido propio del montaje)
    #    asi la interferencia, que es lo mas fuerte, no domina la escala del tono.
    ref_ruido = ref_cont - ref_limpia
    A = np.column_stack([ref_limpia, ref_ruido])
    a, b = np.linalg.lstsq(A, entrada_pico, rcond=None)[0]
    print(f"Ganancia de la senal util: {a:.3f} | del ruido anadido: {b:.3f}")
    return ref_limpia * a, inicio



# Metricas y reporte

def alinear_salidas(res, taps):
    hop = res["hop"]
    s = res["senales"]
    d_fir = (taps - 1) // 2          # retardo del FIR (fase lineal)
    temporal = np.concatenate([s[1][d_fir:], np.zeros(d_fir)])
    espectral = np.concatenate([s[2][hop:], np.zeros(hop)])   # el nodo 3 sale 1 bloque despues
    return {"entrada": s[0], "temporal": temporal, "espectral": espectral}


def reportar(salidas, limpia):
    hop = 512
    a, b = 2 * hop, len(limpia) - 2 * hop      # se ignoran los extremos (transitorios)
    s = limpia[a:b]
    x = salidas["entrada"][a:b]
    print("\nMetodo      SNR in   SNR out    dSNR    MSE         E.util %   Atenuacion dB")
    for nombre in ("temporal", "espectral"):
        y = salidas[nombre][a:b]
        snr_in, snr_out = bp.snr_db(s, x), bp.snr_db(s, y)
        print(f"{nombre:10s} {snr_in:7.2f}  {snr_out:7.2f}  {snr_out - snr_in:6.2f}  "
              f"{bp.mse(s, y):.3e}  {bp.energia_util_pct(s, y):8.1f}   "
              f"{bp.atenuacion_interferencia_db(s, x, y):8.1f}")


# Para las pruebas con tonos: frecuencias utiles y banda de interferencia (Hz)
LINEAS = {
    "P1": ([1000], (2150, 2250)),
    "P2": ([440, 880, 1320], (3000, 7000)),
    "P5": ([440, 880, 1320], (1150, 1600)),
}
ANCHO_TONO_HZ = 40


def potencia_banda(x, fs, bandas):
    # Potencia espectral de x dentro de la union de las bandas (lo, hi)
    x = x[1024:-1024] * np.hanning(len(x) - 2048)
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    mascara = np.zeros(len(f), dtype=bool)
    for lo, hi in bandas:
        mascara |= (f >= lo) & (f <= hi)
    return float(P[mascara].sum())


def reportar_lineas(salidas, fs, prueba):
    # No depende de la alineacion con la referencia: mide en frecuencia.
    tonos, banda = LINEAS[prueba]
    bandas_tono = [(f - ANCHO_TONO_HZ, f + ANCHO_TONO_HZ) for f in tonos]
    # La banda de interferencia no incluye los bins de los tonos utiles
    cortes = [banda]
    for lo_t, hi_t in bandas_tono:
        nuevos = []
        for lo, hi in cortes:
            if hi_t < lo or lo_t > hi:
                nuevos.append((lo, hi))
            else:
                if lo < lo_t:
                    nuevos.append((lo, lo_t))
                if hi_t < hi:
                    nuevos.append((hi_t, hi))
        cortes = nuevos
    p_tono_in = potencia_banda(salidas["entrada"], fs, bandas_tono)
    p_int_in = potencia_banda(salidas["entrada"], fs, cortes)
    print("\nMedicion en frecuencia (no depende de la alineacion):")
    print(f"Entrada: interferencia {10 * np.log10(p_int_in / p_tono_in):.1f} dB respecto a la senal util")
    print("Metodo      Atenuacion interferencia [dB]   Energia util conservada [%]")
    for nombre in ("temporal", "espectral"):
        p_tono = potencia_banda(salidas[nombre], fs, bandas_tono)
        p_int = potencia_banda(salidas[nombre], fs, cortes)
        print(f"{nombre:10s} {10 * np.log10(p_int_in / p_int):20.1f}   {100 * p_tono / p_tono_in:26.1f}")


# Para pruebas de banda ancha (P3, P4): bandas donde se mide el cambio de energia.
# La ultima banda de cada lista no tiene senal util: ahi se ve cuanto ruido se quita.
BANDAS = {
    "P3": [(80, 300), (300, 1500), (1500, 3000), (3000, 3500), (3500, 7500)],
    "P4": [(80, 600), (600, 1100), (1100, 3000), (3000, 7500)],
}


def reportar_bandas(salidas, fs, prueba):
    # Cambio de energia por banda respecto a la entrada. No depende de la alineacion:
    # con ruido de banda ancha, el SNR por forma de onda no es confiable.
    print("\nCambio de energia por banda respecto a la entrada (negativo = se atenuo):")
    print("Banda [Hz]        Temporal [dB]   Espectral [dB]")
    for lo, hi in BANDAS[prueba]:
        p_in = potencia_banda(salidas["entrada"], fs, [(lo, hi)])
        t = 10 * np.log10(potencia_banda(salidas["temporal"], fs, [(lo, hi)]) / p_in)
        e = 10 * np.log10(potencia_banda(salidas["espectral"], fs, [(lo, hi)]) / p_in)
        print(f"{lo:5d} - {hi:5d}   {t:12.1f}   {e:14.1f}")
    print("(la ultima banda no tiene senal util: ahi se mide cuanto ruido se quito)")


def guardar(salidas, fs_real, carpeta, etiqueta):
    carpeta.mkdir(parents=True, exist_ok=True)
    # Escala comun para los tres audios: asi se pueden comparar a oido sin que el
    # volumen cambie solo porque se quito la interferencia.
    pico = max(max(np.max(np.abs(v)) for v in salidas.values()), 1e-9)
    for nombre, v in salidas.items():
        wavfile.write(carpeta / f"{etiqueta}_{nombre}.wav", int(round(fs_real)),
                      (v / pico * 0.9 * 32767).astype(np.int16))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for nombre, color in [("entrada", "#d62728"), ("temporal", "#2a78d6"), ("espectral", "#2ca02c")]:
        v = salidas[nombre][1024:-1024]
        P = np.abs(np.fft.rfft(v * np.hanning(len(v)))) ** 2
        f = np.fft.rfftfreq(len(v), 1 / fs_real)
        ax.plot(f, 10 * np.log10(P / P.max() + 1e-12), label=nombre, color=color, lw=0.9)
    ax.set(xlabel="Frecuencia [Hz]", ylabel="Potencia [dB rel.]", ylim=(-100, 5), title=etiqueta)
    ax.grid(True, color="#e5e5e5")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(carpeta / f"{etiqueta}_espectros.png", dpi=150)
    plt.close(fig)



# Programa principal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", help="ej. COM5 (si se omite, se busca la Pico)")
    ap.add_argument("--prueba", choices=list(bp.PRUEBAS), help="P1..P5 de banco_pruebas.py")
    ap.add_argument("--repeticion", type=int, default=1)
    ap.add_argument("--senal", help="wav contaminado cualquiera (sin referencia, p. ej. P6)")
    ap.add_argument("--set", action="append", default=[], metavar="clave=valor",
                    help="parametro de la Pico, se puede repetir")
    ap.add_argument("--escuchar", choices=["entrada", "temporal", "espectral"],
                    help="reproducir la salida de ese metodo al final")
    ap.add_argument("--sin-audio", action="store_true", help="no reproducir: modo en vivo, otro equipo emite el audio y aqui solo se graba")
    args = ap.parse_args()

    import serial
    puerto = args.puerto or encontrar_puerto()
    if puerto is None:
        sys.exit("No se encontro la Pico. Cierre Thonny y use --puerto COMx")
    ser = serial.Serial(puerto, 115200, timeout=30)
    enviar(ser, "PING")
    print("Conectado:", leer_linea(ser))

    # Banda util de la prueba -> parametros de la Pico (se pueden sobreescribir con --set)
    parametros = []
    if args.prueba:
        s, x = bp.generar_prueba(args.prueba, args.repeticion)
        f_min, f_max = bp.PRUEBAS[args.prueba].get("banda_util", bp.BANDA_UTIL_DEFECTO)
        parametros += [f"f_min_util={f_min}", f"f_max_util={f_max}"]
        etiqueta = args.prueba
    elif args.senal:
        fs, x = wavfile.read(args.senal)
        x = x.astype(np.float64) / 32768.0 if x.dtype == np.int16 else x.astype(np.float64)
        if x.ndim > 1:
            x = x[:, 0]
        if fs != FS_PC:
            x = signal.resample_poly(x, FS_PC, fs)
        s = None
        etiqueta = Path(args.senal).stem
    elif args.sin_audio:
        # Modo en vivo: otro equipo (p. ej. el profe) reproduce el audio; aqui solo se graba
        s = None
        x = None
        etiqueta = "en_vivo"
    else:
        sys.exit("Indique --prueba, --senal o --sin-audio (modo en vivo)")
    aplicar_parametros(ser, parametros + args.set)

    # Se reproduce y, 1 s despues, se manda REC (la grabacion cae dentro de la senal)
    import sounddevice as sd
    if args.sin_audio:
        input("Modo en vivo: pulse Enter cuando el audio ya este sonando... ")
    else:
        sd.play(x * 0.5, FS_PC)           # volumen moderado: evitar saturar el MAX4466 y el parlante
        time.sleep(1.0)
    print("Grabando y procesando en la Pico (puede tardar varios segundos)...")
    enviar(ser, "REC")
    res = leer_resultado(ser)
    sd.stop()

    n = res["n_bloques"]
    print(f"fs real de la Pico: {res['fs_real']:.1f} Hz | bloques: {n} | "
          f"tramas perdidas: {res['perdidas']} | errores CRC: {res['errores_crc']}")
    t = res["t_nodo_us"]
    print(f"Latencia por bloque [ms]: nodo 1 {t[0] / 1000:.1f} | nodo 2 {t[1] / 1000:.1f} | "
          f"nodo 3 {t[2] / 1000:.1f}  (bloque = {res['hop'] / res['fs_real'] * 1000:.1f} ms) | "
          f"memoria libre {res['mem_libre']} B")

    taps = 63
    salidas = alinear_salidas(res, taps)

    # Nivel de la senal que llega al ADC (cuentas de 16 bits, sin la continua).
    # El ruido propio del ADC con WL_GPIO1 es de unas 76 cuentas rms.
    entrada = salidas["entrada"] * 32768
    rms, pico = float(np.sqrt(np.mean(entrada ** 2))), float(np.max(np.abs(entrada)))
    print(f"Nivel de entrada: {rms:.0f} cuentas rms, pico {pico:.0f} de 32768 "
          f"(ruido propio del ADC ~76 rms)")
    if rms < 500:
        print("  AVISO: senal muy debil frente al ruido del ADC. Suba el volumen de la PC, "
              "acerque el parlante al microfono o suba la ganancia del MAX4466.")
    if pico > 30000:
        print("  AVISO: el ADC esta casi saturado. Baje el volumen o la ganancia.")
    if s is not None:
        limpia, inicio = alinear(salidas["entrada"], x, s, res["fs_real"])
        print(f"Grabacion alineada con la referencia (inicio en muestra {inicio})")
        reportar(salidas, limpia)
        if args.prueba in LINEAS:
            reportar_lineas(salidas, res["fs_real"], args.prueba)
        if args.prueba in BANDAS:
            reportar_bandas(salidas, res["fs_real"], args.prueba)
    carpeta = CARPETA.parent / "resultados" / "hardware"
    guardar(salidas, res["fs_real"], carpeta, etiqueta)
    print("Guardado en", carpeta)

    if args.escuchar:
        v = salidas[args.escuchar]
        pico = max(np.max(np.abs(w)) for w in salidas.values())   # misma escala que los wav
        sd.play(v / max(pico, 1e-9) * 0.5, int(round(res["fs_real"])))
        sd.wait()


if __name__ == "__main__":
    main()
