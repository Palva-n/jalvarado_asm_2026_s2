# Datos de hardware - Proyecto Grupal 1 (CE1110)

Corridas reales en la Pico W + MAX4466, con la PC como emisor (`pc_cliente.py`).
Todas: 31 bloques, 0 tramas perdidas, 0 errores CRC. fs real de la Pico ~16130 Hz.

## Archivos
- `corridas_hardware.csv`: una fila por corrida (nivel de entrada, latencias, memoria, metricas).
- `bandas_hardware.csv`: cambio de energia por banda (dB, negativo = se atenuo) para P3 y P4.

## Como leerlos
- `valida = no`: entrada demasiado debil (<500 rms). No usar en las tablas del paper.
- **Metricas de forma de onda** (`snr_*`, `dsnr`, `mse`, `eutil`, `atenuacion`): NO son confiables en el lazo acustico
  (deriva de reloj 16000 vs 16130 Hz, alineacion, ganancia distinta por frecuencia, fase del pasaaltos). Solo orientativas.
- **Metricas confiables:**
  - P1, P2, P5: `linea_*` (atenuacion de la interferencia y % de energia util conservada, medidas en frecuencia).
  - P3, P4: `bandas_hardware.csv` (cambio de energia por banda; la ultima banda es solo ruido).
- `T_` = temporal (nodo 2, notch+pasaaltos IIR+pasabajos FIR), `E_` = espectral (nodo 3, Wiener por bin).
- Configuracion agresiva (P3 agresivo, P4, P5): `--set prominencia=4 --set agresividad=3 --set piso_ganancia=0.03`.
- Banda util: P1/P2/P5 segun banco_pruebas.py; P3 = 80-3000 Hz. P4 usa la de su prueba.

## Notas / dudas
- P1: la corrida `P1_r1` (2170 rms) se tomo como repeticion 1; `P1_voz` fue una repeticion extra con voz del usuario
  durante la grabacion (no usarla en promedios). Verificar contra los wav guardados si hace falta.
- P1/P2: la configuracion exacta (prominencia) esta anotada segun lo dicho por el usuario, no capturada en la salida.
- P3_agr_a: no se capturo el fs real (la primera linea de la salida se cortó).
- P3 agresivo: reps 1-3 = `P3_agr_a`, `P3_agr_b`, `P3_agr_c` (los bloques de salida venian con etiquetas repetidas).
- P4: se comparo solo a oido con y sin parametros agresivos; ambos suenan alterados (los clicks cambian de timbre).
- Latencia por bloque: nodo1 ~84 ms, nodo2 ~103 ms, nodo3 ~257 ms (bloque = 31.7 ms), memoria libre ~116 KB.
