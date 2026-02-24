# Sistema de Transcripción de Reuniones

Este proyecto utiliza **Whisper** (versión optimizada `faster-whisper`) para transcribir grabaciones de audio a texto de forma local, gratuita y privada.

## Instalación (Ya realizada)
1. Instalar FFmpeg y dependencias de Python:
   ```bash
   sudo apt-get install ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev python3.13-venv python3.13-dev
   ```
2. Crear entorno virtual e instalar librerías:
   ```bash
   python3 -m venv venv
   ./venv/bin/pip install -r requirements.txt
   ```

## Cómo usarlo

1. **Coloca tus archivos de audio** (.m4a, .mp3, etc.) en la carpeta `input/`.
2. **Ejecuta el script**:
   ```bash
   ./venv/bin/python transcribe.py
   ```
3. **Recoge tus transcripciones** en la carpeta `output/`.

## Configuración
En `transcribe.py` puedes cambiar:
- `MODEL_SIZE`: "large-v3" (mejor calidad, más lento) o "medium", "small" (más rápido).
- `DEVICE`: "auto" (usa GPU si hay drivers NVIDIA, si no CPU).

---

## ⚡ Transcripción Rápida con Hilos (Recomendado para archivos largos)

Si quieres reducir el tiempo de espera transcribiendo **archivos largos** (como reuniones de 1 o 2 horas), puedes utilizar la versión **concurrente** (`transcribe_v2.py`). 

Esta versión corta el audio en pequeños fragmentos de 10 minutos y usa múltiples hilos de tu procesador/GPU para transcribirlos en paralelo.

### Cómo usar la Versión 2:
El proceso es exactamente el mismo, solo cambia el script que ejecutas:

1. **Coloca tus archivos** en la carpeta `input/`.
2. **Ejecuta el script V2**:
   ```bash
   ./venv/bin/python transcribe_v2.py
   ```
3. **Recoge tus transcripciones** en `output/`.

### Configuración de V2:
En `transcribe_v2.py` puedes ajustar el rendimiento según tu máquina:
- `NUM_THREADS`: Número de hilos simultáneos (Por defecto `2`). **Atención**: Si pones muchos hilos, cada hilo instanciará su propio modelo Whisper, por lo que requerirá multiplicar la cantidad de memoria RAM y VRAM necesaria. Redúcelo si te da error de memoria.
- `CHUNK_LENGTH_MIN`: Cuántos minutos dura cada fragmento (Por defecto `10`).
- `OVERLAP_SEC`: Cuántos segundos de solapamiento mantener entre cortes para evitar cortar palabras a la mitad (Por defecto `2`).
