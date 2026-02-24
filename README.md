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
