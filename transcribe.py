import os
import glob
import time
from faster_whisper import WhisperModel

# Configuración
MODEL_SIZE = "large-v3"
# "cuda" para GPU, "cpu" para procesador. "auto" intenta detectar.
# Si tienes GPU NVIDIA, asegúrate de tener los drivers y CUDA instalados.
DEVICE = "auto" 
COMPUTE_TYPE = "float16" # "int8" si tienes poca VRAM o usas CPU

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def main():
    # Extensiones a buscar
    extensions = ['*.m4a', '*.mp3', '*.wav', '*.mp4', '*.mkv']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not files:
        print(f"No se encontraron archivos de audio en '{INPUT_DIR}'")
        return

    print(f"Se encontraron {len(files)} archivos para procesar.")

    print(f"Cargando modelo Whisper '{MODEL_SIZE}' en {DEVICE} con {COMPUTE_TYPE}...")
    try:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        print("Intentando fallback a CPU int8...")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    print("Modelo cargado.")

    for file_path in files:
        filename = os.path.basename(file_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(OUTPUT_DIR, f"{name}.txt")

        if os.path.exists(output_path):
            print(f"Saltando {filename}, ya existe la transcripción.")
            continue

        print(f"Procesando: {filename}...")
        start_time = time.time()
        
        try:
            # Transcribe con beam_size=5 para mejor calidad
            segments, info = model.transcribe(file_path, beam_size=5)
            
            print(f"  Idioma detectado: {info.language} con probabilidad {info.language_probability:.2f}")
            print(f"  Duración: {info.duration:.2f} segundos")

            with open(output_path, "w", encoding="utf-8") as f:
                # Escribimos cabecera con info
                f.write(f"Archivo: {filename}\n")
                f.write(f"Duración: {time.strftime('%H:%M:%S', time.gmtime(info.duration))}\n")
                f.write("-" * 40 + "\n\n")

                for segment in segments:
                    # Formato: [00:00:00 -> 00:00:05] Texto
                    start = time.strftime('%H:%M:%S', time.gmtime(segment.start))
                    end = time.strftime('%H:%M:%S', time.gmtime(segment.end))
                    line = f"[{start} -> {end}] {segment.text}\n"
                    f.write(line)
                    # imprimir en consola para ver progreso
                    print(line.strip()) 
            
            elapsed = time.time() - start_time
            print(f"Completado en {elapsed:.2f} segundos.")

        except Exception as e:
            print(f"Error procesando {filename}: {e}")

    print("¡Todo listo!")

if __name__ == "__main__":
    # Asegurar que existan los directorios
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
