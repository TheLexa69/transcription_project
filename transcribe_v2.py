import os
import glob
import time
import math
import concurrent.futures
import ffmpeg
from faster_whisper import WhisperModel

# Configuración
MODEL_SIZE = "large-v3"
DEVICE = "auto" # Cambiar a "cuda" o "cpu" según el entorno
COMPUTE_TYPE = "int8" # Cambiar a "float16" si tienes GPU dedicada
NUM_THREADS = 2          # Número de hilos concurrentes sugerido por defecto. Ajustar según RAM.
CHUNK_LENGTH_MIN = 10    # Duración de cada fragmento (en minutos)
OVERLAP_SEC = 2          # Solapamiento entre fragmentos para no cortar palabras (en segundos)

INPUT_DIR = "input"
OUTPUT_DIR = "output"
TEMP_DIR = "temp_chunks"

def get_audio_duration(file_path):
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except ffmpeg.Error as e:
        print(f"Error detectando duración: {e.stderr.decode()}")
        return 0.0

def split_audio(file_path):
    print(f"Cargando {file_path} para dividir (puede tardar un poco dependiendo del tamaño)...")
    total_duration_sec = get_audio_duration(file_path)
    if total_duration_sec == 0:
        return [], 0
    
    chunk_length_sec = CHUNK_LENGTH_MIN * 60
    overlap_sec = OVERLAP_SEC
    
    chunks = []
    start_sec = 0
    chunk_index = 0
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    while start_sec < total_duration_sec:
        end_sec = min(start_sec + chunk_length_sec + overlap_sec, total_duration_sec)
        duration_to_extract = end_sec - start_sec
        
        chunk_filename = f"{base_name}_chunk_{chunk_index:03d}.wav"
        chunk_path = os.path.join(TEMP_DIR, chunk_filename)
        
        try:
            (
                ffmpeg
                .input(file_path, ss=start_sec, t=duration_to_extract)
                .output(chunk_path, acodec='pcm_s16le', ac=1, ar='16k', loglevel='quiet')
                .overwrite_output()
                .run()
            )
        except ffmpeg.Error as e:
            print(f"Error dividiendo audio en chunk {chunk_index}: {e.stderr.decode() if e.stderr else str(e)}")
            break
        
        chunks.append({
            'index': chunk_index,
            'path': chunk_path,
            'start_offset_ms': start_sec * 1000
        })
        
        start_sec += chunk_length_sec
        chunk_index += 1
        
    print(f"Dividido en {len(chunks)} fragmentos.")
    return chunks, total_duration_sec * 1000

def transcribe_chunk(chunk_info, model_size, device, compute_type):
    chunk_path = chunk_info['path']
    start_offset_s = chunk_info['start_offset_ms'] / 1000.0
    index = chunk_info['index']
    
    print(f"[Hilo {index}] Iniciando {os.path.basename(chunk_path)}")
    
    # Cada hilo carga su propio modelo para evitar conflictos si usamos threads
    # (Faster Whisper admite hilos, pero instanciar un modelo distinto o
    # usar model methods compartidos a veces requiere cuidado. En CPU, 
    # instanciar varios modelos puede multiplicar uso de RAM).
    # Como la inferencia en CTranslate2 bloquea el GIL, se requiere ThreadPool.
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        segments, info = model.transcribe(chunk_path, beam_size=5)
        
        transcribed_lines = []
        for segment in segments:
            # Ajustar los tiempos en función del inicio real del chunk
            adjusted_start = segment.start + start_offset_s
            adjusted_end = segment.end + start_offset_s
            
            start_str = time.strftime('%H:%M:%S', time.gmtime(adjusted_start))
            end_str = time.strftime('%H:%M:%S', time.gmtime(adjusted_end))
            
            line = f"[{start_str} -> {end_str}] {segment.text}\n"
            transcribed_lines.append(line)
            
        print(f"[Hilo {index}] Terminado {os.path.basename(chunk_path)}")
        return index, transcribed_lines
    except Exception as e:
        print(f"[Hilo {index}] Error: {e}")
        return index, []

def process_file(file_path):
    filename = os.path.basename(file_path)
    name, _ = os.path.splitext(filename)
    output_path = os.path.join(OUTPUT_DIR, f"{name}.txt")

    if os.path.exists(output_path):
        print(f"Saltando {filename}, ya existe la transcripción.")
        return

    print(f"--- Procesando V2: {filename} ---")
    start_time = time.time()
    
    chunks, total_duration_ms = split_audio(file_path)
    if not chunks:
        return
        
    all_transcriptions = {}
    
    print(f"Distibuyendo {len(chunks)} trabajos en {NUM_THREADS} hilos...")
    
    # Usamos ThreadPoolExecutor para concurrencia
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = []
        for chunk in chunks:
            futures.append(
                executor.submit(transcribe_chunk, chunk, MODEL_SIZE, DEVICE, COMPUTE_TYPE)
            )
            
        for future in concurrent.futures.as_completed(futures):
            idx, lines = future.result()
            all_transcriptions[idx] = lines
            
    print("Concatenando y guardando transcripciones...")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Archivo: {filename} (Procesado Concurrente V2)\n")
        f.write(f"Duración: {time.strftime('%H:%M:%S', time.gmtime(total_duration_ms/1000))}\n")
        f.write("-" * 40 + "\n\n")
        
        for i in range(len(chunks)):
            lines = all_transcriptions.get(i, [])
            for line in lines:
                f.write(line)
                
    elapsed = time.time() - start_time
    print(f"--- {filename} Completado en {elapsed:.2f} segundos. ---")
    
    # Limpieza
    for chunk in chunks:
        if os.path.exists(chunk['path']):
            os.remove(chunk['path'])

def main():
    extensions = ['*.m4a', '*.mp3', '*.wav', '*.mp4', '*.mkv']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

    if not files:
        print(f"No se encontraron archivos de audio en '{INPUT_DIR}'")
        return

    print(f"Se encontraron {len(files)} archivos para procesar en modo concurrente.")
    
    for file_path in files:
        process_file(file_path)
        
    print("¡Todo listo V2!")

if __name__ == "__main__":
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    main()
