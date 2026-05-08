import os
import sys
import time
import json
import logging
import threading
import queue
import csv
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from sklearn.cluster import AgglomerativeClustering
from logging.handlers import RotatingFileHandler
from pyannote.core import Annotation
import subprocess

# --- CONFIGURAZIONE INIZIALE ---
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"  # limito i thread delle librerie numeriche per evitare errori mkl_malloc
os.environ["MKL_NUM_THREADS"] = "4"

# --- CONFIGURAZIONE INIZIALE ---
# Cartella del progetto = la stessa dove si trova lo script
PROJECT_DIR = Path(__file__).resolve().parent

# Tutti gli output (checkpoint, log, ecc.) andranno qui
OUT_DIR = PROJECT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "process.log"


# --- CONFIGURAZIONE LOG ---
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# File log con rotazione: max 10MB, 10 file di backup
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=10, encoding="utf-8")
file_handler.setFormatter(log_formatter)

# Console log (stampa a video)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Logger principale
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [file_handler, console_handler]


# --- UTILS ---
def seconds_to_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"

# --- CLASSI PER GESTIONE THREAD E GUI ---

class TranscriptionApp(tk.Tk):
    """
    GUI principale con gestione threading, barra progressi, scelta modello e token.
    """

    def __init__(self):
        super().__init__()
        self.title("Trascrizione e Diarizzazione Audio/Video")
        self.geometry("650x400")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variabili
        self.HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "TOKEN_NON_TROVATO")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_choice = tk.StringVar(value="medium")
        self.video_path = tk.StringVar()
        self.output_format = tk.StringVar(value="txt")
        self.num_speakers = 2  # fisso
        self.is_running = False
        self.cancel_requested = False


        # Coda messaggi da thread
        self.queue = queue.Queue()
        self._whisper_models = {}

        self.create_widgets()
        self.after(200, self.process_queue)  # loop per gestione messaggi

        # Mostra messaggio dispositivo e token mancante
        self.show_start_info()

    def create_widgets(self):
        # Frame per selezione file video
        frame_file = ttk.LabelFrame(self, text="1. Seleziona video")
        frame_file.pack(fill="x", padx=10, pady=5)

        entry_file = ttk.Entry(frame_file, textvariable=self.video_path)
        entry_file.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        btn_browse = ttk.Button(frame_file, text="Sfoglia", command=self.browse_video)
        btn_browse.pack(side="right", padx=5, pady=5)

        # Frame per token HuggingFace
        frame_token = ttk.LabelFrame(self, text="2. Token HuggingFace")
        frame_token.pack(fill="x", padx=10, pady=5)
        self.entry_token = ttk.Entry(frame_token, show="*", width=50)
        self.entry_token.pack(side="left", padx=5, pady=5)
        if self.HF_TOKEN != "TOKEN_NON_TROVATO":
            self.entry_token.insert(0, self.HF_TOKEN)
            self.entry_token.config(state="readonly")
        btn_token = ttk.Button(frame_token, text="Aggiorna Token", command=self.update_token)
        btn_token.pack(side="right", padx=5, pady=5)

        # Frame modello Whisper e info device
        frame_model = ttk.LabelFrame(self, text="3. Seleziona modello Whisper")
        frame_model.pack(fill="x", padx=10, pady=5)

        label_device = ttk.Label(frame_model, text=f"Dispositivo rilevato: {self.device.upper()}")
        label_device.pack(anchor="w", padx=5)

        label_hint = ttk.Label(frame_model, text="Consiglio: per CPU usare modelli leggeri (tiny, base, small).")
        label_hint.pack(anchor="w", padx=5)

        options = ["tiny", "base", "small", "medium", "large-v1", "large-v3-turbo"]
        for opt in options:
            ttk.Radiobutton(frame_model, text=opt, variable=self.model_choice, value=opt).pack(side="left", padx=5)

        # Frame output format
        frame_output = ttk.LabelFrame(self, text="4. Formato output")
        frame_output.pack(fill="x", padx=10, pady=5)
        formats = [("Testo (.txt)", "txt"), ("Sottotitoli (.srt)", "srt"), ("JSON (.json)", "json"), ("CSV (.csv)", "csv")]
        for txt, val in formats:
            ttk.Radiobutton(frame_output, text=txt, variable=self.output_format, value=val).pack(side="left", padx=5)

        # Label numero speaker (fisso a 2)
        label_speaker = ttk.Label(self, text=f"Numero di speaker previsto: {self.num_speakers}")
        label_speaker.pack(pady=5)

        # Frame progressi
        frame_progress = ttk.LabelFrame(self, text="5. Progresso")
        frame_progress.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_progress, text="Diarizzazione").pack(anchor="w")
        self.progress_diar = ttk.Progressbar(frame_progress, orient="horizontal", length=600, mode="determinate")
        self.progress_diar.pack(pady=2)

        ttk.Label(frame_progress, text="Trascrizione").pack(anchor="w")
        self.progress_transc = ttk.Progressbar(frame_progress, orient="horizontal", length=600, mode="determinate")
        self.progress_transc.pack(pady=2)

        # Label tempo stimato e stato
        self.label_status = ttk.Label(self, text="Pronto")
        self.label_status.pack(pady=5)

        # Frame pulsanti start/cancel
        frame_buttons = ttk.Frame(self)
        frame_buttons.pack(pady=10)

        self.btn_start = ttk.Button(frame_buttons, text="Avvia Trascrizione", command=self.start_processing)
        self.btn_start.pack(side="left", padx=10)

        self.btn_cancel = ttk.Button(frame_buttons, text="Annulla", command=self.cancel_processing, state="disabled")
        self.btn_cancel.pack(side="left", padx=10)
        
    def on_closing(self):
        # Gestisce la chiusura della finestra principale.
        # Se un processo è in corso, chiede conferma all'utente.
        # Se confermato, imposta `cancel_requested = True` per fermare i thread.
        # Attende un attimo che i thread in background si interrompano.
        # Libera la GPU se presente.
        # Chiude in sicurezza la GUI.

        if self.is_running:
            # Mostra un popup di conferma
            if messagebox.askokcancel("Chiudi", "Il processo è in esecuzione. Vuoi davvero uscire?"):
                # Richiede ai thread in background di interrompersi
                self.cancel_requested = True
                # Aggiorna la label di stato così l'utente sa che la chiusura è in corso
                self.label_status.config(text="Chiusura in corso, attendere...")
                self.update()  # forza refresh della GUI

                # Piccola attesa per permettere ai thread di intercettare cancel_requested
                time.sleep(1)

                # Se il modello usa GPU, libera la VRAM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("VRAM liberata durante la chiusura")

                try:
                    # Chiudi la finestra
                    self.destroy()
                except Exception as e:
                    logger.warning(f"Errore nella chiusura GUI: {e}")
        else:
            # Nessun processo attivo -> chiusura immediata
            self.destroy()




    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.mkv;*.avi;*.mov")])
        if path:
            self.video_path.set(path)

    def update_token(self):
        new_token = self.entry_token.get().strip()
        if not new_token or new_token == "TOKEN_NON_TROVATO":
            messagebox.showerror("Errore", "Inserisci un token HuggingFace valido!")
            return
        self.HF_TOKEN = new_token
        os.environ["HUGGINGFACE_TOKEN"] = new_token
        self.entry_token.config(state="readonly")
        messagebox.showinfo("Token aggiornato", "Token HuggingFace aggiornato correttamente.")

    def show_start_info(self):
        # Mostra messaggio dispositivo e token
        if self.HF_TOKEN == "TOKEN_NON_TROVATO":
            messagebox.showwarning("Token mancante", "Token HuggingFace non trovato! Inseriscilo nel campo apposito.")
        msg = f"Dispositivo rilevato: {self.device.upper()}\n"
        msg += "Scegli il modello Whisper in base al dispositivo.\n"
        msg += "Per sola CPU si consiglia modelli più leggeri (tiny, base, small)."
        messagebox.showinfo("Info dispositivo e modello", msg)
    def start_processing(self):
        """
        Avvia la pipeline di trascrizione/diarizzazione.
        - Controlla che non ci sia già un processo in corso.
        - Verifica che siano selezionati video e token HuggingFace.
        - Reset dei checkpoint e dei risultati.
        - Avvia il workflow in un thread separato.
        """
        if self.is_running:
            messagebox.showwarning("Attenzione", "Il processo è già in esecuzione.")
            return
        if not self.video_path.get():
            messagebox.showerror("Errore", "Seleziona un file video valido.")
            return
        if self.HF_TOKEN == "TOKEN_NON_TROVATO":
            messagebox.showerror("Errore", "Inserisci un token HuggingFace valido prima di procedere.")
            return

        # Stato GUI
        self.is_running = True
        self.cancel_requested = False
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.label_status.config(text="Inizializzazione...")

        # Avvia il workflow in un thread separato
        threading.Thread(target=process_workflow_thread, args=(self,), daemon=False).start()


    def cancel_processing(self):
        if self.is_running:
            self.cancel_requested = True
            self.label_status.config(text="Annullamento in corso, attendere...")

    def get_whisper_model(self, model_name):
        """Return cached WhisperModel for the given name/device, creating and warming it if needed."""
        key = (model_name, self.device)
        cached = self._whisper_models.get(key)
        if cached is None:
            logger.info(f"Caricamento modello Whisper {model_name} su {self.device}")
            model = WhisperModel(
                model_name,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            warmup_model(model, self.device)
            cached = {"model": model}
            self._whisper_models[key] = cached
        return cached["model"]

    def process_queue(self):
        """
        Gestione messaggi dalla coda inviata dal thread di lavoro per aggiornare GUI.
        """
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg["type"] == "progress_diar":
                    self.progress_diar['value'] = msg["value"]
                    self.label_status.config(text=msg.get("status", self.label_status['text']))
                elif msg["type"] == "progress_transc":
                    self.progress_transc['value'] = msg["value"]
                    self.label_status.config(text=msg.get("status", self.label_status['text']))
                elif msg["type"] == "done":
                    self.is_running = False
                    self.btn_start.config(state="normal")
                    self.btn_cancel.config(state="disabled")
                    self.label_status.config(text=msg.get("status", "Processo terminato."))
                    logger.info("GUI: processo terminato correttamente")
                elif msg["type"] == "error":
                    self.is_running = False
                    self.btn_start.config(state="normal")
                    self.btn_cancel.config(state="disabled")
                    self.label_status.config(text="Errore durante l'esecuzione")
                    logger.error(f"Errore ricevuto in coda: {msg['message']}")
                    messagebox.showerror("Errore", msg["message"])
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Eccezione in process_queue: {e}", exc_info=True)
        self.after(200, self.process_queue)


# --- BLACKLIST (uguale a prima, lista frasi da ignorare) ---
BLACKLIST_IT = [
    "Sottotitoli creati dalla comunità Amara.org",
    "Sottotitoli di Sottotitoli di Amara.org",
    "Sottotitoli e revisione al canale di Amara.org",
    "Sottotitoli e revisione a cura di Amara.org",
    "Sottotitoli e revisione a cura di QTSS",
    "Sottotitoli e revisione a cura di QTSS.",
    "Sottotitoli a cura di QTSS",
]

# --- FUNZIONI UTILI ---

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("ffmpeg disponibile")
        return True
    except Exception:
        logger.error("ffmpeg non trovato")
        return False

def extract_audio(video_path, audio_path):
    logger.info("Estrazione audio dal video...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", audio_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info(f"Audio salvato in: {audio_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Errore ffmpeg: {e.stderr.decode(errors='ignore')}")
        raise


def format_output_txt(segments, out_path):
    """
    Salva la trascrizione in formato TXT.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(format_segment(seg) + "\n")
    return out_path

def format_output_srt(segments, out_path):
    """
    Salva la trascrizione in formato SRT (sottotitoli).
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seconds_to_srt(seg["start"])
            end = seconds_to_srt(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['speaker']}: {seg['text']}\n\n")
    return out_path



def seconds_to_srt(seconds):
    ms = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def format_output_json(segments, out_path):
    save_json(out_path, segments)

def format_output_csv(segments, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "speaker", "text"])
        for seg in segments:
            writer.writerow([seg['start'], seg['end'], seg['speaker'], seg['text']])

def filter_blacklist_segments(segments):
    """
    Filtra i segmenti che contengono frasi della blacklist.
    Restituisce (lista_filtrata, numero_segmenti_filtrati).
    """
    filtered = []
    filtered_count = 0
    for seg in segments:
        text_lower = seg['text'].lower()
        if any(bl_phrase.lower() in text_lower for bl_phrase in BLACKLIST_IT):
            filtered_count += 1
            continue
        filtered.append(seg)
    return filtered, filtered_count


# --- GESTIONE GPU, MODELLO, WARMUP ---
def warmup_model(model, device):
    logger.info("Warming-up modello...")
    try:
        dummy = np.random.randn(16000).astype(np.float32) * 0.001
        result, _ = model.transcribe(dummy, language="it")
        result = list(result)  # converto il generatore in lista
        _ = " ".join([seg.text for seg in result])  # forzo l'iterazione
        logger.info("Warming-up completato")
    except Exception as e:
        logger.error(f"Errore durante warmup: {e}")



def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("VRAM liberata")

# --- TRANSCRIZIONE E DIARIZZAZIONE ---


def diarize_audio(audio_file, hf_token, queue, cancel_flag):
    try:
        # carico la pipeline ufficiale
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

    except Exception as e:
        queue.put({"type": "error", "message": f"Errore inizializzazione diarizzazione: {e}"})
        return None

    diar_result = None
    try:
        for i in range(10):
            if cancel_flag():
                queue.put({"type": "error", "message": "Processo annullato durante diarizzazione."})
                return None
            time.sleep(0.5)
            queue.put({"type": "progress_diar", "value": (i + 1) * 10,
                       "status": f"Diarizzazione... {((i + 1) * 10)}%"})

        # qui uso la pipeline direttamente
        diar_result = pipeline(audio_file)

        # (opzionale) forzare 2 speaker
        diar_result = force_two_speakers(diar_result)

        queue.put({"type": "progress_diar", "value": 100, "status": "Diarizzazione completata"})
    except Exception as e:
        queue.put({"type": "error", "message": f"Errore durante diarizzazione: {e}"})
        return None

    return diar_result

def force_two_speakers(annotation: Annotation) -> Annotation:
    """
    Applica clustering Agglomerativo per forzare 2 speaker su segmenti annotati.
    """
    segments = list(annotation.itersegments())
    durations = [segment.end - segment.start for segment in segments]
    centers = [segment.start + duration / 2 for segment, duration in zip(segments, durations)]

    X = np.array(centers).reshape(-1, 1)

    clustering = AgglomerativeClustering(n_clusters=2)
    labels = clustering.fit_predict(X)

    new_annotation = Annotation()
    for segment, label in zip(segments, labels):
        new_annotation[segment] = f"SPEAKER_{label+1}"
    return new_annotation
    
def prepare_segments(diar_result):
    """
    Trasforma l'output della diarizzazione in lista segmenti
    con start, end e speaker senza accorpare quelli corti.
    """
    segments = []
    for turn, _, speaker in diar_result.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    segments.sort(key=lambda x: x["start"])
    return segments


def transcribe_segments(segments, audio_path, model, device,
                        queue, cancel_flag):

    """
    Trascrizione parallela segmenti con WhisperModel faster_whisper.
    Gestione batching e warmup.
    """
    # Caricamento audio intero (pydub) per estrazione segmenti
    audio = AudioSegment.from_file(audio_path)
    results_out = {}
    total_segments = len(segments)

    

    # Executor per CPU multiprocesso
    if device == "cpu":
        executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
    else:
        executor = None  # GPU single worker

    # --- Funzione interna per trascrivere un singolo segmento ---
    def transcribe_task(idx, seg):
        """
        Trascrive un singolo segmento audio usando il modello Whisper.

        - idx: indice del segmento
        - seg: dizionario con start, end, speaker
        - Ritorna: (idx, risultato) oppure (idx, None) in caso di errore o segmento vuoto
        """
        if cancel_flag():
            return idx, None


        start, end, speaker = seg["start"], seg["end"], seg["speaker"]

        # Controllo segmento vuoto o non valido
        if end <= start:
            logger.warning(f"Segmento {idx} scartato: start={start}, end={end}")
            return idx, None

        try:
            # Estrai audio del segmento in millisecondi con Pydub
            segment_audio = audio[start * 1000:end * 1000]
            audio_np = segment_audio.get_array_of_samples()

            
            # Normalizzazione per 16-bit PCM (pydub esporta in 16-bit)
            audio_float = np.array(audio_np).astype(np.float32) / 32768.0
            # Se in futuro usi audio a 24/32-bit, sostituire con sample_width per normalizzazione dinamica

            # Trascrizione con faster-whisper
            result, _ = model.transcribe(audio_float, beam_size=5, language="it")
            result = list(result)  # converto in lista
            text = " ".join([segm.text for segm in result])

            # Debug: stampa i primi 50 caratteri del testo
            print(f"[DEBUG] Segmento {idx}: {text[:50]}...")

            return idx, {"start": start, "end": end, "speaker": speaker, "text": text}

        except Exception as e:
            logger.error(f"Errore trascrizione segmento {idx}: {e}")
            return idx, None

    # --- Esecuzione trascrizione ---
    if device == "cpu":
        futures = []
        for idx, segment in enumerate(segments):
            if str(idx) in results_out:
                        # Segmento già trascritto (checkpoint)
                progress = 100 * len(results_out) / total_segments
                queue.put({
                    "type": "progress_transc",
                    "value": progress,
                    "status": f"Segmenti trascritti: {len(results_out)}/{total_segments}"
                })
                continue
            futures.append(executor.submit(transcribe_task, idx, segment))
        for future in as_completed(futures):
            if cancel_flag():
                break
            try:
                result = future.result()
                if result is None:
                    continue  # niente da salvare
                idx, res = result
                if res is not None:
                    results_out[str(idx)] = res
            except Exception as e:
                logger.error(f"Errore futuro CPU: {e}")
                continue

            progress = 100 * len(results_out) / total_segments
            queue.put({
                "type": "progress_transc",
                "value": progress,
                "status": f"Segmenti trascritti: {len(results_out)}/{total_segments}"
            })

    else:
        # GPU: esegui sequenzialmente una sola volta su TUTTI i segmenti
        for idx, segment in enumerate(segments):
            if cancel_flag():
                break
            res = transcribe_task(idx, segment)
            if res is None:
                continue
            idx, seg_res = res
            if seg_res is not None:
                results_out[str(idx)] = seg_res



            # aggiorna progress ogni segmento
            progress = 100 * len(results_out) / total_segments
            queue.put({
                "type": "progress_transc",
                "value": progress,
                "status": f"Segmenti trascritti: {len(results_out)}/{total_segments}"
            })

    # Chiusura executor CPU per rilasciare risorse
    if executor:
        executor.shutdown(wait=True)
        logger.info("Executor CPU chiuso correttamente")

    return results_out

def normalize_segments_by_speaker(segments):
    """
    Rinomina gli speaker in formato SPEAKER 00 mantenendo la suddivisione originale.
    """
    normalized = []
    for seg in segments or []:
        speaker_label = seg.get('speaker', 'SPEAKER_1')
        try:
            speaker_idx = int(str(speaker_label).split('_')[-1])
        except (ValueError, TypeError):
            speaker_idx = 0
        normalized.append({
            'start': seg.get('start', 0.0),
            'end': seg.get('end', 0.0),
            'speaker': f"SPEAKER {speaker_idx:02d}",
            'text': seg.get('text', '')
        })
    return normalized


def format_segment(seg):
    """
    Ritorna un segmento formattato con timestamp e speaker.
    Formato: [hh:mm:ss - hh:mm:ss] SPEAKER 00: testo
    """
    start = seconds_to_hhmmss(seg["start"])
    end = seconds_to_hhmmss(seg["end"])
    return f"[{start} - {end}] {seg['speaker']}: {seg['text']}"


def create_final_output(results_dict, output_format, out_base_path):
    logger.info(f"Creazione output finale in formato {output_format}...")
    logger.info(f"Percorso base output: {out_base_path}")

    try:
        segments = [results_dict[k] for k in sorted(results_dict.keys(), key=lambda x: int(x))]
        logger.info(f"Segmenti da salvare: {len(segments)}")

        segments, filtered_count = filter_blacklist_segments(segments)
        logger.info(f"Segmenti filtrati: {filtered_count}, rimasti: {len(segments)}")

        normalized_segments = normalize_segments_by_speaker(segments)
        logger.info(f"Segmenti normalizzati: {len(normalized_segments)}")

        if output_format == "txt":
            out_file = f"{out_base_path}.txt"
            format_output_txt(normalized_segments, out_file)
        elif output_format == "srt":
            out_file = f"{out_base_path}.srt"
            format_output_srt(normalized_segments, out_file)
        elif output_format == "json":
            out_file = f"{out_base_path}.json"
            format_output_json(normalized_segments, out_file)
        elif output_format == "csv":
            out_file = f"{out_base_path}.csv"
            format_output_csv(normalized_segments, out_file)
        else:
            raise ValueError(f"Formato output non supportato: {output_format}")

        logger.info(f"Output finale creato: {out_file}")
 
        print(f"[DEBUG] File creato: {out_file}")

        return filtered_count, out_file

    except Exception as e:
        logger.error(f"Errore in create_final_output: {e}", exc_info=True)
        raise


# --- MAIN WORKFLOW nel thread ---
def process_workflow_thread(app):
    """
    Funzione principale eseguita in thread per gestire pipeline completa:
    Estrazione audio -> diarizzazione -> trascrizione -> output finale
    """
    try:
        app.queue.put({"type": "progress_diar", "value": 0, "status": "Estrazione audio in corso..."})

        # Percorsi input/output
        video_path = Path(app.video_path.get())
        audio_path = OUT_DIR / "audio.wav"  # audio temporaneo sempre in OUT_DIR

        # Estrazione audio
        extract_audio(str(video_path), str(audio_path))
        if app.cancel_requested:
            app.queue.put({"type": "error", "message": "Processo annullato dall'utente."})
            return

        # Diarizzazione
        app.queue.put({"type": "progress_diar", "value": 5, "status": "Inizializzazione diarizzazione..."})
        diar_result = diarize_audio(str(audio_path), app.HF_TOKEN, app.queue, lambda: app.cancel_requested)
        if diar_result is None:
            logger.error("Diarizzazione non riuscita, nessun risultato.")
            return

        segments = prepare_segments(diar_result)
        if not segments:
            logger.error("Nessun segmento rilevato dalla diarizzazione.")
            app.queue.put({"type": "error", "message": "Nessun segmento rilevato dalla diarizzazione."})
            return

        app.queue.put({"type": "progress_diar", "value": 100, "status": "Diarizzazione completata"})
        app.queue.put({"type": "progress_transc", "value": 0, "status": "Avvio trascrizione..."})

        model_name = app.model_choice.get()
        whisper_model = app.get_whisper_model(model_name)
        results = transcribe_segments(
            segments, str(audio_path), whisper_model, app.device,
            app.queue,
            lambda: app.cancel_requested
        )

        # Debug: quanti risultati sono stati raccolti
        print(f"[DEBUG] Risultati raccolti: {len(results)}")

        logger.info(f"Trascrizione terminata. Segmenti ottenuti: {0 if not results else len(results)}")
        print(f"[DEBUG] Risultati trascrizione: {0 if not results else len(results)}")


        if app.cancel_requested:
            app.queue.put({"type": "error", "message": "Processo annullato dall'utente."})
            return

        # Se results è vuoto -> errore esplicito
        if not results or len(results) == 0:
            logger.error("Nessun risultato dalla trascrizione, impossibile creare output finale.")
            print("\nERRORE: Trascrizione vuota, nessun file generato.")
            app.queue.put({"type": "error", "message": "Trascrizione vuota: nessun file generato."})
            return

        # Creazione output finale
        try:
            # Crea base output nella stessa cartella del video  con il nome del video
            out_base = video_path.parent / video_path.stem
            print(f"[DEBUG] Output base: {out_base}")  # <-- Debug per vedere dove salva

            filtered_count, out_file = create_final_output(
                results, app.output_format.get(), out_base
            )

            # Notifica GUI
            app.queue.put({
                "type": "done",
                "status": f"Processo completato. Output: {out_file}",
                "filtered_count": filtered_count
            })

            # Stampa chiara in console (PowerShell)
            print(f"\nProcesso completato con successo!")
            print(f"   File di output generato: {out_file}")
            if filtered_count > 0:
                print(f"   Segmenti filtrati dalla blacklist: {filtered_count}")

        except Exception as e:
            logger.error(f"Errore durante la creazione dell'output finale: {e}", exc_info=True)
            print(f"\nERRORE durante la creazione output: {e}")
            app.queue.put({"type": "error", "message": f"Errore output: {e}"})

    except Exception as e:
        logger.error(f"Errore inatteso nel workflow: {e}", exc_info=True)
        print(f"\nERRORE inatteso: {e}")
        app.queue.put({"type": "error", "message": f"Errore inatteso: {e}"})


def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


## ---- punto ingresso----
def main():
    if not check_ffmpeg():
        print("ffmpeg non trovato! Installa ffmpeg e riprova.")
        return
    app = TranscriptionApp()
    app.mainloop()
    logger.info("mainloop terminato (GUI chiusa)")
    print("GUI chiusa")

if __name__ == "__main__":
    main()
