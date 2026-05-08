# -*- coding: utf-8 -*-
"""
Confronto fra trascrizioni (riferimento vs automatica) con metriche multiple:
- RapidFuzz / Difflib riga ↔ riga
- RapidFuzz / Difflib / BLEU riga ↔ join di righe automatiche
- RapidFuzz / Difflib / BLEU join di righe di riferimento ↔ riga automatica
- BLEU su tutti i confronti (Base, Join Auto, Join Riferimento)


Output:
- risultati_confronto_con_bleu.xlsx
- risultati_confronto_con_bleu.csv
"""

# ==============================
#           IMPORT
# ==============================
import re
import pandas as pd
from difflib import SequenceMatcher
from rapidfuzz.fuzz import token_set_ratio
from unidecode import unidecode

# --- BLEU: import NLTK con smoothing ---
# Richiede `pip install nltk`
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Margine temporale massimo per il matching (in secondi)
TIME_MARGIN_SECONDS = 120

# ==============================
#            UTILS
# ==============================
def normalize_text(text: str) -> str:
    """
    Normalizza un testo per i confronti lessicali:
    - minuscole
    - rimozione accenti (unidecode)
    - rimozione punteggiatura
    - collassa spazi multipli
    """
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==============================
#       PARSING FILE TESTO
# ==============================
def parse_file1(path: str, debug=False):
    """
    Parsing file RIFERIMENTO con formato: "mm:ss SPEAKER: testo"
    """
    entries, discarded = [], 0
    pattern = re.compile(r"(\d{1,2}):(\d{2})\s+[^:]+:\s+(.+)")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            s = line.strip()
            m = pattern.match(s)
            if not m:
                discarded += 1
                if debug:
                    print(f"[DEBUG][FILE1] Riga {line_num} scartata (formato): {s}")
                continue
            mm, ss, text = m.groups()
            try:
                t = int(mm) * 60 + int(ss)
            except ValueError:
                discarded += 1
                if debug:
                    print(f"[DEBUG][FILE1] Riga {line_num} scartata (timestamp): {s}")
                continue
            entries.append({
                "line_num": line_num,
                "time": t,
                "text": text.strip(),
                "norm_text": normalize_text(text),
            })
    return entries, discarded


def parse_file2(path: str, debug=False):
    """
    Parsing file AUTOMATICO con formato:
    "[hh:mm:ss - hh:mm:ss] SPEAKER: testo"
    """
    entries, discarded = [], 0
    pattern = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\s*-\s*\d{2}:\d{2}:\d{2}\]\s+[^:]+:\s+(.+)")
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            s = line.strip()
            m = pattern.match(s)
            if not m:
                discarded += 1
                if debug:
                    print(f"[DEBUG][FILE2] Riga {line_num} scartata (formato): {s}")
                continue
            hh, mm, ss, text = m.groups()
            try:
                t = int(hh) * 3600 + int(mm) * 60 + int(ss)
            except ValueError:
                discarded += 1
                if debug:
                    print(f"[DEBUG][FILE2] Riga {line_num} scartata (timestamp): {s}")
                continue
            entries.append({
                "line_num": line_num,
                "time": t,
                "text": text.strip(),
                "norm_text": normalize_text(text),
            })
    return entries, discarded


# ==============================
#      METRICA: BLEU (NUOVA)
# ==============================
def bleu_similarity(ref_text: str, cand_text: str) -> float:
    """
    Calcola la similarità BLEU (%) tra due testi, con smoothing.
    Restituisce un valore percentuale [0..100].
    """
    ref_tokens = ref_text.split()
    cand_tokens = cand_text.split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    smooth = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)
    return round(score * 100.0, 2)


# ==============================
#          CONFRONTI
# ==============================
def compare_with_windows(file1_entries, file2_entries, window_size=3, debug=True):
    """
    Confronto BASE riga ↔ riga (RapidFuzz, Difflib, BLEU).
    """
    results = []
    for idx1, e1 in enumerate(file1_entries):
        start = max(0, idx1 - window_size + 1)
        end = min(len(file2_entries), idx1 + window_size)
        for idx2 in range(start, end):
            e2 = file2_entries[idx2]
            dt = abs(e1["time"] - e2["time"])
            if dt > TIME_MARGIN_SECONDS:
                if debug:
                    print(f"[DEBUG][BASE] Salto F1#{e1['line_num']} vs F2#{e2['line_num']} (delta {dt}s > {TIME_MARGIN_SECONDS}s)")
                continue
            s_rapid = token_set_ratio(e1["norm_text"], e2["norm_text"])
            s_diff = SequenceMatcher(None, e1["norm_text"], e2["norm_text"]).ratio() * 100.0
            s_bleu = bleu_similarity(e1["norm_text"], e2["norm_text"])
            if debug:
                print(
                    f"[DEBUG][BASE] Match F1#{e1['line_num']} ({e1['text'][:40]}) -> F2#{e2['line_num']} ({e2['text'][:40]}) | dt={dt}s | "
                    f"Rapid={s_rapid:.2f} Diff={s_diff:.2f} Bleu={s_bleu:.2f}"
                )
            results.append({
                "file1_line": e1["line_num"], "file1_text": e1["text"],
                "file2_line": e2["line_num"], "file2_text": e2["text"],
                "similarity_rapid": round(s_rapid, 2),
                "similarity_diff": round(s_diff, 2),
                "similarity_bleu": round(s_bleu, 2)
            })
    return results


def compare_with_joins_sliding(file1_entries, file2_entries, join_len_auto=3, join_shifts=2, debug=False):
    """
    Confronto riga ↔ join di righe AUTOMATICHE (RapidFuzz, Difflib, BLEU).
    """
    results = []
    n_auto = len(file2_entries)
    for i, e1 in enumerate(file1_entries):
        best = {"rapid": -1, "diff": -1, "bleu": -1,
                "dt": float("inf"), "text": "", "start": None, "end": None}
        for s in range(max(1, join_shifts)):
            start = max(0, i - 1 + s)
            end = min(n_auto, start + max(1, join_len_auto))
            if end <= start:
                continue
            win = file2_entries[start:end]
            dt_values = [abs(e1["time"] - e["time"]) for e in win]
            dt_min = min(dt_values)
            if dt_min > TIME_MARGIN_SECONDS:
                if debug:
                    print(f"[DEBUG][JOIN-AUTO] Salto blocco auto {win[0]['line_num']}-{win[-1]['line_num']} (delta {dt_min}s > {TIME_MARGIN_SECONDS}s)")
                continue
            joined = " ".join(e["text"] for e in win)
            joined_norm = normalize_text(joined)
            s_rapid = token_set_ratio(e1["norm_text"], joined_norm)
            s_diff = SequenceMatcher(None, e1["norm_text"], joined_norm).ratio() * 100.0
            s_bleu = bleu_similarity(e1["norm_text"], joined_norm)
            def better(a, b, dta, dtb): return (a > b) or (abs(a - b) < 1e-9 and dta < dtb)
            if better(s_rapid, best["rapid"], dt_min, best["dt"]):
                best.update({"rapid": s_rapid, "diff": s_diff, "bleu": s_bleu,
                             "dt": dt_min, "text": joined,
                             "start": win[0]["line_num"], "end": win[-1]["line_num"]})
                if debug:
                    print(
                        f"[DEBUG][JOIN-AUTO] Miglior match F1#{e1['line_num']} con blocco {win[0]['line_num']}->{win[-1]['line_num']} | dt={dt_min}s | "
                        f"Rapid={s_rapid:.2f} Diff={s_diff:.2f} Bleu={s_bleu:.2f}"
                    )
        results.append({
            "file1_line": e1["line_num"], "file1_text": e1["text"],
            "join_auto_start": best["start"], "join_auto_end": best["end"], "join_auto_text": best["text"],
            "similarity_join_rapid": round(max(0, best["rapid"]), 2),
            "similarity_join_diff": round(max(0, best["diff"]), 2),
            "similarity_join_bleu": round(max(0, best["bleu"]), 2)
        })
    return results


def compare_join_reference_vs_auto(file1_entries, file2_entries, join_len_reference=2, debug=False):
    """
    Confronto JOIN RIFERIMENTO ↔ riga AUTOMATICA (RapidFuzz, Difflib, BLEU).
    """
    results = []
    n_h = len(file1_entries)
    for i in range(n_h - join_len_reference + 1):
        block = file1_entries[i:i + join_len_reference]
        joined = " ".join(e["text"] for e in block)
        joined_norm = normalize_text(joined)
        t_avg = sum(e["time"] for e in block) // join_len_reference
        best = {"rapid": -1, "diff": -1, "bleu": -1,
                "dt": float("inf"), "text": "", "line": None}
        for e2 in file2_entries:
            dt = abs(t_avg - e2["time"])
            if dt > TIME_MARGIN_SECONDS:
                if debug:
                    print(
                        f"[DEBUG][JOIN-RIFERIMENTO] Salto blocco riferimento {block[0]['line_num']}->{block[-1]['line_num']} con F2#{e2['line_num']} (delta {dt}s > {TIME_MARGIN_SECONDS}s)"
                    )
                continue
            s_rapid = token_set_ratio(joined_norm, e2["norm_text"])
            s_diff = SequenceMatcher(None, joined_norm, e2["norm_text"]).ratio() * 100.0
            s_bleu = bleu_similarity(joined_norm, e2["norm_text"])
            def better(a, b, dta, dtb): return (a > b) or (abs(a - b) < 1e-9 and dta < dtb)
            if better(s_rapid, best["rapid"], dt, best["dt"]):
                best.update({"rapid": s_rapid, "diff": s_diff, "bleu": s_bleu,
                             "dt": dt, "text": e2["text"], "line": e2["line_num"]})
                if debug:
                    print(
                        f"[DEBUG][JOIN-RIFERIMENTO] Miglior match blocco riferimento {block[0]['line_num']}->{block[-1]['line_num']} con F2#{e2['line_num']} | dt={dt}s | "
                        f"Rapid={s_rapid:.2f} Diff={s_diff:.2f} Bleu={s_bleu:.2f}"
                    )
        results.append({
            "reference_start_line": block[0]["line_num"], "reference_end_line": block[-1]["line_num"],
            "reference_join_text": joined,
            "auto_line": best["line"], "auto_text": best["text"],
            "similarity_hjoin_rapid": round(max(0, best["rapid"]), 2),
            "similarity_hjoin_diff": round(max(0, best["diff"]), 2),
            "similarity_hjoin_bleu": round(max(0, best["bleu"]), 2)
        })
    return results


# ==============================
#            MEDIE
# ==============================
def compute_best_averages(results, min_percent=15.0):
    """Media migliori match BASE (RapidFuzz, Difflib, BLEU)."""
    best_r, best_d, best_b = {}, {}, {}
    for r in results:
        if r["similarity_rapid"] >= min_percent:
            best_r[r["file1_line"]] = max(best_r.get(r["file1_line"], 0), r["similarity_rapid"])
        if r["similarity_diff"] >= min_percent:
            best_d[r["file1_line"]] = max(best_d.get(r["file1_line"], 0), r["similarity_diff"])
        if r["similarity_bleu"] >= min_percent:
            best_b[r["file1_line"]] = max(best_b.get(r["file1_line"], 0), r["similarity_bleu"])
    avg_r = sum(best_r.values()) / len(best_r) if best_r else 0.0
    avg_d = sum(best_d.values()) / len(best_d) if best_d else 0.0
    avg_b = sum(best_b.values()) / len(best_b) if best_b else 0.0
    return avg_r, avg_d, avg_b


def compute_join_averages(results, rapid_key, diff_key, bleu_key, min_percent=15.0):
    """Media match JOIN (RapidFuzz, Difflib, BLEU)."""
    rapids = [r.get(rapid_key, 0) for r in results if r.get(rapid_key, 0) >= min_percent]
    diffs = [r.get(diff_key, 0) for r in results if r.get(diff_key, 0) >= min_percent]
    bleus = [r.get(bleu_key, 0) for r in results if r.get(bleu_key, 0) >= min_percent]
    avg_r = sum(rapids) / len(rapids) if rapids else 0.0
    avg_d = sum(diffs) / len(diffs) if diffs else 0.0
    avg_b = sum(bleus) / len(bleus) if bleus else 0.0
    return avg_r, avg_d, avg_b


# ==============================
#            EXPORT
# ==============================
def save_outputs_combined(res_base, res_auto, res_reference):
    """
    Esporta un unico XLSX e un CSV con tutte le sezioni affiancate.
    Blocchi: BASE | RIGA→JOIN AUTO | JOIN RIFERIMENTO→RIGA AUTO
    """
    df_base = pd.DataFrame(res_base)
    df_auto = pd.DataFrame(res_auto)
    df_reference = pd.DataFrame(res_reference)
    max_len = max(len(df_base), len(df_auto), len(df_reference), 1)
    dfs = [df.reindex(range(max_len)) for df in [df_base, df_auto, df_reference]]
    df_base, df_auto, df_reference = dfs
    sep = pd.DataFrame({"": [""] * max_len, " ": [""] * max_len})
    df_combined = pd.concat([df_base, sep, df_auto, sep, df_reference], axis=1)
    out_xlsx = "risultati_confronto_con_bleu.xlsx"
    out_csv = "risultati_confronto_con_bleu.csv"
    df_combined.to_excel(out_xlsx, index=False)
    df_combined.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ File salvati:\n - {out_xlsx}\n - {out_csv}")
# ==============================
#            GUI
# ==============================
import argparse
import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Progressbar, Style

# Proviamo a usare ttkbootstrap per un look moderno, ma senza dipendenza obbligatoria
try:
    import ttkbootstrap as tb
    _HAS_TTKBOOTSTRAP = True
except Exception:
    _HAS_TTKBOOTSTRAP = False


def get_startup_params(default_window=3, default_join_auto=3, default_join_reference=2, default_debug=False):
    """
    Mostra una finestra iniziale per scegliere i parametri opzionali:
    - window (finestra confronto base)
    - join_len_auto (lunghezza join automatico)
    - join_len_reference (lunghezza join riferimento)
    - debug (flag)
    Restituisce: (window, join_len_auto, join_len_reference, debug)
    """
    if _HAS_TTKBOOTSTRAP:
        root = tb.Window(themename="flatly")
    else:
        root = tk.Tk()

    root.title("Parametri di confronto")
    root.geometry("420x300")
    root.resizable(False, False)

    container = ttk.Frame(root, padding=16)
    container.pack(fill="both", expand=True)

    title = ttk.Label(container, text="Imposta i parametri opzionali", font=("Segoe UI", 12, "bold"))
    title.pack(anchor="w", pady=(0, 10))

    # window
    row1 = ttk.Frame(container); row1.pack(fill="x", pady=6)
    ttk.Label(row1, text="Finestra (righe auto attorno a riferimento):").pack(side="left")
    window_var = tk.StringVar(value=str(default_window))
    window_box = ttk.Combobox(row1, textvariable=window_var, state="readonly", width=6,
                              values=[str(i) for i in range(1, 11)])
    window_box.pack(side="right")

    # join_len_auto
    row2 = ttk.Frame(container); row2.pack(fill="x", pady=6)
    ttk.Label(row2, text="Lunghezza join automatico:").pack(side="left")
    join_auto_var = tk.StringVar(value=str(default_join_auto))
    join_auto_box = ttk.Combobox(row2, textvariable=join_auto_var, state="readonly", width=6,
                                 values=[str(i) for i in range(1, 11)])
    join_auto_box.pack(side="right")

    # join_len_reference
    row3 = ttk.Frame(container); row3.pack(fill="x", pady=6)
    ttk.Label(row3, text="Lunghezza join riferimento:").pack(side="left")
    join_reference_var = tk.StringVar(value=str(default_join_reference))
    join_reference_box = ttk.Combobox(row3, textvariable=join_reference_var, state="readonly", width=6,
                                  values=[str(i) for i in range(1, 11)])
    join_reference_box.pack(side="right")

    # debug
    row4 = ttk.Frame(container); row4.pack(fill="x", pady=6)
    debug_var = tk.BooleanVar(value=default_debug)
    debug_cb = ttk.Checkbutton(row4, text="Abilita debug", variable=debug_var)
    debug_cb.pack(side="left")

    # bottone
    btn_row = ttk.Frame(container); btn_row.pack(fill="x", pady=(16, 0))
    def on_start():
        root.quit()
        root.destroy()
    start_btn = ttk.Button(btn_row, text="Avvia confronto", command=on_start)
    start_btn.pack(side="right")

    root.mainloop()

    try: w = int(window_var.get())
    except: w = default_window
    try: ja = int(join_auto_var.get())
    except: ja = default_join_auto
    try: jh = int(join_reference_var.get())
    except: jh = default_join_reference

    return w, ja, jh, bool(debug_var.get())


def show_summary_popup_full(total_lines,
                            sim_base_r, sim_base_d, sim_base_bleu,
                            sim_auto_r, sim_auto_d, sim_auto_bleu,
                            sim_reference_r, sim_reference_d, sim_reference_bleu,
                            discarded_reference, discarded_auto):
    """
    Popup riepilogo con 9 barre:
    Base (RapidFuzz, Difflib, BLEU)
    Riga→Join Auto (RapidFuzz, Difflib, BLEU)
    Join Riferimento→Riga Auto (RapidFuzz, Difflib, BLEU)
    """
    if _HAS_TTKBOOTSTRAP:
        root = tb.Window(themename="flatly")
    else:
        root = tk.Tk()

    root.title("Riepilogo Confronto")
    root.geometry("780x800")
    root.configure(bg="white")

    style = Style(); style.theme_use("default")
    style.configure("green.Horizontal.TProgressbar", troughcolor="white", background="green")
    style.configure("blue.Horizontal.TProgressbar", troughcolor="white", background="blue")
    style.configure("teal.Horizontal.TProgressbar", troughcolor="white", background="#008080")
    style.configure("orange.Horizontal.TProgressbar", troughcolor="white", background="orange")
    style.configure("purple.Horizontal.TProgressbar", troughcolor="white", background="purple")
    style.configure("cyan.Horizontal.TProgressbar", troughcolor="white", background="cyan")
    style.configure("red.Horizontal.TProgressbar", troughcolor="white", background="red")
    style.configure("brown.Horizontal.TProgressbar", troughcolor="white", background="brown")
    style.configure("pink.Horizontal.TProgressbar", troughcolor="white", background="deeppink")

    header = tk.Frame(root, bg="white"); header.pack(padx=20, pady=10, fill="x")
    tk.Label(header, text="Riepilogo confronto", bg="white", font=("Segoe UI", 14, "bold")).pack(anchor="w")
    tk.Label(header, text=f"Totale righe utili (file riferimento): {total_lines}", bg="white", font=("Segoe UI", 10)).pack(anchor="w")
    tk.Label(header, text=f"Righe scartate — Riferimento: {discarded_reference} | Automatico: {discarded_auto}",
             bg="white", font=("Segoe UI", 10)).pack(anchor="w")

    container = tk.Frame(root, bg="white"); container.pack(pady=4, padx=20, fill="both", expand=True)

    def add_bar(parent, title, value, style_name):
        frame = tk.Frame(parent, bg="white"); frame.pack(pady=6, fill="x")
        tk.Label(frame, text=title, bg="white", font=("Segoe UI", 10)).pack(anchor="w")
        pb = Progressbar(frame, orient="horizontal", length=640,
                         mode="determinate", maximum=100, style=style_name)
        pb.pack(fill="x")
        pb["value"] = float(value)
        tk.Label(frame, text=f"{value:.2f}%", bg="white", font=("Segoe UI", 9)).pack(anchor="e")

    # Base
    add_bar(container, "Base RapidFuzz", sim_base_r, "green.Horizontal.TProgressbar")
    add_bar(container, "Base Difflib", sim_base_d, "blue.Horizontal.TProgressbar")
    add_bar(container, "Base BLEU", sim_base_bleu, "teal.Horizontal.TProgressbar")

    # Riga→Join Auto
    add_bar(container, "Riga→Join Auto RapidFuzz", sim_auto_r, "orange.Horizontal.TProgressbar")
    add_bar(container, "Riga→Join Auto Difflib", sim_auto_d, "purple.Horizontal.TProgressbar")
    add_bar(container, "Riga→Join Auto BLEU", sim_auto_bleu, "cyan.Horizontal.TProgressbar")

    # Join Riferimento→Riga Auto
    add_bar(container, "Join Riferimento→Riga Auto RapidFuzz", sim_reference_r, "red.Horizontal.TProgressbar")
    add_bar(container, "Join Riferimento→Riga Auto Difflib", sim_reference_d, "brown.Horizontal.TProgressbar")
    add_bar(container, "Join Riferimento→Riga Auto BLEU", sim_reference_bleu, "pink.Horizontal.TProgressbar")

    btn_frame = tk.Frame(root, bg="white"); btn_frame.pack(pady=12)
    ttk.Button(btn_frame, text="Chiudi", command=root.destroy).pack()
    root.mainloop()


# ==============================
#            MAIN
# ==============================
def main(file1_path, file2_path, window=3, join_len_auto=3, join_len_reference=2, debug=False):
    # Parsing
    file1_entries, discarded_reference = parse_file1(file1_path, debug)
    file2_entries, discarded_auto = parse_file2(file2_path, debug)

    # Base
    if debug: print("\n[DEBUG] --- CONFRONTO BASE ---")
    res_base = compare_with_windows(file1_entries, file2_entries, window_size=window, debug=bool(debug))
    avg_base_r, avg_base_d, avg_base_b = compute_best_averages(res_base)

    # Riga→Join Auto
    if debug: print("\n[DEBUG] --- CONFRONTO RIGA→JOIN AUTO ---")
    res_auto = compare_with_joins_sliding(file1_entries, file2_entries,
                                          join_len_auto=join_len_auto, join_shifts=2, debug=bool(debug))
    avg_auto_r, avg_auto_d, avg_auto_b = compute_join_averages(res_auto,
                                                               "similarity_join_rapid",
                                                               "similarity_join_diff",
                                                               "similarity_join_bleu")

    # Join Riferimento→Riga Auto
    if debug: print("\n[DEBUG] --- CONFRONTO JOIN RIFERIMENTO→RIGA AUTO ---")
    res_reference = compare_join_reference_vs_auto(file1_entries, file2_entries,
                                           join_len_reference=join_len_reference, debug=bool(debug))
    avg_reference_r, avg_reference_d, avg_reference_b = compute_join_averages(res_reference,
                                                                  "similarity_hjoin_rapid",
                                                                  "similarity_hjoin_diff",
                                                                  "similarity_hjoin_bleu")

    # Export
    save_outputs_combined(res_base, res_auto, res_reference)

    # Popup con 9 barre
    show_summary_popup_full(
        total_lines=len(file1_entries),
        sim_base_r=avg_base_r, sim_base_d=avg_base_d, sim_base_bleu=avg_base_b,
        sim_auto_r=avg_auto_r, sim_auto_d=avg_auto_d, sim_auto_bleu=avg_auto_b,
        sim_reference_r=avg_reference_r, sim_reference_d=avg_reference_d, sim_reference_bleu=avg_reference_b,
        discarded_reference=discarded_reference, discarded_auto=discarded_auto
    )


# ==============================
#        AVVIO DA CLI
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Confronto testi multi-metrica (base, join) con BLEU")
    parser.add_argument("file1", help="File riferimento (mm:ss SPEAKER: testo)")
    parser.add_argument("file2", help="[hh:mm:ss - hh:mm:ss] SPEAKER: testo")
    args = parser.parse_args()

    # GUI iniziale per i parametri opzionali
    gui_window, gui_join_auto, gui_join_reference, gui_debug = get_startup_params(
        default_window=3, default_join_auto=3, default_join_reference=2, default_debug=False
    )

    # Esecuzione
    main(
        file1_path=args.file1,
        file2_path=args.file2,
        window=gui_window,
        join_len_auto=gui_join_auto,
        join_len_reference=gui_join_reference,
        debug=gui_debug
    )

