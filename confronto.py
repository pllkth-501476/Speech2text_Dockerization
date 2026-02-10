import argparse
import sys
import os
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from unidecode import unidecode
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def normalize_text(s):
    s = unidecode(s).lower().strip()
    return ' '.join(s.split())

def simple_similarity(a, b):
    try:
        score = token_set_ratio(a, b)
    except Exception:
        score = 0
    return score

def bleu_score(a, b):
    try:
        ref = [b.split()]
        hyp = a.split()
        smoothie = SmoothingFunction().method4
        score = sentence_bleu(ref, hyp, smoothing_function=smoothie)
    except Exception:
        score = 0.0
    return score

def compare_files(ref_path, hyp_path, out_path):
    with open(ref_path, 'r', encoding='utf-8') as f:
        ref_lines = [normalize_text(l) for l in f if l.strip()]
    with open(hyp_path, 'r', encoding='utf-8') as f:
        hyp_lines = [normalize_text(l) for l in f if l.strip()]

    rows = []
    for i, (r, h) in enumerate(zip(ref_lines, hyp_lines)):
        sim = simple_similarity(r, h)
        bleu = bleu_score(h, r)
        rows.append({'segment': i+1, 'ref': r, 'hyp': h, 'rapidfuzz_score': sim, 'bleu': bleu})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding='utf-8')
    logger.info('Comparison saved to %s', out_path)

def parse_args():
    p = argparse.ArgumentParser(description='Compare reference and hypothesis transcripts')
    p.add_argument('--ref', required=True, help='Reference transcript file')
    p.add_argument('--hyp', required=True, help='Hypothesis transcript file (automatic)')
    p.add_argument('--out', required=True, help='Output CSV path')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.ref):
        logger.error('Reference file not found: %s', args.ref); sys.exit(1)
    if not os.path.exists(args.hyp):
        logger.error('Hypothesis file not found: %s', args.hyp); sys.exit(1)
    compare_files(args.ref, args.hyp, args.out)
