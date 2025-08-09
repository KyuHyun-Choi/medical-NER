# -*- coding: utf-8 -*-
import os, re, json
from glob import glob
from dataclasses import dataclass
from typing import List, Tuple, Dict

from tqdm.auto import tqdm

from transformers import AutoTokenizer
from torch.utils.data import Dataset

# ── 데이터 구조 ───────────────────────────────────────────────
@dataclass
class Doc:
    text: str
    spans: List[Tuple[int, int, str]]  # (start, end, label)

# ── 스캔 & 파싱 ───────────────────────────────────────────────
def scan_pairs(data_dir: str):
    txts = sorted(glob(os.path.join(data_dir, "*.txt")))
    pairs = []
    for t in tqdm(txts, desc="스캔(.txt)", leave=False):
        base = os.path.splitext(os.path.basename(t))[0]
        ann = os.path.join(data_dir, base + ".ann")
        if os.path.exists(ann):
            pairs.append((t, ann))
    if not pairs:
        raise FileNotFoundError(f"'{data_dir}'에서 .txt/.ann 페어를 찾지 못했습니다.")
    return pairs

def parse_ann(ann_path: str):
    spans = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            tag = parts[1]  # "Age 8 19" 또는 "Entity 10 12;14 16"
            first_space = tag.find(" ")
            if first_space == -1:
                continue
            label = tag[:first_space]
            offsets_str = tag[first_space:].strip()
            for chunk in offsets_str.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    s, e = chunk.split()
                    s, e = int(s), int(e)
                    if e > s:
                        spans.append((s, e, label))
                except Exception:
                    continue
    # 긴 스팬 우선 (겹침 처리 시 긴 스팬이 우선 할당되도록)
    spans.sort(key=lambda x: (x[1] - x[0]), reverse=True)
    return spans

def load_all_docs(pairs):
    docs = []
    for txt, ann in tqdm(pairs, desc="문서 로드", leave=False):
        with open(txt, "r", encoding="utf-8") as f:
            text = f.read()
        spans = parse_ann(ann)
        docs.append(Doc(text, spans))
    return docs

def collect_label_names(pairs):
    labs = set()
    for _, ann in tqdm(pairs, desc="라벨 스캔(.ann)", leave=False):
        for _, _, L in parse_ann(ann):
            labs.add(L)
    return sorted(labs)

def make_bio_maps(entity_labels):
    labels = ["O"]
    for L in entity_labels:
        labels += [f"B-{L}", f"I-{L}"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label

# ── BIO 라벨링 ────────────────────────────────────────────────
def spans_to_bio_for_offsets(offsets, spans, label2id):
    N = len(offsets)
    labels = [-100] * N
    assigned = [None] * N
    for i, (ts, te) in enumerate(offsets):
        if te <= ts:  # 특수토큰
            continue
        for si, (s, e, L) in enumerate(spans):
            if te <= s or e <= ts:
                continue
            assigned[i] = si
            break
        if assigned[i] is not None:
            L = spans[assigned[i]][2]
            labels[i] = label2id[f"I-{L}"]
    for i in range(N):
        si = assigned[i]
        if si is None or labels[i] == -100:
            continue
        L = spans[si][2]
        prev_same = (i > 0 and assigned[i - 1] == si and labels[i - 1] != -100)
        labels[i] = label2id[(f"I-{L}" if prev_same else f"B-{L}")]
    return labels

# ── 토치 데이터셋 ─────────────────────────────────────────────
class FeaturesDataset(Dataset):
    def __init__(self, docs: List[Doc], tokenizer, label2id, max_length=512, doc_stride=128):
        self.features = []
        self.doc_ids = []  # 문서 경계 추적
        for doc_id, doc in enumerate(tqdm(docs, desc="토크나이즈+BIO 정렬", leave=False)):
            enc = tokenizer(
                doc.text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
            )
            offsets_list = enc.pop("offset_mapping")
            for i, offsets in enumerate(offsets_list):
                labels = spans_to_bio_for_offsets(offsets, doc.spans, label2id)
                self.features.append({
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels": labels,
                })
                self.doc_ids.append(doc_id)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

# ── 문서 단위 분할 ────────────────────────────────────────────
def split_by_document(docs: List[Doc], train_ratio=0.9, seed=42):
    import random
    n = len(docs)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    cut = int(n * train_ratio)
    return set(idxs[:cut]), set(idxs[cut:])
