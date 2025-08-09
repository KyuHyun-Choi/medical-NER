# -*- coding: utf-8 -*-
import os, json
import warnings
warnings.filterwarnings("ignore")

import datetime as dt
import pandas as pd
from tqdm.auto import tqdm

import config as C
from utils import set_seed
from data import scan_pairs, collect_label_names, make_bio_maps, load_all_docs
from trainer import ModelTrainer

def main():
    set_seed(C.RANDOM_SEED)
    os.makedirs(C.OUTPUT_ROOT, exist_ok=True)

    pairs = scan_pairs(C.DATA_DIR)
    label_names = collect_label_names(pairs)
    labels, label2id, id2label = make_bio_maps(label_names)
    docs = load_all_docs(pairs)

    trainer = ModelTrainer(C, labels, label2id, id2label)

    all_results = []
    for k, m in tqdm(C.MODELS.items(), desc="모델 파인튜닝"):
        metrics = trainer.train_one(k, m, docs)
        all_results.append({
            "model": k,
            "eval_ent_f1": float(f"{metrics['eval_ent_f1']:.2f}"),
            "eval_ent_acc": float(f"{metrics['eval_ent_acc']:.2f}"),
        })

    # 결과 저장
    try:
        df = pd.DataFrame(all_results)
        df.insert(0, "run_at", str(dt.datetime.now()))
        df.to_csv(os.path.join(C.OUTPUT_ROOT, "metrics.csv"), index=False, encoding="utf-8")
    except Exception:
        with open(os.path.join(C.OUTPUT_ROOT, "metrics_all.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
