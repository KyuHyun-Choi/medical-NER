# -*- coding: utf-8 -*-
import os, json, math
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from data import FeaturesDataset, split_by_document
from metrics import evaluate_silently

class ModelTrainer:
    def __init__(self, config, labels, label2id, id2label):
        self.cfg = config
        self.labels = labels
        self.label2id = label2id
        self.id2label = id2label

    def _subset(self, base: Dataset, indices):
        class _Subset(Dataset):
            def __init__(self, base, indices):
                self.base = base
                self.indices = indices
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                return self.base[self.indices[i]]
        return _Subset(base, indices)

    def train_one(self, key: str, model_name: str, docs) -> Dict[str, float]:
        out_dir = os.path.join(self.cfg.OUTPUT_ROOT, key)
        os.makedirs(out_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        full_ds = FeaturesDataset(
            docs, tokenizer, self.label2id,
            self.cfg.MAX_LENGTH, self.cfg.DOC_STRIDE
        )

        train_doc_ids, valid_doc_ids = split_by_document(
            docs, self.cfg.TRAIN_SPLIT, self.cfg.RANDOM_SEED
        )
        train_idx = [i for i, did in enumerate(full_ds.doc_ids) if did in train_doc_ids]
        valid_idx = [i for i, did in enumerate(full_ds.doc_ids) if did in valid_doc_ids]

        train_ds = self._subset(full_ds, train_idx)
        valid_ds = self._subset(full_ds, valid_idx)

        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=self.cfg.BATCH_SIZE,
            per_device_eval_batch_size=self.cfg.BATCH_SIZE,
            learning_rate=self.cfg.LR,
            num_train_epochs=self.cfg.EPOCHS,
            weight_decay=self.cfg.WEIGHT_DECAY,
            warmup_ratio=self.cfg.WARMUP_RATIO,
            gradient_accumulation_steps=self.cfg.GRAD_ACCUM_STEPS,
            fp16=torch.cuda.is_available(),

            # 자동 로그/평가/저장 미사용
            logging_strategy="no",
            evaluation_strategy="no",
            save_strategy="no",
            report_to=[],
            load_best_model_at_end=False,

            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer)

        class SilentEpochEval(TrainerCallback):
            def __init__(self):
                self.last = (0.0, 0.0)
            def on_epoch_end(self, args, state, control, **kwargs):
                device = kwargs["model"].device
                f1, acc = evaluate_silently(
                    kwargs["model"], valid_ds, data_collator,
                    self_outer.id2label, device, self_outer.cfg.BATCH_SIZE
                )
                self.last = (f1, acc)
                ep = int(math.ceil(state.epoch)) if state.epoch is not None else "?"
                print(f"Epoch {ep}  F1: {f1:.2f}  ACC: {acc:.2f}")

        # 콜백에서 self 접근을 위해 바깥 참조
        self_outer = self
        cb = SilentEpochEval()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None,
            callbacks=[cb],
        )

        tqdm.write(
            f"[{key}] docs: train={len(train_doc_ids)}, valid={len(valid_doc_ids)} | "
            f"samples: train={len(train_ds)}, valid={len(valid_ds)}"
        )
        trainer.train()

        f1, acc = cb.last
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"labels": self.labels, "label2id": self.label2id, "id2label": self.id2label},
                f, ensure_ascii=False, indent=2
            )
        with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"eval_ent_f1": float(f"{f1:.2f}"),
                 "eval_ent_acc": float(f"{acc:.2f}")},
                f, ensure_ascii=False, indent=2
            )
        return {"eval_ent_f1": f1, "eval_ent_acc": acc}
