# trainer.py
import os, json, math
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import logging

from transformers.utils.logging import set_verbosity_error, disable_progress_bar
set_verbosity_error()
for name in [
    "transformers", "transformers.trainer",
    "transformers.modeling_utils", "transformers.tokenization_utils_base",
    "huggingface_hub", "accelerate",
]:
    logging.getLogger(name).setLevel(logging.ERROR)

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from huggingface_hub import snapshot_download

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

        # 1) 스냅샷 → 로컬 경로
        model_path = snapshot_download(repo_id=model_name, resume_download=True)

        # 2) 토크나이저/데이터셋
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        full_ds = FeaturesDataset(
            docs, tokenizer, self.label2id,
            self.cfg.MAX_LENGTH, self.cfg.DOC_STRIDE
        )

        # 3) 문서 단위 분할
        train_doc_ids, valid_doc_ids = split_by_document(
            docs, self.cfg.TRAIN_SPLIT, self.cfg.RANDOM_SEED
        )
        train_idx = [i for i, did in enumerate(full_ds.doc_ids) if did in train_doc_ids]
        valid_idx = [i for i, did in enumerate(full_ds.doc_ids) if did in valid_doc_ids]
        train_ds = self._subset(full_ds, train_idx)
        valid_ds = self._subset(full_ds, valid_idx)

        # 4) 모델
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # 5) 학습 인자/콜레이터
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

            log_level="error",
            log_level_replica="error",
            evaluation_strategy="no",
            save_strategy="no",
            report_to=[],
            load_best_model_at_end=False,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )

        # pin_memory 오타 수정
        args.dataloader_pin_memory = True

        data_collator = DataCollatorForTokenClassification(tokenizer)

        # 6) 콜백 (순환참조/외부 캡처 없이 깔끔하게)
        class SilentEpochEval(TrainerCallback):
            def __init__(self, id2label, batch_size):
                self.id2label = id2label
                self.batch_size = batch_size
                self.last = (0.0, 0.0)

            def on_epoch_end(self, args, state, control, **kwargs):
                device = kwargs["model"].device
                f1, acc = evaluate_silently(
                    kwargs["model"], valid_ds, data_collator,
                    self.id2label, device, self.batch_size
                )
                self.last = (f1, acc)
                ep = int(math.ceil(state.epoch)) if state.epoch is not None else "?"
                print(f"Epoch {ep}  F1: {f1:.2f}  ACC: {acc:.2f}")

        cb = SilentEpochEval(self.id2label, self.cfg.BATCH_SIZE)

        # 7) 트레이너
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

        # 8) 저장/리턴
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
                {"eval_ent_f1": float(f"{f1:.2f}"), "eval_ent_acc": float(f"{acc:.2f}")},
                f, ensure_ascii=False, indent=2
            )

        return {"eval_ent_f1": f1, "eval_ent_acc": acc}
