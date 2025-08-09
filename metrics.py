from typing import List, Tuple
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification

def bio_decode_batch(pred_ids, label_ids, id2label):
    y_pred, y_true = [], []
    for p, l in zip(pred_ids, label_ids):
        sp, sl = [], []
        for pi, li in zip(p, l):
            if li == -100:
                continue
            sp.append(id2label[int(pi)])
            sl.append(id2label[int(li)])
        y_pred.append(sp)
        y_true.append(sl)
    return y_true, y_pred

def entities_from_bio(seq):
    ents = []
    cur = None
    st = None
    for i, tag in enumerate(seq + ["O"]):
        if tag.startswith("B-"):
            if cur is not None:
                ents.append((cur, st, i))
            cur = tag[2:]
            st = i
        elif tag.startswith("I-"):
            lab = tag[2:]
            if cur is None or lab != cur:
                if cur is not None:
                    ents.append((cur, st, i))
                cur = lab
                st = i
        else:
            if cur is not None:
                ents.append((cur, st, i))
                cur = None
                st = None
    return ents

def entity_f1_acc(y_true, y_pred):
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        T = set(entities_from_bio(t))
        P = set(entities_from_bio(p))
        tp += len(T & P)
        fp += len(P - T)
        fn += len(T - P)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc  = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return round(f1, 2), round(acc, 2)

@torch.no_grad()
def evaluate_silently(model, dataset, data_collator, id2label, device, bs):
    model.eval()
    loader = DataLoader(
        dataset, batch_size=bs, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=data_collator
    )
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        labs  = labels.cpu().numpy()
        all_preds.extend(list(preds))
        all_labels.extend(list(labs))
    y_true, y_pred = bio_decode_batch(all_preds, all_labels, id2label)
    return entity_f1_acc(y_true, y_pred)
