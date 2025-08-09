import os

# ── 경로 ──────────────────────────────────────────────────────
DATA_DIR = "MACCROBAT2020"
OUTPUT_ROOT = "outputs"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ── 시드 & 분할 ───────────────────────────────────────────────
RANDOM_SEED = 42
TRAIN_SPLIT = 0.9   # 문서 기준 분할 비율

# ── 토크나이저/패킹 ───────────────────────────────────────────
MAX_LENGTH = 512
DOC_STRIDE = 128

# ── 학습 ──────────────────────────────────────────────────────
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_ACCUM_STEPS = 1

# ── 모델들 ────────────────────────────────────────────────────
MODELS = {
    # 도메인 전용
    "scibert":         "allenai/scibert_scivocab_uncased",
    "biobert":         "dmis-lab/biobert-base-cased-v1.1",
    "pubmedbert":      "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    # 일반 베이스
    "bert-base":       "bert-base-cased",
    "roberta-base":    "roberta-base",
    "albert-base":     "albert-base-v2",
    "deberta-v3":      "microsoft/deberta-v3-base",
}
