"""Fix 1a: safe-text P(risky) on train label==0 (primary) and test label==0 (reference)."""
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import BertForSequenceClassification

_ROOT = Path(__file__).resolve().parent.parent
ARROW = _ROOT / "arrow_datasets"
MODEL = _ROOT / "models" / "tinybert_guardrail"


def _collect_probs(split_name: str) -> tuple[np.ndarray, int]:
    ds = load_from_disk(str(ARROW))
    sp = ds[split_name]
    idxs = [i for i in range(len(sp)) if int(sp[i]["label"]) == 0]
    model = BertForSequenceClassification.from_pretrained(str(MODEL))
    model.eval()
    all_probs: list[float] = []
    BATCH = 32
    with torch.no_grad():
        for start in range(0, len(idxs), BATCH):
            chunk = idxs[start : start + BATCH]
            batch_in = [sp[i]["input_ids"] for i in chunk]
            batch_tt = [sp[i]["token_type_ids"] for i in chunk]
            maxlen = max(len(x) for x in batch_in)
            n = len(chunk)
            input_ids = torch.zeros(n, maxlen, dtype=torch.long)
            attn = torch.zeros(n, maxlen, dtype=torch.long)
            tti = torch.zeros(n, maxlen, dtype=torch.long)
            for j, ids in enumerate(batch_in):
                L = len(ids)
                input_ids[j, :L] = torch.tensor(ids, dtype=torch.long)
                attn[j, :L] = 1
                tt = batch_tt[j]
                tti[j, :L] = torch.tensor(tt[:L], dtype=torch.long)
            out = model(
                input_ids=input_ids,
                attention_mask=attn,
                token_type_ids=tti,
            )
            probs = torch.softmax(out.logits, dim=-1)[:, 1]
            all_probs.extend(probs.tolist())
    return np.asarray(all_probs, dtype=np.float64), len(idxs)


def _print_block(title: str, arr: np.ndarray, n_req: int) -> None:
    print(title)
    print(f"  (rows used: {n_req})")
    if n_req == 0:
        print("  SKIP: no rows")
        return
    for pct in [50, 75, 90, 95, 99]:
        print(f"  p{pct}: {float(np.percentile(arr, pct)):.3f}")
    print(f"  max:   {float(arr.max()):.3f}")
    print(f"  mean:  {float(arr.mean()):.3f}")
    print("\n  FPR at candidate t (fraction of label=0 rows with P>=t):")
    for t in [0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.65, 0.70]:
        fp = int((arr >= t).sum())
        fpr = fp / n_req
        print(f"    t={t:.2f}: {fp} ({fpr:.1%})")
    p95 = float(np.percentile(arr, 95))
    print(f"\n  p95 = {p95:.3f}  (~5% of this set exceeds if thresholds align)")


def main() -> None:
    arr_tr, n_tr = _collect_probs("train")
    _print_block("=== TRAIN label==0 (use for t_warn / benign FPR) ===", arr_tr, n_tr)

    arr_te, n_te = _collect_probs("test")
    print()
    _print_block("=== TEST label==0 (skewed reference only) ===", arr_te, n_te)


if __name__ == "__main__":
    main()
