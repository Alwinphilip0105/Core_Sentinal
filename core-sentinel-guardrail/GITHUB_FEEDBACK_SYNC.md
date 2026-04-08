# Syncing feedback across laptops with GitHub

Use a **private** GitHub repository (free) so label corrections stay off public search. This is **not** real-time; it is **pull → work → commit → push** on each machine.

## 1. One-time setup

1. Create a **private** repo on GitHub (empty, no README is fine).
2. On this machine, from the repo root (`Core_Sentinal`):

   ```bash
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

   If the project was not a git repo yet:

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

   Then add `remote` and `push` as above.

3. On the **other laptop**: clone the same repo, create the same `.venv`, install deps (`pip install -r core-sentinel-guardrail/requirements.txt` or your usual flow).

## 2. What to sync for feedback / retraining

These paths are **not** ignored by `.gitignore` and are safe to commit if the repo is **private**:

| File | Purpose |
|------|---------|
| `core-sentinel-guardrail/logs/feedback_store.jsonl` | User corrections (predicted vs correct label) |
| `core-sentinel-guardrail/logs/feedback_fulltext.jsonl` | **Full** pasted text per feedback (for training export; prefer syncing this) |
| `core-sentinel-guardrail/logs/hash_index.jsonl` | 500-char preview fallback if fulltext missing (older rows) |

**Do not commit** (already gitignored): `logs/guardrail.db*`, models, `.env`, large datasets.

Optional: `core-sentinel-guardrail/config/user_settings.json` if you want the same UI prefs on both machines (may differ per screen — commit only if you want).

## 3. Routine workflow

**Before** using the guardrail on a laptop:

```bash
cd /path/to/Core_Sentinal
git pull
```

**After** collecting feedback (or end of day):

```bash
git status
git add core-sentinel-guardrail/logs/feedback_store.jsonl core-sentinel-guardrail/logs/hash_index.jsonl
git commit -m "feedback: sync corrections"
git push
```

On the other laptop, `git pull` before the next session so the app reads the latest files.

## 4. Merge conflicts

If both laptops committed **different** new lines to the same JSONL, Git may report a conflict. Open the file, remove `<<<<<<<`, `=======`, `>>>>>>>` markers, and **keep all valid JSON lines** (one JSON object per line). Then:

```bash
git add <resolved-files>
git commit -m "merge: feedback jsonl"
git push
```

**Easier:** avoid editing on both machines the same day; **pull before** you use the app, **push when** you stop.

## 5. Security

- Keep the repo **private**.
- `hash_index.jsonl` can contain **snippets** of pasted text — treat the repo like sensitive data.

## 6. Smoke test on a second laptop

After `git clone` + `git pull`, run `python core-sentinel-guardrail/main.py` (or your entry point) and confirm `logs/feedback_store.jsonl` updates locally when you submit bubble feedback; then commit/push and verify the first laptop sees new lines after `git pull`.

## 7. Using merged feedback to retrain (3-class)

After pulling everyone’s logs, run `python merge_feedback_to_training.py` then `python data.py multi_real_synthetic` and `python train.py`. See **`FEEDBACK_TRAINING_LOOP.md`** for the full loop.
