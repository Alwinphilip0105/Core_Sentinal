"""
Generate synthetic low-risk training samples and append them to the existing
tokenized Arrow dataset under `arrow_datasets_nemotron/`.

Usage:
  python generate_low_samples.py

Optional env vars:
  - GUARDRAIL_ARROW_SAVE_DIR (default: arrow_datasets_nemotron)
  - GUARDRAIL_LOW_PER_CATEGORY (default: 115)
  - GUARDRAIL_TINYBERT_ID (default: huawei-noah/TinyBERT_General_4L_312D)
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer


DEFAULT_TINYBERT_ID = "huawei-noah/TinyBERT_General_4L_312D"
DEFAULT_ARROW_DIR = "arrow_datasets_nemotron"
DEFAULT_PER_CATEGORY = 115


def _generate_category_texts(rng: random.Random, per_category: int) -> list[str]:
    """
    Generate low-risk examples across required categories.
    Target: ~100-150 examples per category (default 120).
    """
    first_names = ["Sarah", "Alex", "Jamie", "Taylor", "Jordan", "Casey", "Morgan", "Riley", "Sam", "Avery"]
    last_names = ["Smith", "Lee", "Brown", "Miller", "Davis", "Clark", "Wilson", "Hall", "Young", "King"]
    initials = ["J.K.", "A.R.", "M.T.", "L.P.", "D.S.", "R.B.", "T.C.", "N.W.", "K.H.", "S.L."]
    titles = ["Dr.", "Prof.", "Mr.", "Ms."]
    cities = ["Austin", "Seattle", "Denver", "Madison", "Portland", "Raleigh", "Columbus", "Tampa"]
    states = ["Texas", "Washington", "Colorado", "Wisconsin", "Oregon", "North Carolina", "Ohio", "Florida"]
    regions = ["Pacific Northwest", "Midwest", "Northeast", "Southeast", "Southwest"]
    landmarks = ["Central Park", "Union Square", "City Hall", "the public library", "the river walk"]
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    role_emails = [
        "support@company.com",
        "info@university.edu",
        "billing@business.org",
        "helpdesk@agency.gov",
        "admissions@college.edu",
        "events@museum.org",
        "careers@startup.com",
    ]
    age_groups = ["18-25", "26-35", "30-40", "40-50", "50-60", "60-70"]

    texts: list[str] = []

    # 1) Partial/Generic Names
    for _ in range(per_category):
        mode = rng.choice(["first", "title", "initials"])
        if mode == "first":
            texts.append(f"Hi, my name is {rng.choice(first_names)}.")
        elif mode == "title":
            texts.append(f"Please ask {rng.choice(titles)} {rng.choice(last_names)} to call back.")
        else:
            texts.append(f"Signed, {rng.choice(initials)}.")

    # 2) Non-sensitive Locations
    for _ in range(per_category):
        mode = rng.choice(["city", "region", "landmark"])
        if mode == "city":
            texts.append(f"I live in {rng.choice(cities)}, {rng.choice(states)}.")
        elif mode == "region":
            texts.append(f"We are based out of the {rng.choice(regions)}.")
        else:
            texts.append(f"Meet me near {rng.choice(landmarks)}.")

    # 3) Generic Contact References (no actual data)
    generic_contact_templates = [
        "Please email me at my work address.",
        "You can reach me by phone during business hours.",
        "Send the form to our office address.",
        "The team can follow up through the usual contact channel.",
        "Please use the contact details on file.",
    ]
    for _ in range(per_category):
        texts.append(rng.choice(generic_contact_templates))

    # 4) Publicly Known Emails (role-based)
    for _ in range(per_category):
        email = rng.choice(role_emails)
        texts.append(f"Contact {email} for assistance with this request.")

    # 5) Non-sensitive Dates
    for _ in range(per_category):
        mode = rng.choice(["meeting", "fiscal", "birth_year"])
        if mode == "meeting":
            day = rng.randint(1, 28)
            texts.append(f"The meeting is on {rng.choice(months)} {day}th.")
        elif mode == "fiscal":
            texts.append(f"Our fiscal year ends in {rng.choice(months)}.")
        else:
            texts.append(f"She was born in {rng.randint(1970, 2004)}.")

    # 6) Generic ID References (no actual numbers)
    generic_id_templates = [
        "Your order number will be emailed to you.",
        "Please have your membership ID ready.",
        "Reference your ticket number when calling.",
        "Keep your confirmation number available for support.",
        "Use your account reference when submitting the form.",
    ]
    for _ in range(per_category):
        texts.append(rng.choice(generic_id_templates))

    # 7) Demographic Info (non-identifying)
    for _ in range(per_category):
        mode = rng.choice(["patient", "survey", "age_group"])
        if mode == "patient":
            age = rng.randint(18, 75)
            gender = rng.choice(["male", "female"])
            texts.append(f"The patient is a {age}-year-old {gender}.")
        elif mode == "survey":
            gender = rng.choice(["female", "male", "non-binary"])
            texts.append(f"Survey respondent identified as {gender}.")
        else:
            texts.append(f"Age group: {rng.choice(age_groups)}.")

    return texts


def main() -> None:
    guardrail_dir = Path(__file__).resolve().parent
    arrow_dir = Path(os.environ.get("GUARDRAIL_ARROW_SAVE_DIR", DEFAULT_ARROW_DIR))
    if not arrow_dir.is_absolute():
        arrow_dir = (guardrail_dir / arrow_dir).resolve()

    per_category = int(os.environ.get("GUARDRAIL_LOW_PER_CATEGORY", DEFAULT_PER_CATEGORY))
    tinybert_id = os.environ.get("GUARDRAIL_TINYBERT_ID", DEFAULT_TINYBERT_ID)

    if not arrow_dir.exists():
        raise FileNotFoundError(
            f"Arrow dataset dir not found: {arrow_dir}\n"
            f"Build it first with: python data.py nemotron (or set GUARDRAIL_ARROW_SAVE_DIR)."
        )

    print(f"[generate_low_samples] Loading dataset from: {arrow_dir}")
    ds_dict: DatasetDict = load_from_disk(str(arrow_dir))
    if "train" not in ds_dict:
        raise ValueError("Dataset is missing a 'train' split.")

    tokenizer = AutoTokenizer.from_pretrained(tinybert_id)

    rng = random.Random(42)
    new_texts = _generate_category_texts(rng, per_category=per_category)

    enc = tokenizer(
        new_texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors=None,
    )

    token_type_ids = enc.get("token_type_ids")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    new_rows = []
    for j in range(len(new_texts)):
        row = {
            "risk": "low",
            "label": 0,  # low=0 for 3-class models
            "input_ids": input_ids[j],
            "attention_mask": attention_mask[j],
            "token_type_ids": token_type_ids[j] if token_type_ids is not None else [0] * len(input_ids[j]),
        }
        new_rows.append(row)

    new_ds = Dataset.from_list(new_rows)
    print(
        "[generate_low_samples] Generated "
        f"{len(new_ds)} low samples (~{per_category} per category across 7 categories)."
    )

    old_train = ds_dict["train"]
    old_val = ds_dict.get("validation")
    updated_train = concatenate_datasets([old_train, new_ds])
    updated_val = concatenate_datasets([old_val, new_ds]) if old_val is not None else new_ds
    test_split = ds_dict.get("test")
    ds_dict = DatasetDict(
        {
            "train": updated_train,       # original + synthetic low
            "validation": updated_val,    # original + synthetic low
            "test": test_split,           # unchanged
        }
    )

    # Print final distribution (risk column is string)
    from collections import Counter

    train_dist = Counter(ds_dict["train"]["risk"])
    val_dist = Counter(ds_dict["validation"]["risk"]) if ds_dict.get("validation") is not None else {}
    n_train = ds_dict["train"].num_rows
    n_val = ds_dict["validation"].num_rows if ds_dict.get("validation") is not None else 0
    n_test = ds_dict["test"].num_rows if ds_dict.get("test") is not None else 0
    print(f"[generate_low_samples] New train size: {n_train}  distribution: {dict(train_dist)}")
    print(f"[generate_low_samples] New validation size: {n_val}  distribution: {dict(val_dist)}")
    print(f"[verify] About to save: train={len(updated_train)}, val={len(updated_val)}, test={n_test}")

    print(f"[generate_low_samples] Saving updated dataset back to: {arrow_dir}")
    tmp_dir = str(arrow_dir) + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    ds_dict.save_to_disk(tmp_dir)
    shutil.rmtree(str(arrow_dir))
    os.rename(tmp_dir, str(arrow_dir))
    print(f"[generate_low_samples] Successfully saved to {arrow_dir}")
    print("[generate_low_samples] Done.")


if __name__ == "__main__":
    main()

