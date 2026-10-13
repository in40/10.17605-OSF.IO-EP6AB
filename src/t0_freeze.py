"""
t0_freeze.py — T0 freeze: validate the 2x2 dataset, hash everything, emit manifest_final.json.

This is the integrity anchor of the preregistration. Run BEFORE any fitting (T1).
It does NOT train anything. It locks:
  - SHA-256 of the dataset file (labeled_2x2.jsonl)
  - SHA-256 of the frozen taxonomies (families, constraint types, system prompts, C3 protocol)
  - the split assignment summary (train/val/test, which F/T/SP in each)
  - the validator result (all T0 invariants)
  - a readiness flag

If ready=False, do NOT proceed to T1.
"""
from __future__ import annotations
import json, hashlib, datetime
from pathlib import Path
from collections import Counter
from manifest_spec import (validate, ATTACK_FAMILIES, CONSTRAINT_TYPES,
                          SYSTEM_PROMPTS, C3_PROTOCOL, TRAIN_FAMILIES,
                          TEST_FAMILIES, TRAIN_CONSTRAINT_TYPES, TEST_CONSTRAINT_TYPES)

DATA = Path(__file__).parent / "data" / "labeled_2x2.jsonl"
OUT = Path(__file__).parent / "manifest_final.json"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_obj(obj):
    # canonical JSON: sorted keys, no ascii escape -> deterministic hash
    s = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(s.encode()).hexdigest()

def main():
    recs = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
    for r in recs:
        r["category"] = r["cell"]  # align with validate() key

    # 1. validate
    violations = validate(recs)

    # 2. hashes
    data_hash = sha256_file(DATA)
    taxonomy_hash = sha256_obj({
        "attack_families": ATTACK_FAMILIES,
        "constraint_types": CONSTRAINT_TYPES,
        "system_prompts": {k: v[1] for k, v in SYSTEM_PROMPTS.items()},
        "c3_protocol": C3_PROTOCOL,
        "train_families": TRAIN_FAMILIES,
        "test_families": TEST_FAMILIES,
        "train_constraint_types": TRAIN_CONSTRAINT_TYPES,
        "test_constraint_types": TEST_CONSTRAINT_TYPES,
    })

    # 3. split summary
    split_counts = dict(Counter(r["split"] for r in recs))
    cell_counts = dict(Counter(r["cell"] for r in recs))
    test_fams = sorted({r["attack_family"] for r in recs if r["split"]=="test" and r["attack_family"]})
    test_types = sorted({r["constraint_type"] for r in recs if r["split"]=="test" and r["constraint_type"]})
    nonexcl_test = sorted(t for t in test_types if not CONSTRAINT_TYPES[t]["exclusion_pattern"])
    ov = [r for r in recs if r["O"]==1]
    nov = [r for r in recs if r["O"]==0]

    # 4. target checks
    checks = {
        "override_total>=600": len(ov) >= 600,
        "nonoverride_C2C4>=600": sum(1 for r in recs if r["cell"] in ("C2","C4")) >= 600,
        "H3_heldout_constraints>=500": sum(1 for r in recs if r["cell"]=="CONSTRAINT" and r["split"]=="test") >= 500,
        "val_override>=200": sum(1 for r in ov if r["split"]=="val") >= 200,
        "k_test_families>=8": len(test_fams) >= 8,
        "k_types_test>=8": len(test_types) >= 8,
        "nonexclusion_test>=2": len(nonexcl_test) >= 2,
        "no_validator_violations": len(violations) == 0,
    }
    ready = all(checks.values())

    manifest = {
        "version": "T0-final",
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "prereg_draft": "6ac910dee14a8f5f57482614",
        "dataset_file": str(DATA.name),
        "dataset_sha256": data_hash,
        "taxonomy_sha256": taxonomy_hash,
        "n_records": len(recs),
        "split_counts": split_counts,
        "cell_counts": cell_counts,
        "override_total": len(ov),
        "nonoverride_total": len(nov),
        "test_families": test_fams,
        "test_constraint_types": test_types,
        "nonexclusion_test_types": nonexcl_test,
        "validator_violations": violations,
        "target_checks": checks,
        "READY_FOR_T1": ready,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT}")
    print(f"dataset_sha256   = {data_hash}")
    print(f"taxonomy_sha256  = {taxonomy_hash}")
    print(f"records={len(recs)}  split={split_counts}")
    print(f"test_families={test_fams}")
    print(f"nonexclusion_test={nonexcl_test}")
    print("\n--- target checks ---")
    for k, v in checks.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print(f"\nREADY_FOR_T1 = {ready}")
    if not ready:
        print(">>> DO NOT proceed to T1. Fix failing checks first.")

if __name__ == "__main__":
    main()
