# Compute the intersection of image IDs and text IDs for train/test sets
import numpy as np
import json
import os

# ── Paths (update these to match your environment) ──────────────────────────
save_id_npy_folder = "./inverted_index/"
train_img_id_path  = "./inverted_index/train_img_npy_id.json"
train_txt_id_path  = "./inverted_index/train_txt_npy_id.json"
test_img_id_path   = "./inverted_index/test_img_npy_id.json"
test_txt_id_path   = "./inverted_index/test_txt_npy_id.json"
# ─────────────────────────────────────────────────────────────────────────────

# ── Training set intersection ────────────────────────────────────────────────
with open(train_img_id_path, "rb") as fp:
    train_img_ids = json.loads(str(fp.read(), encoding="UTF-8"))
with open(train_txt_id_path, "rb") as fp:
    train_txt_ids = set(json.loads(str(fp.read(), encoding="UTF-8")))

train_ids = [id for id in train_img_ids if id in train_txt_ids]

print("#Train images:", len(train_img_ids))
print("#Train texts: ", len(train_txt_ids))
print("Train samples (intersection):", len(train_ids))

# ── Test set intersection ────────────────────────────────────────────────────
with open(test_img_id_path, "rb") as fp:
    test_img_ids = json.loads(str(fp.read(), encoding="UTF-8"))
with open(test_txt_id_path, "rb") as fp:
    test_txt_ids = set(json.loads(str(fp.read(), encoding="UTF-8")))

test_ids = [id for id in test_img_ids if id in test_txt_ids]

print("=====================")
print("#Test images:", len(test_img_ids))
print("#Test texts: ", len(test_txt_ids))
print("Test samples (intersection):", len(test_ids))

# ── Save results ─────────────────────────────────────────────────────────────
with open(os.path.join(save_id_npy_folder, "train_npy_id.json"), "w", encoding="UTF-8") as fp:
    json.dump(train_ids, fp, ensure_ascii=False)
with open(os.path.join(save_id_npy_folder, "test_npy_id.json"), "w", encoding="UTF-8") as fp:
    json.dump(test_ids, fp, ensure_ascii=False)
