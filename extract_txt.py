# Extract text features from captions using Word2Vec embeddings
import numpy as np
import json
import time
import os

# ── Paths (update these to match your environment) ──────────────────────────
word_file  = "./meta_data/word2vec.utf8"
train_file = "./meta_data/train_data.json"
test_file  = "./meta_data/test_data.json"

save_word_npy_folder      = "./train_text_npy/"
test_save_word_npy_folder = "./test_text_npy/"
save_id_npy_folder        = "./inverted_index/"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(save_word_npy_folder, exist_ok=True)
os.makedirs(test_save_word_npy_folder, exist_ok=True)
os.makedirs(save_id_npy_folder, exist_ok=True)

# ── Build word2vec lookup dictionary ────────────────────────────────────────
begintime = time.perf_counter()
print("---------------------Building word2vec dict---------------------------")

t0 = time.perf_counter()
with open(word_file, "r", encoding="UTF-8") as fp:
    buffer = fp.read()
word = buffer.split("\n")
print("Reading time: {:.3f}s".format(time.perf_counter() - t0))

t0 = time.perf_counter()
usr_dict = {}
for i in range(len(word)):
    try:
        buffer = [item for item in word[i].split("\t")]
        usr_dict[buffer[0]] = [float(item) for item in buffer[2].split(" ")]
    except Exception:
        print("Could not parse line:", buffer)
print("Build time: {:.3f}s".format(time.perf_counter() - t0))
print("Word2vec dict size:", len(usr_dict))
print("Total time: {:.3f}s".format(time.perf_counter() - begintime))
print("---------------------Building word2vec dict---------------------------\n")

word = []  # Free memory

# ── Process training set ─────────────────────────────────────────────────────
print("TRAIN----------------word2vec---------------------------")
begintime = time.perf_counter()

t0 = time.perf_counter()
with open(train_file, "rb") as fp:
    buffer = fp.read()
dictlist = []
for item in str(buffer, encoding="UTF-8").split("\n"):
    try:
        dictlist.append(json.loads(item))
    except Exception:
        continue
print("Reading time: {:.3f}s".format(time.perf_counter() - t0))

t0 = time.perf_counter()
new_dict = []
for i in range(len(dictlist)):
    try:
        np.save(
            os.path.join(save_word_npy_folder, str(dictlist[i]["pic_id"]) + ".npy"),
            np.array(usr_dict[dictlist[i]["tags_term"]])
        )
        new_dict.append(str(dictlist[i]["pic_id"]))
    except Exception:
        continue

with open(os.path.join(save_id_npy_folder, "train_txt_npy_id.json"), "w", encoding="UTF-8") as fp:
    json.dump(new_dict, fp, ensure_ascii=False)
print("Processing time: {:.3f}s".format(time.perf_counter() - t0))
print("Total time: {:.3f}s".format(time.perf_counter() - begintime))
print("TRAIN----------------word2vec---------------------------\n")

# ── Process test set ─────────────────────────────────────────────────────────
print("TEST-----------------word2vec---------------------------")
begintime = time.perf_counter()

t0 = time.perf_counter()
with open(test_file, "rb") as fp:
    buffer = fp.read()
dictlist = []
for item in str(buffer, encoding="UTF-8").split("\n"):
    try:
        dictlist.append(json.loads(item))
    except Exception:
        continue
print("Reading time: {:.3f}s".format(time.perf_counter() - t0))

t0 = time.perf_counter()
new_dict = []
for i in range(len(dictlist)):
    try:
        np.save(
            os.path.join(test_save_word_npy_folder, str(dictlist[i]["pic_id"]) + ".npy"),
            np.array(usr_dict[dictlist[i]["tags_term"]])
        )
        new_dict.append(str(dictlist[i]["pic_id"]))
    except Exception:
        continue

with open(os.path.join(save_id_npy_folder, "test_txt_npy_id.json"), "w", encoding="UTF-8") as fp:
    json.dump(new_dict, fp, ensure_ascii=False)
print("Processing time: {:.3f}s".format(time.perf_counter() - t0))
print("Total time: {:.3f}s".format(time.perf_counter() - begintime))
print("TEST-----------------word2vec---------------------------")
print("\nPipeline complete.")
