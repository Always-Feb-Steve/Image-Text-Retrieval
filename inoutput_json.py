# Read the caption metadata and build an inverted index (word -> pic_ids) saved as output.json
import json
import re
import time

# ── Paths (update these to match your environment) ──────────────────────────
input_filenames = ["./meta_data/train_data.json", "./meta_data/val_data.json", "./meta_data/test_data.json"]
output_filename = "./meta_data/output.json"
# ─────────────────────────────────────────────────────────────────────────────

inverted_index = {}
total_start = time.time()

# Step 1: Load the JSON files (one object per line)
t0 = time.time()
dictlist = []
for input_filename in input_filenames:
    with open(input_filename, "r", encoding="utf8") as fp:
        for line in fp:
            dictlist.append(json.loads(line.strip()))
print("Reading time: {:.3f}s".format(time.time() - t0))

# Step 2: Build inverted index — key: lowercase caption word, value: list of pic_ids
t0 = time.time()
for item in dictlist:
    words = set(re.findall(r"[a-z]+", " ".join(item["captions"]).lower()))
    for word in words:
        inverted_index.setdefault(word, []).append(item["pic_id"])

print("#Index entries:", len(inverted_index))
print("Building time: {:.3f}s".format(time.time() - t0))

# Step 3: Write inverted index to output.json
t0 = time.time()
with open(output_filename, "w", encoding="utf8") as fp:
    json.dump(inverted_index, fp)
print("Writing time: {:.3f}s".format(time.time() - t0))

print("Total time: {:.3f}s".format(time.time() - total_start))
