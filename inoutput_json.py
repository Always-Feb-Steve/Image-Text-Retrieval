# Read input.json and build an inverted index saved as output.json
import json
import time

# ── Paths (update these to match your environment) ──────────────────────────
input_filename  = "./meta_data/input.json"
output_filename = "./meta_data/output.json"
# ─────────────────────────────────────────────────────────────────────────────

inverted_index = {}
total_start = time.time()

# Step 1: Load the JSON file
t0 = time.time()
dictlist = []
with open(input_filename, "r", encoding="utf8") as fp:
    for line in fp:
        item = json.loads(line.strip())
        dictlist.append(item)
print("Reading time: {:.3f}s".format(time.time() - t0))

# Step 2: Build inverted index — key: tags_term, value: list of pic_ids
t0 = time.time()
for item in dictlist:
    tag = item["tags_term"]
    if tag in inverted_index:
        inverted_index[tag].append(item["pic_id"])
    else:
        inverted_index[tag] = [item["pic_id"]]

print("#Index entries:", len(inverted_index))
print("Building time: {:.3f}s".format(time.time() - t0))

# Step 3: Write inverted index to output.json
t0 = time.time()
with open(output_filename, "w", encoding="utf8") as fp:
    json.dump(inverted_index, fp)
print("Writing time: {:.3f}s".format(time.time() - t0))

print("Total time: {:.3f}s".format(time.time() - total_start))
