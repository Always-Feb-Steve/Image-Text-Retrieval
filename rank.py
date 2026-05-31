# Train a cross-modal ranking model (image-text retrieval) using hinge loss
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch import optim
import time

# ── Architecture options (from experimentation) ───────────────────────────────
# Baseline:   (2048 -> 128)  —  (200 -> 128)
# Variant 1:  (2048 -> 512)  —  (200 -> 512)
# Variant 2:  (2048 -> 512 -> 128)  —  (200 -> 512 -> 128)
# Variant 3:  (2048 -> 512(softmax) -> 128)  —  (200 -> 512(softmax) -> 128)
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths (update these to match your environment) ──────────────────────────
train_id_file      = "./inverted_index/train_npy_id.json"
test_id_file       = "./inverted_index/test_npy_id.json"
train_pic_npy_folder = "./train_pic_npy/"
test_pic_npy_folder  = "./test_pic_npy/"
train_txt_npy_folder = "./train_text_npy/"
test_txt_npy_folder  = "./test_text_npy/"
save_model_path    = "./models/net.pkl"
# ─────────────────────────────────────────────────────────────────────────────

# ── Load training data ───────────────────────────────────────────────────────
pic_arr, word_arr, dict_id = [], [], []
with open(train_id_file, 'r') as fp:
    dict_id = json.load(fp)
print("#Training samples:", len(dict_id))

for i in range(len(dict_id)):
    npy_data = np.load(os.path.join(train_pic_npy_folder, dict_id[i] + '.npy'))
    pic_arr.append(torch.tensor(npy_data.reshape(2048)))

for i in range(len(dict_id)):
    npy_data = np.load(os.path.join(train_txt_npy_folder, dict_id[i] + '.npy'))
    word_arr.append(torch.tensor(npy_data).float())

print("#Training samples loaded:", len(pic_arr), len(word_arr))

# ── Load test data ───────────────────────────────────────────────────────────
test_pic_arr, test_word_arr, dict_id = [], [], []
with open(test_id_file, 'r') as fp:
    dict_id = json.load(fp)
print("#Testing samples:", len(dict_id))

for i in range(len(dict_id)):
    npy_data = np.load(os.path.join(test_pic_npy_folder, dict_id[i] + '.npy'))
    test_pic_arr.append(torch.tensor(npy_data.reshape(2048)))

for i in range(len(dict_id)):
    npy_data = np.load(os.path.join(test_txt_npy_folder, dict_id[i] + '.npy'))
    test_word_arr.append(torch.tensor(npy_data).float())

print("#Testing samples loaded:", len(test_pic_arr), len(test_word_arr))


# ── Dataset ──────────────────────────────────────────────────────────────────
class RetrievalDataset(Dataset):
    def __init__(self):
        self.x_data = pic_arr   # image features (2048-dim)
        self.y_data = word_arr  # text features  (200-dim)
        self.length = len(pic_arr)

    def __getitem__(self, index):
        # Returns: (positive image, negative image, matching text)
        # Negative image is offset by +3 to create a non-matching pair
        return (
            self.x_data[index],
            self.x_data[(index + 3) % self.length],
            self.y_data[index]
        )

    def __len__(self):
        return self.length


batch_size   = 32
trainingdata = RetrievalDataset()
print("#Training dataset size:", len(trainingdata.x_data))
train_loader = DataLoader(trainingdata, batch_size=batch_size, shuffle=True)


# ── Model ────────────────────────────────────────────────────────────────────
class CrossModalNet(nn.Module):
    """Projects image (2048-dim) and text (200-dim) into a shared 128-dim embedding space."""

    def __init__(self):
        super(CrossModalNet, self).__init__()
        embed_size = 128
        self.img_fc  = nn.Linear(2048, embed_size)  # Image branch: 2048 -> 128
        self.text_fc = nn.Linear(200,  embed_size)  # Text branch:  200  -> 128

    def forward(self, pic, word):
        # pic:  (batch_size, 2048)
        # word: (batch_size, 200)
        out_img  = self.img_fc(pic)   # (batch_size, 128)
        out_text = self.text_fc(word) # (batch_size, 128)
        return out_img, out_text


# ── Hinge Loss ───────────────────────────────────────────────────────────────
class HingeLoss(nn.Module):
    """Ranking loss: pushes positive pairs closer and negative pairs further apart."""

    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = torch.tensor(0.2)

    def forward(self, y_pos, y_neg):
        # y_pos: cosine similarity of (positive image, text)
        # y_neg: cosine similarity of (negative image, text)
        # Target: y_pos >> y_neg; loss = max(0, margin - (y_pos - y_neg))
        zero = Variable(torch.tensor(0.0), requires_grad=True)
        return max(zero, -(y_pos - (self.margin + y_neg)))


# ── Training ─────────────────────────────────────────────────────────────────
loss_function = HingeLoss()
trainnet      = CrossModalNet()
print("Model:", trainnet)

learning_rate = 1e-2
epochs        = 10
optimizer     = optim.SGD(trainnet.parameters(), lr=learning_rate)

best_acc = -1000.0
print("#Batches per epoch:", len(train_loader))

import os
os.makedirs("./models", exist_ok=True)

for epoch in range(epochs):
    running_loss = torch.tensor(0.0)
    running_acc  = 0.0

    for index, data in enumerate(train_loader):
        right, wrong, labels = data
        bs = right.size(0)

        right  = Variable(right,  requires_grad=True)
        wrong  = Variable(wrong,  requires_grad=True)
        labels = Variable(labels, requires_grad=True)

        optimizer.zero_grad()

        # Forward pass — positive pair
        t1, t2 = trainnet(right, labels)
        # Forward pass — negative pair (wrong image, same text)
        t3, t4 = trainnet(wrong, labels)

        # Cosine similarities
        y_pos = torch.cosine_similarity(t1, t2, dim=1)  # (batch_size,)
        y_neg = torch.cosine_similarity(t3, t4, dim=1)  # (batch_size,)

        loss = loss_function(y_pos.sum() / bs, y_neg.sum() / bs)
        loss.backward()
        running_loss += loss * bs
        optimizer.step()

    print("Epoch [{}/{}]  Loss: {:.6f}".format(epoch + 1, epochs, running_loss / len(pic_arr)))

    # Evaluation on test set
    for k in range(len(test_pic_arr)):
        temp1, temp2 = trainnet(test_pic_arr[k], test_word_arr[k])
        temp3, temp4 = trainnet(test_pic_arr[(k + 3) % len(test_pic_arr)], test_word_arr[k])

        y_pos = torch.cosine_similarity(temp1, temp2, dim=0)
        y_neg = torch.cosine_similarity(temp3, temp4, dim=0)

        if y_pos > y_neg:
            running_acc += 1.0

    acc = running_acc / len(test_pic_arr)
    print("Epoch [{}/{}]  Test Accuracy: {:.4f}".format(epoch + 1, epochs, acc))

    if running_acc > best_acc:
        best_acc = running_acc
        torch.save(trainnet, save_model_path + ".best.pkl")

torch.save(trainnet, save_model_path)
