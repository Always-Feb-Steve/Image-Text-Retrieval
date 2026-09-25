# Train a cross-modal ranking model (image-text retrieval) using hinge loss
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Architecture options (from experimentation, see README "Results") ─────────
# linear (default):   (2048 -> embed)        —  (200 -> embed)
# mlp (Variant 2):    (2048 -> 512 -> embed) —  (200 -> 512 -> embed)
# ─────────────────────────────────────────────────────────────────────────────

# ── Paths (update these to match your environment) ──────────────────────────
feature_folder  = "./features/"
save_model_path = "./models/net.pt"
# ─────────────────────────────────────────────────────────────────────────────


def load_split(split):
    """Image features (N, 2048), caption features (M, 200) and, per caption, the row of its image."""
    img = torch.from_numpy(np.load(os.path.join(feature_folder, f"img_{split}.npy")))
    txt = torch.from_numpy(np.load(os.path.join(feature_folder, f"txt_{split}.npy")))
    txt_img = torch.from_numpy(np.load(os.path.join(feature_folder, f"txt_{split}_img.npy"))).long()
    return img, txt, txt_img


# ── Model ────────────────────────────────────────────────────────────────────
class CrossModalNet(nn.Module):
    """Projects image (2048-dim) and text (200-dim) into a shared embedding space."""

    def __init__(self, arch="mlp", embed_size=128, hidden=512, dropout=0.2):
        super(CrossModalNet, self).__init__()

        def branch(in_dim):
            if arch == "linear":
                return nn.Linear(in_dim, embed_size)
            return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, embed_size))

        self.img_fc  = branch(2048)  # Image branch
        self.text_fc = branch(200)   # Text branch

    def encode_image(self, pic):
        return F.normalize(self.img_fc(F.normalize(pic, dim=-1)), dim=-1)

    def encode_text(self, word):
        return F.normalize(self.text_fc(F.normalize(word, dim=-1)), dim=-1)

    def forward(self, pic, word):
        # pic:  (batch_size, 2048)   word: (batch_size, 200)
        # returns unit-length embeddings, so a dot product is the cosine similarity
        return self.encode_image(pic), self.encode_text(word)


# ── Hinge Loss ───────────────────────────────────────────────────────────────
class HingeLoss(nn.Module):
    """Bidirectional ranking loss over all in-batch negatives.

    For every matching (image, caption) pair, each other image in the batch is a negative
    for the caption and each other caption is a negative for the image:
        loss = sum max(0, margin - s(pos) + s(neg))
    Pairs that share the same image (Flickr8k has 5 captions per image) are not negatives.
    """

    def __init__(self, margin=0.2, hard_negative=False):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.hard_negative = hard_negative  # VSE++: only the hardest negative per query

    def forward(self, img_emb, txt_emb, img_ids):
        scores = img_emb @ txt_emb.t()                   # (B, B), diagonal = positive pairs
        pos = scores.diag().view(-1, 1)
        same_image = img_ids.view(-1, 1) == img_ids.view(1, -1)

        cost_txt = (self.margin + scores - pos).clamp(min=0).masked_fill(same_image, 0)      # image -> wrong captions
        cost_img = (self.margin + scores - pos.t()).clamp(min=0).masked_fill(same_image, 0)  # caption -> wrong images
        if self.hard_negative:
            return (cost_txt.max(dim=1)[0] + cost_img.max(dim=0)[0]).mean()
        return (cost_txt.sum(dim=1) + cost_img.sum(dim=0)).mean()


# ── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(net, img, txt, txt_img, ks=(1, 5, 10)):
    """Recall@K for text->image (each caption finds its image among all images of the split)
    and image->text (each image finds any of its captions among all captions)."""
    net.eval()
    img_emb, txt_emb = net(img, txt)
    sims = txt_emb @ img_emb.t()                                        # (captions, images)

    # text -> image: rank of the correct image for each caption
    # ties count against the model (a collapsed model with identical scores must not look good)
    correct = sims[torch.arange(len(txt)), txt_img].view(-1, 1)
    t2i_rank = (sims >= correct).sum(dim=1) - 1

    # image -> text: best rank among the image's own captions
    order = sims.t().argsort(dim=1, descending=True)                   # (images, captions)
    is_own = txt_img[order] == torch.arange(len(img)).view(-1, 1)
    i2t_rank = is_own.float().argmax(dim=1)

    result = {}
    for k in ks:
        result[f"t2i_R@{k}"] = (t2i_rank < k).float().mean().item() * 100
        result[f"i2t_R@{k}"] = (i2t_rank < k).float().mean().item() * 100
    result["rsum"] = sum(result.values())
    net.train()
    return result


def fmt(r):
    return "text->image R@1/5/10: {:.1f}/{:.1f}/{:.1f}   image->text R@1/5/10: {:.1f}/{:.1f}/{:.1f}".format(
        r["t2i_R@1"], r["t2i_R@5"], r["t2i_R@10"], r["i2t_R@1"], r["i2t_R@5"], r["i2t_R@10"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the image-text ranking model")
    parser.add_argument("--arch", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--embed_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--loss", choices=["hard", "sum"], default="hard",
                        help="hard: VSE++ (hardest in-batch negative only); sum: sum over all in-batch negatives")
    parser.add_argument("--warmup_epochs", type=int, default=0,
                        help="with --loss hard: use the sum loss for the first N epochs")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--save_model_path", default=save_model_path)
    args = parser.parse_args()
    args.hard_negative = args.loss == "hard"
    torch.manual_seed(0)

    # ── Load data ────────────────────────────────────────────────────────────
    train_img, train_txt, train_txt_img = load_split("train")
    val = load_split("val")
    test = load_split("test")
    print("#Training images: {}  captions: {}".format(len(train_img), len(train_txt)))
    print("#Val images: {}  #Test images: {}".format(len(val[0]), len(test[0])))

    # Each training sample is one (caption, its image) pair
    train_loader = DataLoader(TensorDataset(train_txt, train_txt_img), batch_size=args.batch_size,
                              shuffle=True, drop_last=True)

    # ── Training ─────────────────────────────────────────────────────────────
    net = CrossModalNet(arch=args.arch, embed_size=args.embed_size)
    loss_function = HingeLoss(margin=args.margin, hard_negative=args.hard_negative)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print("Model:", net)

    os.makedirs(os.path.dirname(args.save_model_path) or ".", exist_ok=True)
    best = {"rsum": -1}
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        running_loss = 0.0
        # hardest-negative training can collapse from a random start, so it may warm up with the sum loss
        loss_function.hard_negative = args.hard_negative and epoch >= args.warmup_epochs
        for words, img_ids in train_loader:
            img_emb, txt_emb = net(train_img[img_ids], words)
            loss = loss_function(img_emb, txt_emb, img_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        r = evaluate(net, *val)
        print("Epoch [{}/{}]  loss {:.4f}  val {}  ({:.1f}s)".format(
            epoch + 1, args.epochs, running_loss / len(train_loader), fmt(r), time.perf_counter() - t0))

        # Model selection on the validation split; the test split is only used once at the end
        if r["rsum"] > best["rsum"]:
            best = dict(r, epoch=epoch + 1)
            torch.save({"state_dict": net.state_dict(), "config": {"arch": args.arch, "embed_size": args.embed_size}},
                       args.save_model_path)

    # ── Final test evaluation with the best checkpoint ───────────────────────
    checkpoint = torch.load(args.save_model_path)
    net.load_state_dict(checkpoint["state_dict"])
    test_result = evaluate(net, *test)
    print("\nBest epoch {} (val rsum {:.1f})".format(best["epoch"], best["rsum"]))
    print("TEST (1,000 images / 5,000 captions): " + fmt(test_result))

    with open(os.path.splitext(args.save_model_path)[0] + "_results.json", "w") as fp:
        json.dump({"args": vars(args), "best_epoch": best["epoch"], "val": best, "test": test_result}, fp, indent=2)
