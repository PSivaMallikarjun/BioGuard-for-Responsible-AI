"""
BioGuard Training Script
=========================
Trains the Sparse Connectome Guard (SCG) on adversarial prompt detection.

Usage:
  python train.py --topology bio_layered --n_nodes 512 --epochs 20

Key design choices:
  - Binary cross-entropy loss
  - Contrastive pair augmentation (optional)
  - L1 sparsity regularization on edge weights (maintain bio-sparsity)
  - Early stopping on val AUROC
  - Checkpoint saves best model only
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import numpy as np

from scg_model import SparseConnectomeGuard
from dataset   import build_dataloaders


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class BioGuardTrainer:

    def __init__(self, args):
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device: {self.device}")

        # ── Tokenizer ───────────────────────────────────────────────────
        from local_tokenizer import get_tokenizer
        print(f"[Trainer] Loading tokenizer: {args.tokenizer}")
        self.tokenizer = get_tokenizer(args.tokenizer)

        # ── Dataset ─────────────────────────────────────────────────────
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            self.tokenizer,
            use_hf_datasets=args.use_hf,
            custom_csv=args.csv,
            max_length=args.max_len,
            batch_size=args.batch_size,
        )

        # ── Model ────────────────────────────────────────────────────────
        self.model = SparseConnectomeGuard(
            vocab_size  = self.tokenizer.vocab_size,
            embed_dim   = args.embed_dim,
            n_nodes     = args.n_nodes,
            n_layers    = args.n_layers,
            topology    = args.topology,
            avg_degree  = args.avg_degree,
            seed        = args.seed,
        ).to(self.device)

        pc = self.model.count_parameters()
        print(f"[Trainer] Model params → total: {pc['total']:,} | trainable: {pc['trainable']:,}")

        # ── Optimizer & scheduler ────────────────────────────────────────
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.criterion = nn.BCEWithLogitsLoss()

        # ── Output dir ───────────────────────────────────────────────────
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.best_auroc   = 0.0
        self.patience_ctr = 0
        self.history      = []

    # ── Training one epoch ──────────────────────────────────────────────

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss, all_logits, all_labels = 0.0, [], []

        for batch in self.train_loader:
            input_ids  = batch["input_ids"].to(self.device)
            attn_mask  = batch["attention_mask"].to(self.device)
            labels     = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attn_mask)
            loss   = self.criterion(logits, labels)

            # L1 sparsity regularization on edge weights (preserve bio-sparsity)
            l1_reg = sum(
                layer.edge_weight.abs().mean()
                for layer in self.model.scg_layers
            )
            loss = loss + self.args.l1_lambda * l1_reg

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

        logits_cat = torch.cat(all_logits).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        preds      = (logits_cat > 0).astype(int)

        return {
            "loss":  total_loss / len(self.train_loader),
            "auroc": roc_auc_score(labels_cat, logits_cat) if len(np.unique(labels_cat)) > 1 else 0.5,
            "f1":    f1_score(labels_cat, preds, zero_division=0),
        }

    # ── Evaluation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader, split_name: str = "val") -> dict:
        self.model.eval()
        total_loss, all_logits, all_labels = 0.0, [], []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            labels    = batch["label"].to(self.device)

            logits    = self.model(input_ids, attn_mask)
            loss      = self.criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        logits_cat = torch.cat(all_logits).numpy()
        labels_cat = torch.cat(all_labels).numpy()
        preds      = (logits_cat > 0).astype(int)

        auroc = roc_auc_score(labels_cat, logits_cat) if len(np.unique(labels_cat)) > 1 else 0.5
        f1    = f1_score(labels_cat, preds, zero_division=0)

        return {
            "loss":   total_loss / max(len(loader), 1),
            "auroc":  auroc,
            "f1":     f1,
            "preds":  preds,
            "labels": labels_cat,
        }

    # ── Main training loop ───────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"  BioGuard SCG Training  |  Topology: {self.args.topology}")
        print(f"  Nodes: {self.args.n_nodes}  |  Layers: {self.args.n_layers}  |  Epochs: {self.args.epochs}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.args.epochs + 1):
            t0 = time.time()
            train_m = self.train_epoch()
            val_m   = self.evaluate(self.val_loader, "val")
            self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{self.args.epochs}  "
                f"| train_loss: {train_m['loss']:.4f}  train_auroc: {train_m['auroc']:.4f}  "
                f"| val_loss: {val_m['loss']:.4f}  val_auroc: {val_m['auroc']:.4f}  "
                f"val_f1: {val_m['f1']:.4f}  [{elapsed:.1f}s]"
            )

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
                                    **{f"val_{k}": v for k, v in val_m.items() if k not in ("preds","labels")}}
            self.history.append(row)

            # ── Checkpoint best model ────────────────────────────────────
            if val_m["auroc"] > self.best_auroc:
                self.best_auroc = val_m["auroc"]
                self.patience_ctr = 0
                ckpt_path = self.out_dir / "best_scg.pt"
                torch.save({
                    "epoch":      epoch,
                    "model_state": self.model.state_dict(),
                    "val_auroc":  val_m["auroc"],
                    "args":       vars(self.args),
                }, ckpt_path)
                print(f"  OK New best val AUROC: {self.best_auroc:.4f} -> saved {ckpt_path}")
            else:
                self.patience_ctr += 1
                if self.patience_ctr >= self.args.patience:
                    print(f"\n[EarlyStop] No improvement for {self.args.patience} epochs. Stopping.")
                    break

        # ── Final test evaluation ────────────────────────────────────────
        print(f"\n{'-'*60}")
        print("Loading best checkpoint for test evaluation...")
        ckpt = torch.load(self.out_dir / "best_scg.pt", map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])

        test_m = self.evaluate(self.test_loader, "test")
        print(f"\nTest Results:")
        print(f"  AUROC : {test_m['auroc']:.4f}")
        print(f"  F1    : {test_m['f1']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(test_m["labels"], test_m["preds"],
                                     target_names=["Benign", "Adversarial"]))

        # Save history + results
        with open(self.out_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        results = {"test_auroc": test_m["auroc"], "test_f1": test_m["f1"],
                   "best_val_auroc": self.best_auroc, "args": vars(self.args)}
        with open(self.out_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nOK Training complete. Outputs saved to: {self.out_dir}/")
        return results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train BioGuard SCG")

    # Model
    p.add_argument("--n_nodes",    type=int,   default=512,          help="Number of SCG graph nodes")
    p.add_argument("--n_layers",   type=int,   default=3,            help="Number of SCG message-passing layers")
    p.add_argument("--embed_dim",  type=int,   default=64,           help="Node feature dimension")
    p.add_argument("--topology",   type=str,   default="bio_layered",
                   choices=["random_sparse", "scale_free", "bio_layered"])
    p.add_argument("--avg_degree", type=int,   default=5,            help="Avg degree (random_sparse only)")
    p.add_argument("--seed",       type=int,   default=42)

    # Data
    p.add_argument("--tokenizer",  type=str,   default="local",      help="Tokenizer name; use 'local' for no-HF mode")
    p.add_argument("--max_len",    type=int,   default=128,          help="Max token sequence length")
    p.add_argument("--use_hf",     action="store_true", default=False, help="Pull datasets from HuggingFace")
    p.add_argument("--csv",        type=str,   default=None,         help="Optional custom CSV dataset path")

    # Training
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--l1_lambda",  type=float, default=1e-4,         help="L1 sparsity regularization weight")
    p.add_argument("--patience",   type=int,   default=5,            help="Early stopping patience (epochs)")
    p.add_argument("--out_dir",    type=str,   default="outputs/")

    return p.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    trainer = BioGuardTrainer(args)
    trainer.train()
