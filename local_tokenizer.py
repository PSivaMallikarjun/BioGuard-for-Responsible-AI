"""
Lightweight Local Tokenizer for BioGuard
==========================================
Character-level BPE-lite tokenizer — no HuggingFace dependency.
Vocab is built from a character set + common subwords.
Works out-of-the-box with no internet access.
"""

import re
from typing import List


class LocalTokenizer:
    """
    Simple character-level tokenizer with a fixed vocab.
    Sufficient for jailbreak detection (pattern-level signal).

    Compatible with: tokenizer(text, ...) and AutoTokenizer interface.
    """

    # Special tokens
    PAD_ID  = 0
    UNK_ID  = 1
    BOS_ID  = 2
    EOS_ID  = 3

    def __init__(self, max_vocab: int = 512):
        # Build vocab: printable ASCII + special tokens
        special = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
        chars   = list("abcdefghijklmnopqrstuvwxyz"
                       "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                       "0123456789"
                       " .,!?;:'\"()-_/\\@#$%^&*+=<>[]{}|`~\n\t")
        # Common adversarial subwords (improve signal for jailbreaks)
        subwords = [
            "ignore", "previous", "instructions", "override", "system",
            "bypass", "restrict", "DAN", "jailbreak", "roleplay", "pretend",
            "forget", "without", "filter", "unrestricted", "anything",
            "always", "never", "must", "prompt", "admin", "mode",
            "activated", "disabled", "safety", "protocol",
        ]
        vocab_list = special + chars + subwords
        self.vocab      = {tok: idx for idx, tok in enumerate(vocab_list)}
        self.inv_vocab  = {idx: tok for tok, idx in self.vocab.items()}
        self._vocab_size = len(self.vocab)
        self.pad_token  = "[PAD]"
        self.eos_token  = "[EOS]"

    @property
    def vocab_size(self):
        return self._vocab_size

    def tokenize(self, text: str) -> List[int]:
        """Greedy longest-match tokenization."""
        ids   = [self.BOS_ID]
        i     = 0
        text  = text[:1024]    # hard cap
        while i < len(text):
            matched = False
            # Try subwords (longest first, len > 1)
            for length in range(min(20, len(text) - i), 1, -1):
                sub = text[i:i+length]
                if sub in self.vocab:
                    ids.append(self.vocab[sub])
                    i += length
                    matched = True
                    break
            if not matched:
                ch = text[i]
                ids.append(self.vocab.get(ch, self.UNK_ID))
                i += 1
        ids.append(self.EOS_ID)
        return ids

    def __call__(
        self,
        text,
        max_length:      int  = 128,
        padding:         str  = "max_length",
        truncation:      bool = True,
        return_tensors:  str  = None,
    ):
        import torch

        # Handle both single string and list of strings
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts  = text
            single = False

        all_ids   = []
        all_masks = []

        for t in texts:
            ids = self.tokenize(t)
            if truncation:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding == "max_length":
                pad_len = max_length - len(ids)
                ids  = ids  + [self.PAD_ID] * pad_len
                mask = mask + [0]           * pad_len
            all_ids.append(ids)
            all_masks.append(mask)

        result = {
            "input_ids":      torch.tensor(all_ids,   dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }
        if single:
            result = {k: v for k, v in result.items()}   # keep as batch dim 1
        return result


def get_tokenizer(name: str = "local"):
    """
    Returns a tokenizer.
    'local' → our built-in LocalTokenizer (no HF needed)
    Anything else → attempts HuggingFace AutoTokenizer
    """
    if name == "local":
        return LocalTokenizer()
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as e:
        print(f"[Tokenizer] HuggingFace '{name}' unavailable ({e}). Falling back to local tokenizer.")
        return LocalTokenizer()
