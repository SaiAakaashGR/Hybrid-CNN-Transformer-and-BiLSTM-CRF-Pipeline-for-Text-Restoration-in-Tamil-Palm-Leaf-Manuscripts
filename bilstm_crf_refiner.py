"""
BiLSTM-CRF Refinement Module for Tamil Diacritic & Space Restoration
=====================================================================
Restores diacritical marks (dots / pulli) and word-boundary spaces to
continuous Tamil character strings produced by OCR of palm-leaf manuscripts.

Pipeline
--------
Raw Tamil text  →  TamilProcessor (char→index)
                →  Embedding layer
                →  BiLSTM (2-layer, bidirectional)
                →  Linear projection  →  emission scores  (seq_len × 3)
                →  CRF layer (Viterbi decode during inference)
                →  Label sequence  →  reconstruct formatted text

Labels
------
  0 — no change
  1 — insert space after this character
  2 — insert dot (pulli / virama) on this character  e.g. "க" → "க்"

Hyperparameters (from paper)
----------------------------
  embedding_dim  = 64
  hidden_dim     = 128   (64 per direction × 2)
  dropout        = 0.3
  batch_size     = 32
  learning_rate  = 0.001
  lr_patience    = 5   (ReduceLROnPlateau halves LR)
  early_stop     = 10  (epochs without val-loss improvement)
  grad_clip      = 1.0
"""

import re
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional

# ─────────────────────────────────────────────
#  Constants / Label map
# ─────────────────────────────────────────────
PAD_IDX   = 0          # reserved padding index in vocabulary
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

NO_CHANGE    = 0
INSERT_SPACE = 1
INSERT_DOT   = 2
NUM_TAGS     = 3

# Unicode virama (pulli) used to form a dot/half-consonant in Tamil
TAMIL_VIRAMA = "\u0bcd"   # ்


# =============================================================================
# 1.  TamilProcessor — character ↔ integer vocabulary
# =============================================================================

class TamilProcessor:
    """
    Builds a character-level vocabulary from a corpus and provides
    bidirectional char↔index conversions.

    Special tokens
    --------------
    <PAD>  index 0  — padding
    <UNK>  index 1  — unknown character at inference time
    """

    def __init__(self):
        self.char2idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2char: Dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self._next_idx = 2

    # ── Vocabulary construction ──────────────────────────────────────────

    def build_vocab(self, texts: List[str]) -> None:
        """Scan *texts* and register every new character."""
        for text in texts:
            for ch in text:
                if ch not in self.char2idx:
                    self.char2idx[ch] = self._next_idx
                    self.idx2char[self._next_idx] = ch
                    self._next_idx += 1

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    # ── Encoding / decoding ──────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Convert a character string to a list of integer indices."""
        return [self.char2idx.get(ch, self.char2idx[UNK_TOKEN]) for ch in text]

    def decode(self, indices: List[int]) -> str:
        """Convert integer indices back to a character string."""
        return "".join(self.idx2char.get(i, UNK_TOKEN) for i in indices)

    # ── Synthetic data helpers ───────────────────────────────────────────

    @staticmethod
    def strip_spaces_and_dots(text: str) -> str:
        """
        Programmatically remove spaces and virama diacritics to simulate
        the raw, unprocessed appearance of palm-leaf OCR output.
        """
        text = re.sub(r"\s+", "", text)
        text = text.replace(TAMIL_VIRAMA, "")
        return text

    @staticmethod
    def build_labels(raw: str, ground_truth: str) -> List[int]:
        """
        Align *raw* (stripped) characters against *ground_truth* and
        produce a label sequence of length len(raw).

        Algorithm
        ---------
        Walk through ground_truth; whenever a space is encountered emit
        INSERT_SPACE for the preceding raw character; whenever a virama
        is encountered emit INSERT_DOT for the preceding raw character;
        otherwise emit NO_CHANGE.
        """
        labels: List[int] = []
        gt_iter = iter(ground_truth)
        pending_label = NO_CHANGE

        for gt_ch in gt_iter:
            if gt_ch == " ":
                # mark the *last* appended character
                if labels:
                    labels[-1] = INSERT_SPACE
            elif gt_ch == TAMIL_VIRAMA:
                if labels:
                    labels[-1] = INSERT_DOT
            else:
                labels.append(pending_label)
                pending_label = NO_CHANGE

        # Ensure length parity (safety)
        if len(labels) < len(raw):
            labels += [NO_CHANGE] * (len(raw) - len(labels))
        return labels[:len(raw)]

    # ── Inference-time sanitisation ──────────────────────────────────────

    @staticmethod
    def sanitize_input(text: str) -> str:
        """Strip residual spaces and diacritics (OCR simulation)."""
        return TamilProcessor.strip_spaces_and_dots(text)

    # ── Output reconstruction ────────────────────────────────────────────

    @staticmethod
    def reconstruct(chars: List[str], labels: List[int]) -> str:
        """
        Rebuild formatted Tamil text from characters and predicted labels.

          label 0 → append char as-is
          label 1 → append char + space
          label 2 → append char + VIRAMA  (e.g. "க" → "க்")
        """
        result = []
        for ch, lbl in zip(chars, labels):
            result.append(ch)
            if lbl == INSERT_SPACE:
                result.append(" ")
            elif lbl == INSERT_DOT:
                result.append(TAMIL_VIRAMA)
        return "".join(result)


# =============================================================================
# 2.  Dataset & DataLoader
# =============================================================================

class TamilSequenceDataset(Dataset):
    """
    Holds parallel (input_indices, label_indices) pairs.

    Parameters
    ----------
    raw_texts    : List of raw (stripped) Tamil strings
    label_seqs   : Corresponding label lists  (0 / 1 / 2 per character)
    processor    : Fitted TamilProcessor instance
    """

    def __init__(self,
                 raw_texts: List[str],
                 label_seqs: List[List[int]],
                 processor: TamilProcessor):
        assert len(raw_texts) == len(label_seqs)
        self.samples = [
            (torch.tensor(processor.encode(raw), dtype=torch.long),
             torch.tensor(lbl,                   dtype=torch.long))
            for raw, lbl in zip(raw_texts, label_seqs)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Custom collation:
      • Sort by descending sequence length (required by pack_padded_sequence)
      • Pad inputs and targets to the longest sequence in the batch
      • Build a boolean mask (True = valid, False = padding)
    """
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    inputs, targets = zip(*batch)

    lengths = torch.tensor([len(s) for s in inputs], dtype=torch.long)
    max_len = lengths[0].item()

    padded_inputs  = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.full((len(inputs), max_len), fill_value=-1,
                                dtype=torch.long)      # -1 = ignore in loss
    mask = torch.zeros(len(inputs), max_len, dtype=torch.bool)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = len(inp)
        padded_inputs[i,  :L] = inp
        padded_targets[i, :L] = tgt
        mask[i, :L] = True

    return padded_inputs, padded_targets, lengths, mask


# =============================================================================
# 3.  CRF Layer
# =============================================================================

class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field.

    Learns a (num_tags × num_tags) transition matrix T where T[i,j] is
    the score of transitioning from tag i to tag j.

    Special START and END tags are added internally.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # Transition matrix:  transitions[i, j] = score(i → j)
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        # Scores into START and out of END are not allowed
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions   = nn.Parameter(torch.empty(num_tags))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.uniform_(self.transitions,       -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions,   -0.1, 0.1)

    # ── Forward algorithm (log partition function) ───────────────────────

    def _forward_alg(self,
                     emissions: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
        """
        Compute log Z (denominator of CRF loss) using the forward algorithm.

        Parameters
        ----------
        emissions : (batch, seq_len, num_tags)
        mask      : (batch, seq_len)  bool

        Returns
        -------
        log_Z : (batch,)
        """
        batch, seq_len, _ = emissions.shape
        # initialise: score of starting at each tag
        score = self.start_transitions + emissions[:, 0, :]   # (B, T)

        for t in range(1, seq_len):
            # broadcast: (B, T, 1) + (1, T, T) + (B, 1, T)
            broadcast_score = score.unsqueeze(2)                   # (B, T, 1)
            broadcast_emit  = emissions[:, t, :].unsqueeze(1)      # (B, 1, T)
            next_score = broadcast_score + self.transitions + broadcast_emit
            next_score = torch.logsumexp(next_score, dim=1)        # (B, T)

            # Apply mask: keep previous score for padded positions
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)

        score = score + self.end_transitions   # (B, T)
        return torch.logsumexp(score, dim=1)   # (B,)

    # ── Score of the gold sequence (numerator) ───────────────────────────

    def _score_sentence(self,
                        emissions: torch.Tensor,
                        tags: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the score of the ground-truth label sequence.

        Equation 2:  score(x,y) = Σ emission[t,y_t] + Σ T[y_{t-1}, y_t]

        Parameters
        ----------
        emissions : (batch, seq_len, num_tags)
        tags      : (batch, seq_len)   long
        mask      : (batch, seq_len)   bool

        Returns
        -------
        score : (batch,)
        """
        batch, seq_len, _ = emissions.shape

        score = self.start_transitions[tags[:, 0]]             # (B,)
        score = score + emissions[:, 0, :].gather(
            1, tags[:, 0].unsqueeze(1)).squeeze(1)             # (B,)

        for t in range(1, seq_len):
            trans  = self.transitions[tags[:, t-1], tags[:, t]]  # (B,)
            emit   = emissions[:, t, :].gather(
                1, tags[:, t].unsqueeze(1)).squeeze(1)            # (B,)
            score += (trans + emit) * mask[:, t].float()

        # Add end transition for the last valid tag
        seq_ends = mask.long().sum(dim=1) - 1                    # (B,)
        last_tags = tags.gather(1, seq_ends.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score   # (B,)

    # ── Viterbi decoding ─────────────────────────────────────────────────

    def viterbi_decode(self,
                       emissions: torch.Tensor,
                       mask: torch.Tensor) -> List[List[int]]:
        """
        Equation 4:  δ_t(j) = max_{y_{1:t-1}}  score(y_{1:t-1}, j | x)

        Returns the most probable tag sequence for each item in the batch.

        Parameters
        ----------
        emissions : (batch, seq_len, num_tags)
        mask      : (batch, seq_len)  bool

        Returns
        -------
        List of predicted tag sequences (variable length, no padding)
        """
        batch, seq_len, _ = emissions.shape
        viterbi_score = self.start_transitions + emissions[:, 0, :]   # (B, T)
        history: List[torch.Tensor] = []

        for t in range(1, seq_len):
            broadcast = viterbi_score.unsqueeze(2)                    # (B, T, 1)
            trans_score = broadcast + self.transitions                 # (B, T, T)
            best_score, best_tag = trans_score.max(dim=1)             # (B, T)
            next_score = best_score + emissions[:, t, :]              # (B, T)

            viterbi_score = torch.where(
                mask[:, t].unsqueeze(1), next_score, viterbi_score)
            history.append(best_tag)

        # Add end transition
        viterbi_score += self.end_transitions
        _, best_last_tag = viterbi_score.max(dim=1)   # (B,)

        # Backtrack
        seq_lengths = mask.long().sum(dim=1)           # (B,)
        best_paths  = []
        for b in range(batch):
            L    = seq_lengths[b].item()
            path = [best_last_tag[b].item()]
            for hist in reversed(history[:L-1]):
                path.append(hist[b, path[-1]].item())
            path.reverse()
            best_paths.append(path)

        return best_paths

    # ── Loss (Equation 3) ────────────────────────────────────────────────

    def forward(self,
                emissions: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        CRF negative log-likelihood loss.

        L = log Z(x) - score(x, y*)   (averaged over batch)

        Parameters
        ----------
        emissions : (batch, seq_len, num_tags)
        tags      : (batch, seq_len)  long  — ground-truth labels (-1 = pad)
        mask      : (batch, seq_len)  bool

        Returns
        -------
        loss : scalar tensor
        """
        # Replace padding marker -1 with 0 to avoid index errors
        safe_tags = tags.clone()
        safe_tags[safe_tags < 0] = 0

        log_z         = self._forward_alg(emissions, mask)
        gold_score    = self._score_sentence(emissions, safe_tags, mask)
        nll           = (log_z - gold_score).mean()
        return nll


# =============================================================================
# 4.  BiLSTM-CRF Model
# =============================================================================

class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF sequence labeller for Tamil diacritic & space restoration.

    Parameters
    ----------
    vocab_size    : Number of unique characters (including PAD, UNK)
    embedding_dim : Character embedding size           (paper: 64)
    hidden_dim    : Total BiLSTM hidden size           (paper: 128 = 64×2)
    num_tags      : Number of output labels            (3)
    num_layers    : Number of stacked BiLSTM layers    (2)
    dropout       : Dropout rate between LSTM layers   (paper: 0.3)
    pad_idx       : Padding index in the vocabulary    (0)
    """

    def __init__(self,
                 vocab_size:    int,
                 embedding_dim: int   = 64,
                 hidden_dim:    int   = 128,
                 num_tags:      int   = NUM_TAGS,
                 num_layers:    int   = 2,
                 dropout:       float = 0.3,
                 pad_idx:       int   = PAD_IDX):
        super().__init__()

        assert hidden_dim % 2 == 0, "hidden_dim must be even (split across directions)"

        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      padding_idx=pad_idx)
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,    # each direction gets hidden_dim/2
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout   = nn.Dropout(p=dropout)
        # Linear: hidden_dim → num_tags  (emission scores)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf        = CRF(num_tags)

    # ── Emission score computation ───────────────────────────────────────

    def _get_emissions(self,
                       input_ids: torch.Tensor,
                       lengths: torch.Tensor) -> torch.Tensor:
        """
        Embedding → BiLSTM → Linear  →  emission scores.

        Parameters
        ----------
        input_ids : (batch, seq_len)   long
        lengths   : (batch,)           long — actual (unpadded) lengths

        Returns
        -------
        emissions : (batch, seq_len, num_tags)
        """
        embeds = self.dropout(self.embedding(input_ids))   # (B, L, E)

        packed = pack_padded_sequence(embeds, lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=True)
        lstm_out, _ = self.bilstm(packed)                  # packed
        lstm_out, _ = pad_packed_sequence(lstm_out,
                                          batch_first=True) # (B, L, H)
        lstm_out    = self.dropout(lstm_out)

        emissions = self.hidden2tag(lstm_out)               # (B, L, num_tags)
        return emissions

    # ── Training forward (returns loss) ─────────────────────────────────

    def forward(self,
                input_ids: torch.Tensor,
                tags:      torch.Tensor,
                lengths:   torch.Tensor,
                mask:      torch.Tensor) -> torch.Tensor:
        """
        Compute CRF NLL loss.

        Parameters
        ----------
        input_ids : (batch, seq_len)
        tags      : (batch, seq_len)   ground-truth labels
        lengths   : (batch,)
        mask      : (batch, seq_len)   bool

        Returns
        -------
        loss : scalar tensor
        """
        emissions = self._get_emissions(input_ids, lengths)
        return self.crf(emissions, tags, mask)

    # ── Inference (returns predicted tag sequences) ──────────────────────

    @torch.no_grad()
    def predict(self,
                input_ids: torch.Tensor,
                lengths:   torch.Tensor,
                mask:      torch.Tensor) -> List[List[int]]:
        """
        Viterbi decode → list of predicted tag sequences.
        """
        emissions = self._get_emissions(input_ids, lengths)
        return self.crf.viterbi_decode(emissions, mask)


# =============================================================================
# 5.  Inference pipeline
# =============================================================================

def restore_text(raw_input: str,
                 model: BiLSTMCRF,
                 processor: TamilProcessor,
                 device: torch.device) -> str:
    """
    End-to-end inference:
      raw Tamil string  →  sanitize  →  encode  →  BiLSTM-CRF  →  reconstruct

    Parameters
    ----------
    raw_input  : Continuous Tamil characters (spaces/virama already absent,
                 OR unseen OCR output — will be sanitised automatically)
    model      : Trained BiLSTMCRF instance
    processor  : Fitted TamilProcessor
    device     : torch device

    Returns
    -------
    Formatted Tamil string with restored spaces and diacritics
    """
    model.eval()

    # Step 1 — sanitise (remove residual spaces/diacritics via regex)
    sanitized = TamilProcessor.sanitize_input(raw_input)
    if not sanitized:
        return ""

    # Step 2 — encode characters to indices
    indices = processor.encode(sanitized)
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(indices)], dtype=torch.long)
    mask    = torch.ones(1, len(indices), dtype=torch.bool).to(device)

    # Step 3 — emission scores via BiLSTM
    # Step 4 — CRF Viterbi decode
    predicted_labels = model.predict(input_tensor, lengths, mask)[0]

    # Step 5 — reconstruct formatted output
    chars = list(sanitized)
    return TamilProcessor.reconstruct(chars, predicted_labels)


# =============================================================================
# 6.  Training loop
# =============================================================================

def train(model:       BiLSTMCRF,
          train_loader: DataLoader,
          val_loader:   DataLoader,
          num_epochs:   int   = 100,
          lr:           float = 0.001,
          grad_clip:    float = 1.0,
          lr_patience:  int   = 5,
          early_stop:   int   = 10,
          device:       Optional[torch.device] = None) -> BiLSTMCRF:
    """
    Full training routine with:
      • Adam optimiser (lr = 0.001)
      • ReduceLROnPlateau scheduler (factor 0.5, patience 5)
      • Gradient clipping (max_norm 1.0)
      • Early stopping (patience 10)
      • Masking of padded positions in loss

    Parameters
    ----------
    model        : BiLSTMCRF to train (moved to *device* internally)
    train_loader : DataLoader using collate_fn
    val_loader   : DataLoader using collate_fn
    num_epochs   : Maximum training epochs
    lr           : Initial learning rate
    grad_clip    : Max gradient norm
    lr_patience  : ReduceLROnPlateau patience (epochs)
    early_stop   : Early stopping patience (epochs)
    device       : torch.device (defaults to CUDA if available)

    Returns
    -------
    Trained model (best weights restored)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, verbose=True)

    best_val_loss   = math.inf
    no_improve_cnt  = 0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):

        # ── Training ──────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for input_ids, tags, lengths, mask in train_loader:
            input_ids = input_ids.to(device)
            tags      = tags.to(device)
            lengths   = lengths.to(device)
            mask      = mask.to(device)

            optimizer.zero_grad()
            loss = model(input_ids, tags, lengths, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, tags, lengths, mask in val_loader:
                input_ids = input_ids.to(device)
                tags      = tags.to(device)
                lengths   = lengths.to(device)
                mask      = mask.to(device)
                val_loss += model(input_ids, tags, lengths, mask).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:4d} | train_loss {train_loss:.4f} "
              f"| val_loss {val_loss:.4f} "
              f"| lr {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(val_loss)

        # ── Early stopping & best-model tracking ──────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            no_improve_cnt = 0
            best_state_dict = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            no_improve_cnt += 1
            if no_improve_cnt >= early_stop:
                print(f"Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {early_stop} epochs).")
                break

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


# =============================================================================
# 7.  Evaluation helpers
# =============================================================================

@torch.no_grad()
def evaluate_accuracy(model:      BiLSTMCRF,
                       loader:    DataLoader,
                       device:    torch.device) -> float:
    """
    Token-level accuracy (ignoring padding positions).
    """
    model.eval()
    correct = total = 0
    for input_ids, tags, lengths, mask in loader:
        input_ids = input_ids.to(device)
        mask      = mask.to(device)
        preds     = model.predict(input_ids, lengths, mask)  # List[List[int]]

        for pred_seq, gold_seq, valid_len in zip(preds, tags, lengths):
            L = valid_len.item()
            for p, g in zip(pred_seq[:L], gold_seq[:L].tolist()):
                if g >= 0:
                    correct += int(p == g)
                    total   += 1

    return correct / total if total > 0 else 0.0


# =============================================================================
# 8.  Smoke test / demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 64)
    print("  BiLSTM-CRF Tamil Refinement Module — Smoke Test")
    print("=" * 64)

    # ── Synthetic Tamil corpus ────────────────────────────────────────
    #   Ground-truth sentences (properly spaced, with virama ்)
    ground_truth_samples = [
        "அவன் வீட்டில் இருக்கிறான்",
        "தமிழ் மொழி மிகவும் இனிமையானது",
        "நான் படிக்கிறேன்",
        "அவள் பாடல் பாடுகிறாள்",
        "குழந்தைகள் விளையாடுகின்றனர்",
    ] * 20   # inflate corpus for demo

    processor = TamilProcessor()
    processor.build_vocab(ground_truth_samples)

    raw_texts  = [TamilProcessor.strip_spaces_and_dots(t) for t in ground_truth_samples]
    label_seqs = [TamilProcessor.build_labels(r, g)
                  for r, g in zip(raw_texts, ground_truth_samples)]

    # ── Dataset & DataLoader ──────────────────────────────────────────
    split = int(0.8 * len(raw_texts))
    train_ds = TamilSequenceDataset(raw_texts[:split],  label_seqs[:split],  processor)
    val_ds   = TamilSequenceDataset(raw_texts[split:],  label_seqs[split:],  processor)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              collate_fn=collate_fn)

    print(f"\nVocab size : {processor.vocab_size}")
    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")

    # ── Build model ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BiLSTMCRF(
        vocab_size    = processor.vocab_size,
        embedding_dim = 64,
        hidden_dim    = 128,
        num_tags      = NUM_TAGS,
        num_layers    = 2,
        dropout       = 0.3
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {n_params:,}")

    # ── Quick training (5 epochs for demo) ───────────────────────────
    print("\nTraining (5 demo epochs) …")
    model = train(model, train_loader, val_loader,
                  num_epochs=5, lr=0.001, grad_clip=1.0,
                  lr_patience=5, early_stop=10, device=device)

    # ── Inference demo ────────────────────────────────────────────────
    test_raw = TamilProcessor.strip_spaces_and_dots("அவன் வீட்டில் இருக்கிறான்")
    restored = restore_text(test_raw, model, processor, device)

    print(f"\nInput (raw) : {test_raw}")
    print(f"Restored    : {restored}")
    print(f"Ground truth: அவன் வீட்டில் இருக்கிறான்")

    # ── Token accuracy ────────────────────────────────────────────────
    acc = evaluate_accuracy(model, val_loader, device)
    print(f"\nVal token accuracy (5-epoch demo): {acc:.4f}")
    print("\n✓ Smoke test complete.\n")
    print(model)