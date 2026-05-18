"""Unit tests for the LOM (feature/lom) architecture.

Tests cover four concerns:
  1. Data loading — ControllableGPTDataset/Loader shapes and alignment (synthetic).
  2. Attention mask — _build_lam_attn_mask correctness for all horizons.
  3. Model forward — shapes, [MASK] usage, and loss-target alignment.
  4. Zinc data values — token correctness on the actual zinc dataset.
"""
import math
import tempfile
import os

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from safetensors import safe_open

from interdiff.data.ControllableGPTLoader import ControllableGPTDataset, build_dataloaders
from interdiff.models import LatentActionModel, ControllableGPT


# ---------------------------------------------------------------------------
# Zinc dataset constants (real data)
# ---------------------------------------------------------------------------

ZINC_DATASET = os.path.join(
    os.path.dirname(__file__), "..",
    "interdiff/data/processed/zinc_tok_seqlen_128_vocabsize_50/dataset.safetensors",
)
ZINC_VOCAB_SIZE = 50
ZINC_CTX_LEN = 128
ZINC_PAD_ID = 0
ZINC_MASK_ID = 2
ZINC_BOS_ID = 3
ZINC_EOS_ID = 4

zinc_available = pytest.mark.skipif(
    not os.path.exists(ZINC_DATASET),
    reason="zinc dataset not found — run scripts/tokenise_dataset.py first",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 64
CONTEXT_LEN = 16
N_EMBD = 64
N_LAYER = 2
N_HEAD = 2
NUM_LATENTS = 8
PAD_ID = 0
MASK_ID = 2
BOS_ID = 3
EOS_ID = 4
BATCH_SIZE = 4


def make_lam(horizon: int = -1) -> LatentActionModel:
    return LatentActionModel(
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=0.0,
        context_length=CONTEXT_LEN,
        lm_head_out_size=VOCAB_SIZE,
        latent_action_dim=N_EMBD,
        num_latents=NUM_LATENTS,
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        horizon=horizon,
    )


def make_batch(B: int = BATCH_SIZE, T: int = CONTEXT_LEN) -> dict:
    """Synthetic batch matching ControllableGPTDataset output."""
    x = torch.randint(5, VOCAB_SIZE, (B, T))   # avoid special tokens 0-4
    y = x[:, 1:]                                # (B, T-1)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

class TestDataLoading:

    def test_dataset_length(self):
        n_rows, ctx = 100, CONTEXT_LEN
        data = torch.randint(5, VOCAB_SIZE, (n_rows, ctx))
        ds = ControllableGPTDataset(data)
        assert len(ds) == n_rows

    def test_x_shape(self):
        data = torch.randint(5, VOCAB_SIZE, (50, CONTEXT_LEN))
        ds = ControllableGPTDataset(data)
        item = ds[0]
        assert item["x"].shape == (CONTEXT_LEN,), \
            f"x should be (context_len,), got {item['x'].shape}"

    def test_y_shape(self):
        data = torch.randint(5, VOCAB_SIZE, (50, CONTEXT_LEN))
        ds = ControllableGPTDataset(data)
        item = ds[0]
        assert item["y"].shape == (CONTEXT_LEN - 1,), \
            f"y should be (context_len-1,), got {item['y'].shape}"

    def test_y_is_x_shifted_left(self):
        """y must equal x[1:] — the standard next-token prediction target."""
        data = torch.randint(5, VOCAB_SIZE, (50, CONTEXT_LEN))
        ds = ControllableGPTDataset(data)
        for i in range(min(10, len(ds))):
            item = ds[i]
            assert torch.equal(item["y"], item["x"][1:]), \
                f"y != x[1:] for sample {i}"

    def test_token_ids_in_range(self):
        data = torch.randint(0, VOCAB_SIZE, (50, CONTEXT_LEN))
        ds = ControllableGPTDataset(data)
        for i in range(len(ds)):
            item = ds[i]
            assert item["x"].min() >= 0
            assert item["x"].max() < VOCAB_SIZE

    def test_dataloader_batch_shapes(self):
        """build_dataloaders returns batches with the expected shapes."""
        n_rows = 200
        data = torch.randint(5, VOCAB_SIZE, (n_rows, CONTEXT_LEN))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.safetensors")
            save_file({"tokens": data}, path)
            train_loader, val_loader = build_dataloaders(
                path=path, seed=0, val_ratio=0.1, batch_size=BATCH_SIZE
            )
        batch = next(iter(train_loader))
        assert batch["x"].shape == (BATCH_SIZE, CONTEXT_LEN)
        assert batch["y"].shape == (BATCH_SIZE, CONTEXT_LEN - 1)

    def test_train_val_split_sizes(self):
        """Train and val loaders together cover all rows exactly once."""
        n_rows = 200
        val_ratio = 0.1
        data = torch.randint(5, VOCAB_SIZE, (n_rows, CONTEXT_LEN))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tokens.safetensors")
            save_file({"tokens": data}, path)
            train_loader, val_loader = build_dataloaders(
                path=path, seed=0, val_ratio=val_ratio, batch_size=1, drop_last=False
            )
        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)
        assert n_train + n_val == n_rows
        assert n_val == max(1, round(n_rows * val_ratio))


# ---------------------------------------------------------------------------
# 2. Attention mask
# ---------------------------------------------------------------------------

class TestAttentionMask:

    @pytest.mark.parametrize("T", [3, 5, 8])
    def test_mask_shape(self, T):
        lam = make_lam()
        mask = lam._build_lam_attn_mask(T, torch.device("cpu"))
        assert mask.shape == (2 * T - 1, 2 * T - 1)

    @pytest.mark.parametrize("T", [3, 5, 8])
    def test_action_cannot_see_next_token(self, T):
        """a_t must never attend to x_{t+1}."""
        lam = make_lam(horizon=-1)
        mask = lam._build_lam_attn_mask(T, torch.device("cpu"))
        for t in range(T - 1):
            action_row = 2 * t + 1
            blocked_col = 2 * t + 2      # x_{t+1}
            assert mask[action_row, blocked_col].item() == float("-inf"), \
                f"a_{t} should not see x_{t+1} (T={T})"

    @pytest.mark.parametrize("T", [3, 5, 8])
    def test_action_sees_all_other_tokens(self, T):
        """a_t can attend to every token except x_{t+1}."""
        lam = make_lam(horizon=-1)
        mask = lam._build_lam_attn_mask(T, torch.device("cpu"))
        for t in range(T - 1):
            action_row = 2 * t + 1
            for s in range(T):
                col = 2 * s
                if s == t + 1:
                    continue  # blocked — tested separately
                assert mask[action_row, col].item() == 0.0, \
                    f"a_{t} should see x_{s} (T={T})"

    @pytest.mark.parametrize("T", [3, 5, 8])
    def test_action_sees_other_action_slots(self, T):
        """a_t can attend to all other action slots (full bidirectional over actions)."""
        lam = make_lam(horizon=-1)
        mask = lam._build_lam_attn_mask(T, torch.device("cpu"))
        for t in range(T - 1):
            action_row = 2 * t + 1
            for s in range(T - 1):
                col = 2 * s + 1          # other action slot
                assert mask[action_row, col].item() == 0.0, \
                    f"a_{t} should see a_{s} (T={T})"

    @pytest.mark.parametrize("T", [3, 5, 8])
    def test_token_positions_attend_everywhere(self, T):
        """Token positions (even rows) have unrestricted attention."""
        lam = make_lam(horizon=-1)
        mask = lam._build_lam_attn_mask(T, torch.device("cpu"))
        L = 2 * T - 1
        for t in range(T):
            row = 2 * t
            for col in range(L):
                assert mask[row, col].item() == 0.0, \
                    f"x_{t} should see position {col} (T={T})"

    @pytest.mark.parametrize("k,T", [(0, 5), (1, 5), (2, 6)])
    def test_horizon_blocks_far_future_tokens(self, k, T):
        """With horizon=k, a_t cannot attend to tokens beyond x_{t+k+2}."""
        lam = make_lam(horizon=k)
        mask = lam._build_lam_attn_mask(T, torch.device("cpu"))
        for t in range(T - 1):
            action_row = 2 * t + 1
            first_blocked = 2 * (t + k + 2) + 1
            for pos in range(first_blocked, 2 * T - 1):
                assert mask[action_row, pos].item() == float("-inf"), \
                    f"a_{t} should NOT see position {pos} with horizon={k} (T={T})"

    @pytest.mark.parametrize("k,T", [(0, 5), (1, 5), (2, 6)])
    def test_horizon_allows_window_tokens(self, k, T):
        """With horizon=k, a_t CAN attend to x_{t+2}..x_{t+k+2} and their action slots."""
        lam = make_lam(horizon=k)
        mask = lam._build_lam_attn_mask(T, torch.device("cpu"))
        for t in range(T - 1):
            action_row = 2 * t + 1
            for j in range(t + 2, min(t + k + 3, T)):  # token indices in window
                assert mask[action_row, 2 * j].item() == 0.0, \
                    f"a_{t} should see x_{j} with horizon={k} (T={T})"


# ---------------------------------------------------------------------------
# 3. Model forward — data fed correctly
# ---------------------------------------------------------------------------

class TestModelForward:

    def test_lam_forward_shapes(self):
        """LatentActionModel.forward returns correct shapes."""
        lam = make_lam()
        lam.eval()
        batch = make_batch(B=BATCH_SIZE, T=CONTEXT_LEN)
        x = batch["x"]
        logits, actions, vq_loss_dict = lam(x)

        T = x.shape[1]
        assert logits.shape  == (BATCH_SIZE, T - 1, VOCAB_SIZE), \
            f"logits shape mismatch: {logits.shape}"
        assert actions.shape == (BATCH_SIZE, T - 1, N_EMBD), \
            f"actions shape mismatch: {actions.shape}"
        assert "vq_loss" in vq_loss_dict

    def test_vq_encode_shapes(self):
        """vq_encode returns (z_q, loss_dict, indices) with correct shapes."""
        lam = make_lam()
        lam.eval()
        x = torch.randint(5, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LEN))
        z_q, _, indices = lam.vq_encode(x)

        T = CONTEXT_LEN
        assert z_q.shape    == (BATCH_SIZE, T - 1, N_EMBD)
        assert indices.shape == (BATCH_SIZE, T - 1)

    def test_mask_token_used_as_action_placeholder(self):
        """Action slots are filled with the [MASK] embedding (token id 2),
        not a separate nn.Parameter."""
        lam = make_lam()

        # Confirm there is no 'action_token' parameter
        param_names = [n for n, _ in lam.named_parameters()]
        assert not any("action_token" in n for n in param_names), \
            "action_token parameter should not exist — use [MASK] embedding instead"

        # Confirm mask_token_id is 2
        assert lam.mask_token_id == MASK_ID

        # Confirm the [MASK] embedding is used: intercept wte calls
        seen_ids = []
        original_wte = lam.encoder.wte

        class TrackingEmbedding(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, ids):
                seen_ids.append(ids.detach().cpu())
                return self.inner(ids)

        lam.encoder.wte = TrackingEmbedding(original_wte)
        x = torch.randint(5, VOCAB_SIZE, (2, 8))
        lam.vq_encode(x)

        all_ids = torch.cat([ids.view(-1) for ids in seen_ids])
        assert MASK_ID in all_ids.tolist(), \
            f"[MASK] token (id={MASK_ID}) was never passed to wte during vq_encode"

    def test_lam_ce_loss_target_alignment(self):
        """The lam logits align with y = x[:, 1:] — the standard next-token targets."""
        lam = make_lam()
        lam.eval()
        batch = make_batch(B=BATCH_SIZE, T=CONTEXT_LEN)
        x, y = batch["x"], batch["y"]

        logits, _, _ = lam(x)
        # loss must not crash and must be finite
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            ignore_index=PAD_ID,
        )
        assert torch.isfinite(loss), f"lam_ce_loss is not finite: {loss}"

    def test_controllable_gpt_forward_shapes(self):
        """ControllableGPT.forward returns the three expected tensors."""
        cgpt = ControllableGPT(
            vocab_size=VOCAB_SIZE,
            n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
            dropout=0.0, context_length=CONTEXT_LEN,
            lm_head_out_size=VOCAB_SIZE,
            latent_action_dim=N_EMBD, num_latents=NUM_LATENTS,
            pad_token_id=PAD_ID, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
        )
        cgpt.eval()
        x = make_batch()["x"]
        lam_logits, dm_logits, vq_loss_dict = cgpt(x)

        T = x.shape[1]
        assert lam_logits.shape == (BATCH_SIZE, T - 1, VOCAB_SIZE)
        assert dm_logits.shape  == (BATCH_SIZE, T - 1, VOCAB_SIZE)
        assert torch.isfinite(vq_loss_dict["vq_loss"])

    def test_dynamics_loss_target_alignment(self):
        """The dynamics model logits also align with y = x[:, 1:]."""
        cgpt = ControllableGPT(
            vocab_size=VOCAB_SIZE,
            n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
            dropout=0.0, context_length=CONTEXT_LEN,
            lm_head_out_size=VOCAB_SIZE,
            latent_action_dim=N_EMBD, num_latents=NUM_LATENTS,
            pad_token_id=PAD_ID, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
        )
        cgpt.eval()
        batch = make_batch()
        x, y = batch["x"], batch["y"]
        _, dm_logits, _ = cgpt(x)

        loss = F.cross_entropy(
            dm_logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
            ignore_index=PAD_ID,
        )
        assert torch.isfinite(loss), f"dynamics_loss is not finite: {loss}"

    @pytest.mark.parametrize("horizon", [-1, 0, 2])
    def test_forward_works_for_all_horizons(self, horizon):
        """Model forward should succeed for any supported horizon value."""
        lam = make_lam(horizon=horizon)
        lam.eval()
        x = make_batch()["x"]
        logits, actions, losses = lam(x)
        assert logits.shape == (BATCH_SIZE, CONTEXT_LEN - 1, VOCAB_SIZE)
        assert torch.isfinite(losses["vq_loss"])


# ---------------------------------------------------------------------------
# 4. Gradient flow
# ---------------------------------------------------------------------------

def make_cgpt() -> ControllableGPT:
    return ControllableGPT(
        vocab_size=VOCAB_SIZE,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
        dropout=0.0, context_length=CONTEXT_LEN,
        lm_head_out_size=VOCAB_SIZE,
        latent_action_dim=N_EMBD, num_latents=NUM_LATENTS,
        pad_token_id=PAD_ID, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
    )


class TestGradientFlow:
    """Verify which parameters receive gradients from each loss term.

    Three invariants:
      A. Dynamics model loss does NOT back-propagate into the LAM encoder
         (actions are detached before being fed to the dynamics model).
      B. LAM reconstruction loss DOES reach the LAM encoder
         (straight-through estimator passes gradients through z_q → encoder).
      C. LAM reconstruction loss does NOT reach the VQ codebook
         (STE blocks the recon path to the codebook; codebook learns only
          from the explicit q_loss term in vq_loss_dict).
    """

    def _forward(self, cgpt, x):
        """Run one forward pass and return all outputs."""
        lam_logits, dm_logits, vq_loss_dict = cgpt(x)
        y = x[:, 1:]
        lam_loss = F.cross_entropy(lam_logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
        dm_loss  = F.cross_entropy(dm_logits.reshape(-1, VOCAB_SIZE),  y.reshape(-1))
        return lam_loss, dm_loss, vq_loss_dict

    def test_dynamics_loss_does_not_reach_lam_encoder(self):
        """Backward through dm_loss only — LAM encoder must receive no gradient."""
        cgpt = make_cgpt()
        x = make_batch()["x"]
        _, dm_loss, _ = self._forward(cgpt, x)

        dm_loss.backward()

        for name, p in cgpt.lam.encoder.named_parameters():
            assert p.grad is None or p.grad.abs().max().item() == 0.0, \
                f"lam.encoder.{name} has non-zero grad from dm_loss — detach() missing?"

    def test_lam_recon_loss_reaches_lam_encoder(self):
        """Backward through lam_loss — LAM encoder must receive non-zero gradients."""
        cgpt = make_cgpt()
        x = make_batch()["x"]
        lam_loss, _, _ = self._forward(cgpt, x)

        lam_loss.backward()

        has_grad = False
        for name, p in cgpt.lam.encoder.named_parameters():
            if p.grad is not None and p.grad.abs().max().item() > 0.0:
                has_grad = True
                break
        assert has_grad, \
            "No LAM encoder parameter received a gradient from lam_loss — STE may be broken"

    def test_lam_recon_loss_does_not_reach_codebook(self):
        """Backward through lam_loss only (no vq_loss) — codebook must get no gradient.

        The straight-through estimator copies gradients to the encoder input, not
        to the codebook vectors.  Codebook gradients come exclusively from q_loss.
        """
        cgpt = make_cgpt()
        x = make_batch()["x"]
        lam_loss, _, _ = self._forward(cgpt, x)

        lam_loss.backward()

        cb = cgpt.lam.vq.codebook
        assert cb.grad is None or cb.grad.abs().max().item() == 0.0, \
            "Codebook received gradient from reconstruction loss — STE may not be applied"

    def test_dynamics_loss_reaches_dynamics_model(self):
        """Backward through dm_loss — dynamics model parameters must receive gradients."""
        cgpt = make_cgpt()
        x = make_batch()["x"]
        _, dm_loss, _ = self._forward(cgpt, x)

        dm_loss.backward()

        has_grad = False
        for name, p in cgpt.dynamics_model.named_parameters():
            if p.grad is not None and p.grad.abs().max().item() > 0.0:
                has_grad = True
                break
        assert has_grad, \
            "No dynamics model parameter received a gradient from dm_loss"

    def test_vq_loss_reaches_codebook(self):
        """Backward through vq_loss — codebook must receive non-zero gradients."""
        cgpt = make_cgpt()
        x = make_batch()["x"]
        _, _, vq_loss_dict = self._forward(cgpt, x)

        vq_loss_dict["vq_loss"].backward()

        cb = cgpt.lam.vq.codebook
        assert cb.grad is not None and cb.grad.abs().max().item() > 0.0, \
            "Codebook received no gradient from vq_loss"


# ---------------------------------------------------------------------------
# 5. VQ dropout wiring
# ---------------------------------------------------------------------------

class TestVQDropout:
    """Verify vq_dropout is correctly threaded to the VQ module and is active."""

    def _make_lam(self, vq_dropout: float) -> LatentActionModel:
        return LatentActionModel(
            vocab_size=VOCAB_SIZE, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
            dropout=0.0, context_length=CONTEXT_LEN, lm_head_out_size=VOCAB_SIZE,
            latent_action_dim=N_EMBD, num_latents=NUM_LATENTS,
            pad_token_id=PAD_ID, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
            vq_dropout=vq_dropout,
        )

    def test_vq_dropout_rate_stored_in_vq_module(self):
        """vq_dropout value reaches vq.dropout.p, not just the LAM config."""
        for p in (0.0, 0.2, 0.5):
            lam = self._make_lam(vq_dropout=p)
            assert lam.vq.dropout.p == p, f"expected vq.dropout.p={p}, got {lam.vq.dropout.p}"

    def test_vq_dropout_via_cgpt_constructor(self):
        """vq_dropout threads correctly through ControllableGPT → LatentActionModel → VQ."""
        cgpt = ControllableGPT(
            vocab_size=VOCAB_SIZE, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD,
            dropout=0.0, context_length=CONTEXT_LEN, lm_head_out_size=VOCAB_SIZE,
            latent_action_dim=N_EMBD, num_latents=NUM_LATENTS,
            pad_token_id=PAD_ID, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
            vq_dropout=0.3,
        )
        assert cgpt.lam.vq.dropout.p == 0.3

    def test_vq_assignments_stochastic_in_train_mode(self):
        """With vq_dropout > 0 and model in train mode, the same input yields different code assignments across runs."""
        lam = self._make_lam(vq_dropout=0.5)
        lam.train()
        x = torch.randint(5, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LEN))
        indices = []
        for _ in range(10):
            with torch.no_grad():
                _, _, vq_dict = lam(x)
            indices.append(vq_dict['indices'].clone())
        all_equal = all(torch.equal(indices[0], r) for r in indices[1:])
        assert not all_equal, "VQ assignments are identical across runs — dropout is not active during training"

    def test_vq_assignments_deterministic_in_eval_mode(self):
        """With model in eval mode, the same input always yields identical code assignments."""
        lam = self._make_lam(vq_dropout=0.5)
        lam.eval()
        x = torch.randint(5, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LEN))
        indices = []
        for _ in range(5):
            with torch.no_grad():
                _, _, vq_dict = lam(x)
            indices.append(vq_dict['indices'].clone())
        assert all(torch.equal(indices[0], r) for r in indices[1:]), \
            "VQ assignments differ in eval mode — dropout is incorrectly active during eval"


# 6. Zinc data values — actual token correctness
# ---------------------------------------------------------------------------

def _load_zinc_dataset(n_samples: int = 2000) -> ControllableGPTDataset:
    """Load the first n_samples rows from the pre-built zinc safetensors file."""
    with safe_open(ZINC_DATASET, framework="pt") as f:
        key = list(f.keys())[0]
        tokens = f.get_tensor(key)[:n_samples]
    return ControllableGPTDataset(tokens)


@zinc_available
class TestZincDataValues:
    """Checks that the packed zinc dataset produces correct token values.

    With packed sampling, all molecules are concatenated into a flat stream and
    chunked into context_length blocks — no PAD tokens, chunks may start
    mid-molecule.  Tests are updated to reflect these invariants.
    """

    N = 2000  # number of rows checked in each test

    def test_no_pad_tokens_anywhere(self):
        """Packed data must contain zero PAD tokens — every position is a real token."""
        ds = _load_zinc_dataset(self.N)
        for i in range(len(ds)):
            x = ds[i]["x"]
            assert ZINC_PAD_ID not in x.tolist(), \
                f"row {i}: PAD token (id={ZINC_PAD_ID}) found in packed chunk"

    def test_y_is_x_shifted_left(self):
        """y must equal x[1:] for every sample — catches any off-by-one in the loader."""
        ds = _load_zinc_dataset(self.N)
        for i in range(len(ds)):
            item = ds[i]
            assert torch.equal(item["y"], item["x"][1:]), \
                f"row {i}: y != x[1:]\n  x={item['x'].tolist()}\n  y={item['y'].tolist()}"

    def test_eos_always_followed_by_bos(self):
        """In the packed stream, every EOS must be immediately followed by BOS.

        The flat token stream is [BOS] mol [EOS] [BOS] mol [EOS] ..., so wherever
        EOS appears inside a chunk, the next token must be BOS.  If it isn't, the
        packing concatenation inserted a gap or duplicated a boundary token.
        """
        ds = _load_zinc_dataset(self.N)
        for i in range(len(ds)):
            row = ds[i]["x"].tolist()
            for pos, tok in enumerate(row[:-1]):
                if tok == ZINC_EOS_ID:
                    assert row[pos + 1] == ZINC_BOS_ID, \
                        f"row {i}: EOS at pos {pos} not followed by BOS: {row[pos-2:pos+4]}"

    def test_no_mask_token_in_training_data(self):
        """[MASK] (id=2) must not appear in raw training data."""
        ds = _load_zinc_dataset(self.N)
        for i in range(len(ds)):
            x = ds[i]["x"]
            assert ZINC_MASK_ID not in x.tolist(), \
                f"row {i}: [MASK] token (id={ZINC_MASK_ID}) found in training data"

    def test_token_ids_in_vocab_range(self):
        """All token ids must be in [0, ZINC_VOCAB_SIZE)."""
        ds = _load_zinc_dataset(self.N)
        for i in range(len(ds)):
            x = ds[i]["x"]
            assert x.min().item() >= 0 and x.max().item() < ZINC_VOCAB_SIZE, \
                f"row {i}: token id out of range [0, {ZINC_VOCAB_SIZE}): {x.tolist()}"

    def test_batch_x_y_column_alignment(self):
        """In a batch, batch['y'][:, t] == batch['x'][:, t+1] for every column t."""
        train_loader, _ = build_dataloaders(
            path=ZINC_DATASET, seed=0, val_ratio=0.1, batch_size=32, drop_last=False,
        )
        batch = next(iter(train_loader))
        x, y = batch["x"], batch["y"]
        T = x.shape[1]
        for t in range(T - 1):
            assert torch.equal(y[:, t], x[:, t + 1]), \
                f"column misalignment at t={t}: y[:,{t}] != x[:,{t+1}]"

    def test_all_tokens_are_real(self):
        """Every position in every chunk must be a real token (no PAD anywhere).

        Packing concatenates molecules into a flat stream and chunks into
        context_length blocks, so there is no structural reason for PAD to
        appear.  This test is the primary efficiency invariant of packed sampling:
        100% of every forward pass contributes to the loss.
        """
        with safe_open(ZINC_DATASET, framework="pt") as f:
            tokens = f.get_tensor(list(f.keys())[0])
        pad_count = (tokens == ZINC_PAD_ID).sum().item()
        assert pad_count == 0, \
            f"{pad_count} PAD tokens found — dataset may not have been packed correctly"
