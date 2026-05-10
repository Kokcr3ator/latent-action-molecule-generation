"""Supervised pretraining / policy-distillation training script.

Usage:
    python -m scripts.train pretrain-base [options]
    python -m scripts.train pretrain-controllable [options]
    python -m scripts.train policy-distill --controllable-gpt-path <path> [options]
"""
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Union, Annotated

import torch
import torch.optim.lr_scheduler
import tyro

from interdiff.models import GPT, ControllableGPT, PolicyNetwork
from interdiff.trainers import GPTTrainer, ControllableGPTTrainer, PretrainPolicyTrainer
from interdiff.trainers.base import TrainConfig
from interdiff.data.GPTLoader import GPTLoader
from interdiff.data.ControllableGPTLoader import ControllableGPTLoader
from interdiff.data.PretrainPolicyLoader import PretrainPolicyLoader
from interdiff.optim import adam_w
from interdiff.logging.wandb_logger import WandbLogger
from interdiff.logging.logger import log_run_setup
from interdiff.metrics import parameter_counts
from interdiff.io import seed_all
from scripts.tokenise_dataset import run_tokenisation


# ---------------------------------------------------------------------------
# Shared config blocks
# ---------------------------------------------------------------------------

@dataclass
class TokenizerCfg:
    vocab_size: int = 2_048
    context_length: int = 128
    use_selfies: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 3
    eos_token_id: int = 4


@dataclass
class TrainingCfg:
    max_iters: int = 100_000
    batch_size: int = 2_048
    gradient_accumulation_steps: int = 1
    grad_clip: float = 1.0
    warmup_iters: int = 1_000
    eval_interval: int = 200
    eval_iters: int = 5
    log_interval: int = 20
    n_mols_generate: int = 200
    mixed_dtype: str = "float16"
    compile_model: bool = True


@dataclass
class OptimCfg:
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.0
    eta_min: float = 1e-6


@dataclass
class WandbCfg:
    project: str = "interdiff"
    entity: str = "latent-action-interdiff"
    group: str = ""
    dir: str = ""


# ---------------------------------------------------------------------------
# Per-stage model configs
# ---------------------------------------------------------------------------

@dataclass
class GPTModelCfg:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False


@dataclass
class ControllableGPTModelCfg:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    latent_action_dim: int = 384
    num_latents: int = 128
    entropy_weight: float = 0.01
    vq_beta: float = 0.25
    norm_mode: str = "none"
    norm_penalty_weight: float = 1.0


@dataclass
class PolicyNetworkModelCfg:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    num_latents: int = 128


# ---------------------------------------------------------------------------
# Per-stage top-level configs
# ---------------------------------------------------------------------------

@dataclass
class PretrainBaseCfg:
    """Pretrain a base autoregressive GPT on SMILES."""
    seed: int = 42
    data_smiles: str = "interdiff/data/zinc/zinc.txt"
    ckpt_root: str = "/scratch/uceeepi/interdiff/ckpts"
    resume: bool = False
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    model: GPTModelCfg = field(default_factory=GPTModelCfg)
    training: TrainingCfg = field(default_factory=lambda: TrainingCfg(max_iters=20_000))
    optim: OptimCfg = field(default_factory=OptimCfg)
    wandb: WandbCfg = field(default_factory=lambda: WandbCfg(group="pretrain_base"))


@dataclass
class PretrainControllableCfg:
    """Pretrain a ControllableGPT with VQ latent action model."""
    seed: int = 42
    data_smiles: str = "interdiff/data/zinc/zinc.txt"
    ckpt_root: str = "/scratch/uceeepi/interdiff/ckpts"
    resume: bool = False
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    model: ControllableGPTModelCfg = field(default_factory=ControllableGPTModelCfg)
    training: TrainingCfg = field(default_factory=lambda: TrainingCfg(max_iters=20_000))
    optim: OptimCfg = field(default_factory=OptimCfg)
    wandb: WandbCfg = field(default_factory=lambda: WandbCfg(group="pretrain_controllable"))


@dataclass
class PolicyDistillCfg:
    """Distil a PolicyNetwork from a trained ControllableGPT."""
    controllable_gpt_path: str  # required — no default
    seed: int = 42
    data_smiles: str = "interdiff/data/zinc/zinc.txt"
    ckpt_root: str = "/scratch/uceeepi/interdiff/ckpts"
    resume: bool = False
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    model: PolicyNetworkModelCfg = field(default_factory=PolicyNetworkModelCfg)
    training: TrainingCfg = field(default_factory=lambda: TrainingCfg(max_iters=20_000))
    optim: OptimCfg = field(default_factory=OptimCfg)
    wandb: WandbCfg = field(default_factory=lambda: WandbCfg(group="policydistillation"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_name(cfg) -> str:
    tok = cfg.tokenizer
    if isinstance(cfg, PretrainBaseCfg):
        return f"pretrain_base_vocab{tok.vocab_size}_seed{cfg.seed}"
    if isinstance(cfg, PretrainControllableCfg):
        norm = cfg.model.norm_mode
        norm_suffix = f"_norm{norm}" if norm != "none" else ""
        return f"pretrain_controllable_vocab{tok.vocab_size}_nlatent{cfg.model.num_latents}{norm_suffix}_seed{cfg.seed}"
    if isinstance(cfg, PolicyDistillCfg):
        return f"policydistillation_nlatents{cfg.model.num_latents}_vocab{tok.vocab_size}_seed{cfg.seed}"
    raise ValueError(f"Unknown config type: {type(cfg)}")


def _build_train_config(cfg, ckpt_path: str, tok_dir: str) -> TrainConfig:
    t, tok = cfg.training, cfg.tokenizer
    return TrainConfig(
        max_iters=t.max_iters,
        batch_size=t.batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        grad_clip=t.grad_clip,
        warmup_iters=t.warmup_iters,
        eval_interval=t.eval_interval,
        eval_iters=t.eval_iters,
        log_interval=t.log_interval,
        n_mols_generate=t.n_mols_generate,
        mixed_dtype=t.mixed_dtype,
        compile_model=t.compile_model,
        ckpt_path=ckpt_path,
        tokenizer_dir=tok_dir,
        pad_token_id=tok.pad_token_id,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = tyro.cli(Union[
        Annotated[PretrainBaseCfg, tyro.conf.subcommand("pretrain-base")],
        Annotated[PretrainControllableCfg, tyro.conf.subcommand("pretrain-controllable")],
        Annotated[PolicyDistillCfg, tyro.conf.subcommand("policy-distill")],
    ])

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("train")

    seed_all(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_name = _run_name(cfg)
    tok = cfg.tokenizer
    tok_dir, save_path = run_tokenisation(
        data_smiles=cfg.data_smiles,
        vocab_size=tok.vocab_size,
        context_length=tok.context_length,
        use_selfies=tok.use_selfies,
    )
    log.info(f"Using tokenised dataset at {save_path}")

    ckpt_path = os.path.join(cfg.ckpt_root, run_name)
    train_cfg = _build_train_config(cfg, ckpt_path, tok_dir)
    train_cfg.device = device

    # Build model
    m = cfg.model
    if isinstance(cfg, PretrainBaseCfg):
        model = GPT(
            vocab_size=tok.vocab_size, n_layer=m.n_layer, n_head=m.n_head, n_embd=m.n_embd,
            dropout=m.dropout, bias=m.bias, context_length=tok.context_length,
            lm_head_out_size=tok.vocab_size,
            pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        )
    elif isinstance(cfg, PretrainControllableCfg):
        model = ControllableGPT(
            vocab_size=tok.vocab_size, n_layer=m.n_layer, n_head=m.n_head, n_embd=m.n_embd,
            dropout=m.dropout, bias=m.bias, context_length=tok.context_length,
            lm_head_out_size=tok.vocab_size,
            pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
            latent_action_dim=m.latent_action_dim, num_latents=m.num_latents,
            entropy_weight=m.entropy_weight, vq_beta=m.vq_beta,
            norm_mode=m.norm_mode, norm_penalty_weight=m.norm_penalty_weight,
        )
    elif isinstance(cfg, PolicyDistillCfg):
        model = PolicyNetwork(
            vocab_size=tok.vocab_size, n_layer=m.n_layer, n_head=m.n_head, n_embd=m.n_embd,
            dropout=m.dropout, bias=m.bias, context_length=tok.context_length,
            num_latents=m.num_latents,
            pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        )

    o = cfg.optim
    optimizer = adam_w(model=model, learning_rate=o.lr, beta1=o.beta1, beta2=o.beta2,
                       weight_decay=o.weight_decay, device_type=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.max_iters, eta_min=o.eta_min,
    )

    logger = WandbLogger(
        project=cfg.wandb.project, entity=cfg.wandb.entity, name=run_name, group=cfg.wandb.group,
        dir=cfg.wandb.dir,
    )

    log_run_setup(logger, asdict(cfg), model=model)

    counts = parameter_counts(model)
    log.info("Model parameters: total=%s trainable=%s",
             f"{int(counts['num_parameters']):,}", f"{int(counts['num_trainable_parameters']):,}")

    # Build trainer
    if isinstance(cfg, PretrainBaseCfg):
        trainer = GPTTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                             logger=logger, train_cfg=train_cfg)
    elif isinstance(cfg, PretrainControllableCfg):
        trainer = ControllableGPTTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                         logger=logger, train_cfg=train_cfg)
    elif isinstance(cfg, PolicyDistillCfg):
        trainer = PretrainPolicyTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                                        logger=logger, train_cfg=train_cfg,
                                        controllable_gpt_path=cfg.controllable_gpt_path)

    if cfg.resume:
        resume_path = os.path.join(ckpt_path, "best.pt")
        trainer.load_checkpoint(resume_path)
        log.info("Resumed from %s at step %d", resume_path, trainer.state.step)

    # Build loaders
    if isinstance(cfg, PretrainBaseCfg):
        loaders = GPTLoader(dataset_path=save_path, pad_token_id=tok.pad_token_id,
                            batch_size=train_cfg.batch_size, seed=cfg.seed)
    elif isinstance(cfg, PretrainControllableCfg):
        loaders = ControllableGPTLoader(dataset_path=save_path,
                                        batch_size=train_cfg.batch_size, seed=cfg.seed)
    elif isinstance(cfg, PolicyDistillCfg):
        loaders = PretrainPolicyLoader(
            controllable_gpt_path=cfg.controllable_gpt_path, dataset_path=save_path,
            action_dataset_out_dir=f"data/processed/zinc",
            pad_token_id=tok.pad_token_id, batch_size=train_cfg.batch_size, seed=cfg.seed,
        )

    trainer.fit(loaders.train_loader, loaders.val_loader)

    os.makedirs(ckpt_path, exist_ok=True)
    sentinel = os.path.join(ckpt_path, "done")
    open(sentinel, "w").close()
    log.info("Training complete. Sentinel written to %s", sentinel)

    best_ckpt = os.path.join(ckpt_path, "best.pt")
    if logger and os.path.exists(best_ckpt):
        logger.log_artifact(best_ckpt, name=os.path.basename(ckpt_path))

    if logger:
        logger.finalize()


if __name__ == "__main__":
    main()
