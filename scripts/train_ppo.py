"""PPO finetuning script for molecule generation.

Usage:
    python -m scripts.train_ppo finetune-base --pretrained-ckpt <path> --task qed [options]
    python -m scripts.train_ppo finetune-controllable --pretrained-ckpt <path> --controllable-gpt-path <path> --task qed [options]
"""

import os
import math
import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Union, Annotated

import torch
import tyro
from safetensors import safe_open
from torch.utils.data import DataLoader

from interdiff.data.RLLoader import FinetuneBaseLoader, FinetuneControlable
from interdiff.trainers.base_RL import RLTrainerBase, RLTrainConfig
from interdiff.io import load_tokenizer
from interdiff.logging.wandb_logger import WandbLogger
from interdiff.utils.torch_utils import seed_all
from interdiff.utils.model_stats import log_run_setup, parameter_counts
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
class WandbCfg:
    project: str = "interdiff"
    entity: str = "latent-action-interdiff"
    group: str = ""
    enabled: bool = True


@dataclass
class PPOCfg:
    num_envs: int = 16
    num_steps: int = 256
    budget: int = 250_000_000
    num_epochs: int = 1
    num_minibatches: int = 8
    clip_eps: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    lr: float = 1e-6
    normalise_advantage: bool = True
    clip_value_loss: bool = True
    gae_lambda: float = 0.95
    discount: float = 1.0
    weight_decay: float = 0.01
    anneal_lr: bool = True
    lambda_kld: float = 0.05
    log_frequency: int = 1
    eval_frequency: int = 1
    random_start: bool = False


# ---------------------------------------------------------------------------
# Per-stage top-level configs
# ---------------------------------------------------------------------------


@dataclass
class FinetuneBaseCfg:
    """PPO finetuning of a pretrained base GPT."""

    pretrained_ckpt: str  # path to pretrained GPT checkpoint dir
    task: str = "qed"
    seed: int = 42
    data_smiles: str = "interdiff/data/zinc/zinc.txt"
    ckpt_root: str = "ckpts"
    from_scratch: bool = False
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    wandb: WandbCfg = field(default_factory=lambda: WandbCfg(group="ppo_base"))


@dataclass
class FinetuneControllableCfg:
    """PPO finetuning with latent action controllable model."""

    pretrained_ckpt: str  # path to policy-distilled PolicyNetwork checkpoint dir
    controllable_gpt_path: str  # path to ControllableGPT checkpoint dir
    task: str = "qed"
    seed: int = 42
    data_smiles: str = "interdiff/data/zinc/zinc.txt"
    ckpt_root: str = "ckpts"
    from_scratch: bool = False
    num_latents: int = 128
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    wandb: WandbCfg = field(default_factory=lambda: WandbCfg(group="ppo_controllable"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_name(cfg) -> str:
    ppo = cfg.ppo
    if isinstance(cfg, FinetuneBaseCfg):
        return f"ppo_{cfg.task}_envs{ppo.num_envs}_steps{ppo.num_steps}_seed{cfg.seed}"
    if isinstance(cfg, FinetuneControllableCfg):
        return f"ppo_{cfg.task}_controllable_nlatents{cfg.num_latents}_envs{ppo.num_envs}_steps{ppo.num_steps}_seed{cfg.seed}"
    raise ValueError(f"Unknown config type: {type(cfg)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = tyro.cli(
        Union[
            Annotated[FinetuneBaseCfg, tyro.conf.subcommand("finetune-base")],
            Annotated[
                FinetuneControllableCfg, tyro.conf.subcommand("finetune-controllable")
            ],
        ]
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log = logging.getLogger("train_ppo")

    seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    run_name = _run_name(cfg)
    tok = cfg.tokenizer
    tok_dir, save_path = run_tokenisation(
        data_smiles=cfg.data_smiles,
        vocab_size=tok.vocab_size,
        context_length=tok.context_length,
        use_selfies=tok.use_selfies,
    )
    log.info(f"Using tokenised dataset at {save_path}")

    tokenizer = load_tokenizer(os.path.join(tok_dir, "tokenizer.json"))
    log.info(f"Tokenizer loaded with vocab size {tokenizer.get_vocab_size()}")

    # Load pretrained model via stage-specific loader
    if isinstance(cfg, FinetuneBaseCfg):
        loader = FinetuneBaseLoader()
    elif isinstance(cfg, FinetuneControllableCfg):
        loader = FinetuneControlable(
            ckpt_controllable_path=cfg.controllable_gpt_path,
            ckpt_name="best.pt",
        )

    # Patch cfg into the shape loader.load_pretrained_model expects
    # by building a minimal namespace it can read ckpt.init_from and ckpt.path from.
    from types import SimpleNamespace

    loader_cfg = SimpleNamespace(
        ckpt=SimpleNamespace(
            init_from="resume", path=cfg.pretrained_ckpt, ckpt_name="best.pt"
        ),
        tokenizer=tok,
        ppo=cfg.ppo,
        reward=SimpleNamespace(task=cfg.task),
        context=SimpleNamespace(seq_len=tok.context_length),
        rl_train_cfg=SimpleNamespace(
            tokenizer_dir=tok_dir, ckpt_path=os.path.join(cfg.ckpt_root, run_name)
        ),
        wandb_log=cfg.wandb.enabled,
    )

    model = loader.load_pretrained_model(loader_cfg, device)
    log.info(f"Pretrained model loaded with embedding dim {model.config.n_embd}")

    if cfg.from_scratch:
        log.info(f"Reinitialising model weights from scratch")
        model.apply(model._init_weights)
        for pn, p in model.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * model.config.n_layer)
                )
        loader_cfg.ppo.lambda_kld = 0.0

    reference_model = None
    if cfg.ppo.lambda_kld > 0:
        reference_model = deepcopy(model)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

    env = loader.setup_environment(loader_cfg, model, tokenizer, device)
    log.info(
        f"Environment: {env.num_envs} envs, action space [0, {env.action_space.upper})"
    )

    ppo_agent = loader.setup_ppo_agent(loader_cfg, model, reference_model, device)

    logger = (
        WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            group=cfg.wandb.group,
        )
        if cfg.wandb.enabled
        else None
    )

    log_run_setup(logger, asdict(cfg), model=model, ppo_agent=ppo_agent)

    for name, m in [("model", model), ("ppo_agent", ppo_agent)]:
        counts = parameter_counts(m)
        log.info(
            "%s parameters: total=%s trainable=%s",
            name,
            f"{int(counts['num_parameters']):,}",
            f"{int(counts['num_trainable_parameters']):,}",
        )

    ckpt_path = os.path.join(cfg.ckpt_root, run_name)
    ppo = cfg.ppo
    from interdiff.ppo import HParams

    hparams = HParams(
        num_actions=model.lm_head_out_size,
        num_envs=ppo.num_envs,
        num_steps=ppo.num_steps,
        budget=ppo.budget,
        num_epochs=ppo.num_epochs,
        num_minibatches=ppo.num_minibatches,
        clip_eps=ppo.clip_eps,
        ent_coef=ppo.ent_coef,
        max_episode_length=tok.context_length,
        vf_coef=ppo.vf_coef,
        max_grad_norm=ppo.max_grad_norm,
        lr=ppo.lr,
        normalise_advantage=ppo.normalise_advantage,
        clip_value_loss=ppo.clip_value_loss,
        gae_lambda=ppo.gae_lambda,
        discount=ppo.discount,
        weight_decay=ppo.weight_decay,
        anneal_lr=ppo.anneal_lr,
        lambda_kld=ppo.lambda_kld,
        log_frequency=ppo.log_frequency,
        log_to_wandb=cfg.wandb.enabled,
        wandb_project_name=cfg.wandb.project,
        save_dir=ckpt_path,
        eval_frequency=ppo.eval_frequency,
        random_start=ppo.random_start,
    )
    ppo_agent.hparams = hparams

    rl_train_cfg = RLTrainConfig(
        device=str(device),
        budget=ppo.budget,
        log_frequency=ppo.log_frequency,
        eval_frequency=ppo.eval_frequency,
        ckpt_path=ckpt_path,
        n_mols_generate=100,
        tokenizer_dir=tok_dir,
        pad_token_id=tok.pad_token_id,
    )

    with safe_open(save_path, framework="pt") as f:
        tokens_tensor = f.get_tensor(list(f.keys())[0])
    if tokens_tensor.is_cuda:
        tokens_tensor = tokens_tensor.cpu()
    from interdiff.data.GPTLoader import NextTokenDataset

    train_dataloader = DataLoader(
        NextTokenDataset(tokens_tensor, pad_token_id=tok.pad_token_id),
        batch_size=2048,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    trainer = RLTrainerBase(
        ppo_agent=ppo_agent, env=env, logger=logger, train_cfg=rl_train_cfg
    )
    trainer.fit(train_dataloader=train_dataloader)

    sentinel = os.path.join(ckpt_path, "done")
    open(sentinel, "w").close()
    log.info("Training complete. Sentinel written to %s", sentinel)


if __name__ == "__main__":
    main()
