import logging
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, get_original_cwd
import torch

from interdiff.utils.torch_utils import seed_all
from scripts.tokenise_dataset import run_tokenisation

@hydra.main(version_base=None, config_path="../interdiff/conf", config_name="config")
def main(cfg: DictConfig):

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("train")

    seed_all(cfg.seed)

    save_path = run_tokenisation(cfg)  # returns path to dataset.safetensors
    log.info(f"Using tokenised dataset at {save_path}")

    # Instantiate components from config
    model = instantiate(cfg.model)
    train_cfg = instantiate(cfg.train_cfg)
    optim = instantiate(cfg.optim, model=model)
    sched = instantiate(cfg.sched, optimizer=optim)
    logger = instantiate(cfg.log) if bool(cfg.wandb_log) else None

    cfg.train_cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = instantiate(cfg.trainer, model=model, scheduler=sched, optimizer=optim, logger=logger, train_cfg=train_cfg)

    loaders = instantiate(cfg.loader, dataset_path = save_path)
    train_dataloader = loaders.train_loader
    val_dataloader = loaders.val_loader

    trainer.fit(train_dataloader, val_dataloader)

    if logger:
        logger.finalize()

if __name__ == "__main__":
    main()
