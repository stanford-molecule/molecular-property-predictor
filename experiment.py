import logging
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict
from typing import List
from uuid import uuid4

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch.autograd import Variable
from tqdm import tqdm
import wandb

from dataset import DataLoaderGMN, DataLoaderGNN
from gnn import GNN

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class GNNExperiment:
    """
    Run a single experiment (i.e. train and then evaluate on train/test/validation splits)
    given the provided parameters, and store in a pickle file for analysis.

    Most of the code is either copied from https://github.com/snap-stanford/ogb or heavily inspired by it.
    TODO: create a factory method to switch between GMN and normal
    """

    # we'll only be working on `molhiv` for this project
    DATASET_NAME = "ogbg-molhiv"
    PROJECT_NAME = 'gmn'
    NUM_TASKS = 1
    DEBUG_EPOCHS = 2  # makes for quick debugging
    DEBUG_BATCHES = 20

    def __init__(
        self,
        gnn_type: str,
        dropout: float,
        num_layers: int,
        emb_dim: int,
        epochs: int,
        lr: float,
        device: int,
        batch_size: int,
        num_workers: int,
        debug: bool = False,
    ):
        self.param_gnn_type = gnn_type
        self.param_dropout = dropout
        self.param_num_layers = num_layers
        self.param_emb_dim = emb_dim
        self.param_epochs = epochs if not debug else self.DEBUG_EPOCHS
        self.param_batch_size = batch_size
        self.param_lr = lr
        self.num_workers = num_workers
        self.debug = debug

        self.dataset = PygGraphPropPredDataset(self.DATASET_NAME)
        self.eval_metric = self.dataset.eval_metric
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        self.split_idx = self.dataset.get_idx_split()
        self.evaluator = Evaluator(self.DATASET_NAME)
        self.model: GNN = self._get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param_lr)
        loader_partial = partial(
            self._get_loader,
            dataset=self.dataset,
            split_idx=self.split_idx,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.loaders = {
            "train": loader_partial(data_split="train", shuffle=True),
            "valid": loader_partial(data_split="valid", shuffle=False),
            "test": loader_partial(data_split="test", shuffle=False),
        }

        # placeholders
        self.times = {}
        self.loss_curve = []
        self.train_curve = []
        self.valid_curve = []
        self.test_curve = []
        self.epoch = None

    def _init_wandb(self):
        wandb.init(project=self.PROJECT_NAME, config=self.params)

    def run(self):
        """
        Run the experiment + store results.
        """
        self._init_wandb()

        self.times["start"] = datetime.now()
        for self.epoch in range(self.param_epochs):
            logger.info("=====Epoch {}".format(self.epoch + 1))
            logger.info("Training...")
            loss = float(self._train())
            logger.info(f"Loss: {loss:.4f}")
            self.loss_curve.append(loss)

            logger.info("Evaluating...")
            train_perf = float(self._eval(data_split="train"))
            valid_perf = float(self._eval(data_split="valid"))
            test_perf = float(self._eval(data_split="test"))
            logger.info(
                {"Train": train_perf, "Validation": valid_perf, "Test": test_perf}
            )
            wandb.log({'train_loss': loss, 'train_acc': train_perf, 'val_acc': valid_perf, 'test_acc': test_perf})

            self.train_curve.append(train_perf)
            self.valid_curve.append(valid_perf)
            self.test_curve.append(test_perf)

            if self.stop_early:
                break

        self.times["end"] = datetime.now()
        best_val_epoch = int(np.array(self.valid_curve).argmax())
        best_train = max(self.train_curve)

        logger.info("All done!")
        logger.info(f"Best train : {best_train}")
        logger.info(f"Best valid : {self.valid_curve[best_val_epoch]}")
        logger.info(f"Best test  : {self.test_curve[best_val_epoch]}")
        self._store_results()

    @property
    def stop_early(self) -> bool:
        return False

    @property
    def max_nodes(self):
        return max(len(x.x) for x in self.dataset)

    def _store_results(self) -> None:
        """
        Stores all the curves and experiment parameters in a pickle file.
        """
        results = {
            "loss": self.loss_curve,
            "train": self.train_curve,
            "valid": self.valid_curve,
            "test": self.test_curve,
        }
        to_store = {**results, **self.params, **self.times}
        path = Path(f"results/{uuid4()}.pkl")
        path.parent.mkdir(exist_ok=True)
        pickle.dump(to_store, open(path, "wb"))

    @property
    def params(self) -> Dict:
        return {k.replace('param_', ''): v for k, v in self.__dict__.items() if k.startswith("param_")}

    def _get_model(self) -> GNN:
        gnn_partial = partial(
            GNN,
            num_tasks=self.NUM_TASKS,
            num_layer=self.param_num_layers,
            emb_dim=self.param_emb_dim,
            drop_ratio=self.param_dropout,
            virtual_node="virtual" in self.param_gnn_type,
        )
        return gnn_partial(gnn_type=self.param_gnn_type).to(self.device)

    def _get_loader(
        self,
        dataset: PygGraphPropPredDataset,
        split_idx: Dict[str, torch.Tensor],
        data_split: str,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoaderGNN:
        dataset_split = dataset[split_idx[data_split]]
        data_loader_cls = getattr(self, "data_loader_cls", DataLoaderGNN)
        return data_loader_cls(
            dataset_split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            max_nodes=self.max_nodes,
        )

    def _train(self) -> float:
        self.model.train()
        loss = 0

        for batch in tqdm(self.loaders["train"], desc="Training"):
            batch = batch.to(self.device)

            if len(batch.x) > 1 and batch.batch[-1] > 0:
                pred = self.model(batch)
                self.optimizer.zero_grad()
                # ignore nan targets (unlabeled) when computing training loss.
                is_labeled = batch.y == batch.y
                loss_tensor = self.loss_fn(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled],
                )
                loss += loss_tensor.item()
                loss_tensor.backward()
                self.optimizer.step()
            else:
                raise ValueError("how is this possible???")

            if self.debug:
                break

        return loss

    def __eval(self, y_true, y_pred):
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return self.evaluator.eval(input_dict)[self.eval_metric]

    def _eval_batch(self, batch, y_true, y_pred):
        batch = batch.to(self.device)

        if len(batch.x) > 1:
            with torch.no_grad():
                pred = self.model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
        return y_true, y_pred

    def _eval(self, data_split: str) -> float:
        y_true = []
        y_pred = []

        tqdm_desc = f"Evaluating the {data_split} split"
        with torch.no_grad():
            self.model.eval()

            for idx, batch in enumerate(tqdm(self.loaders[data_split], desc=tqdm_desc)):
                # TODO: fix the 0-d tensor case for GMN
                # if len(batch["label"].size()) > 0:
                y_true, y_pred = self._eval_batch(batch, y_true, y_pred)
                if self.debug and idx + 1 >= self.DEBUG_BATCHES:
                    break

        return self.__eval(y_true, y_pred)


class GMNExperiment(GNNExperiment):
    def __init__(
        self,
        dropout: float,
        num_layers: int,
        emb_dim: int,
        epochs: int,
        lr: float,
        device: int,
        batch_size: int,
        num_workers: int,
        alpha: float,
        e_out: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pos_dim: int,
        num_centroids: List[int],
        weight_decay: float,
        decay_step: int,
        cluster_heads: int,
        learn_centroid: str,
        backward_period: int,
        clip: float,
        avg_grad: bool,
        num_clusteriter: int,
        use_rwr: bool,
        mask_nodes: bool,
        batchnorm: bool,
        c_heads_pool: str,
        p2p: bool,
        linear_block: bool,
        debug: bool = False,
    ):
        self.data_loader_cls = DataLoaderGMN
        super().__init__(
            "gmn",
            dropout,
            num_layers,
            emb_dim,
            epochs,
            lr,
            device,
            batch_size,
            num_workers,
            debug,
        )
        from model import GMN

        self.num_centroids = num_centroids
        self.weight_decay = weight_decay
        self.decay_step = decay_step
        self.learn_centroid = learn_centroid
        self.backward_period = backward_period
        self.clip = clip
        self.avg_grad = avg_grad
        self.num_clusteriter = num_clusteriter
        self.use_rwr = use_rwr
        self.mask_nodes = mask_nodes

        self.total_num_cluster = len(self.num_centroids)
        self.model = GMN(
            alpha,
            e_out,
            input_dim,
            hidden_dim,
            output_dim,
            pos_dim,
            num_centroids,
            self.max_nodes,
            cluster_heads,
            dropout,
            batchnorm,
            c_heads_pool,
            p2p,
            linear_block,
            backward_period,
        )
        self.opt1, self.opt2, self.opt3 = self._get_optimizers()

    def _get_optimizers(self):
        param_dict = [
            {"params": self.model.centroids, "lr": self.param_lr},
            {"params": list(self.model.parameters())[1:], "lr": self.param_lr},
        ]
        param_dict_3 = [
            {"params": list(self.model.parameters())[1:], "lr": self.param_lr}
        ]
        opt1 = torch.optim.Adam(
            param_dict, lr=self.param_lr, weight_decay=self.weight_decay
        )
        opt2 = torch.optim.Adam(
            [self.model.centroids], lr=self.param_lr, weight_decay=self.weight_decay
        )
        opt3 = torch.optim.Adam(
            param_dict_3, lr=self.param_lr, weight_decay=self.weight_decay
        )
        return opt1, opt2, opt3

    def _adjust_lr(self):
        self.param_lr *= self.weight_decay
        for param_group in self.opt1.param_groups:
            param_group["lr"] = self.param_lr
        for param_group in self.opt2.param_groups:
            param_group["lr"] = self.param_lr
        for param_group in self.opt3.param_groups:
            param_group["lr"] = self.param_lr

    def _compute(self, batch, is_train: bool):
        batch_num_nodes = batch["num_nodes"].int().numpy() if self.mask_nodes else None
        h0 = Variable(batch["feats"].float(), requires_grad=False).to(self.device)
        label = Variable(batch["label"].long()).to(self.device)
        adj_key = "rwr" if self.use_rwr else "adj"
        adj = Variable(batch[adj_key].float(), requires_grad=False).to(self.device)

        for c_layer in range(self.total_num_cluster):
            if c_layer == 0:
                # clone, detach, and not trainable for the first layer
                new_adj = adj.clone().detach().requires_grad_(False)
                new_feat = h0.clone().detach().requires_grad_(False)
                del adj, h0
            else:
                new_adj.requires_grad_(is_train)
                new_feat.requires_grad_(is_train)

            # if it's not the last cluster
            master_node_flag = False if c_layer + 1 < self.total_num_cluster else True
            if c_layer + 1 < self.total_num_cluster:
                for _ in range(self.num_clusteriter):
                    _, hard_loss, new_adj, new_feat, _ = self.model(
                        new_feat,
                        new_adj,
                        self.epoch,
                        batch_num_nodes,
                        c_layer,
                        master_node_flag,
                    )

            # for the last cluster
            else:
                *_, h_prime = self.model(
                    new_feat,
                    new_adj,
                    self.epoch,
                    batch_num_nodes,
                    c_layer,
                    master_node_flag,
                )

        return h_prime, label, hard_loss

    def _train(self) -> float:
        self.model.train()
        loss = 0
        batch_cnt = 0
        todo = (  # TODO: what is this condition?
            self.epoch > 0
            and self.epoch % self.backward_period == 1
            and len(self.num_centroids) > 1
            and self.learn_centroid is not "f"
        )

        if self.epoch > 0 and self.epoch % self.decay_step == 0:
            self._adjust_lr()

        # TODO: enhance data loader to support this
        for idx, batch in enumerate(tqdm(self.loaders["train"], desc="Training")):
            batch_cnt += 1
            h_prime, label, hard_loss = self._compute(batch, is_train=True)
            preds = torch.squeeze(h_prime)
            loss_tensor = self.model.loss(preds, label)
            loss += loss_tensor.item()
            self.model.centroids.requires_grad_(False)

            if todo:
                self.model.centroids.requires_grad_(True)
                if self.learn_centroid in ("a", "c"):
                    hard_loss.backward()
            else:
                self.opt3.zero_grad()
                loss_tensor.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.opt3.step()

            if self.debug and idx >= self.DEBUG_BATCHES:
                break

        if todo:
            self.model.centroids.requires_grad_(True)

            if self.avg_grad:
                for i, m in enumerate(self.model.parameters()):
                    if m.grad is not None:
                        list(self.model.parameters())[i].grad = m.grad / (batch_cnt - 1)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            if self.learn_centroid == "c":
                self.opt2.step()
                self.opt2.zero_grad()
            elif self.learn_centroid == "a":
                self.opt1.step()
                self.opt1.zero_grad()

        return loss

    def _eval_batch(self, batch, y_true, y_pred):
        h_prime, label, _ = self._compute(batch, is_train=False)
        # preds = torch.squeeze(h_prime)
        preds = h_prime
        scores, _ = torch.max(preds, dim=1)
        y_true.append(label.unsqueeze(dim=-1).detach().cpu())
        y_pred.append(scores.unsqueeze(dim=-1).detach().cpu())
        return y_true, y_pred


class GMNExperimentRethink(GNNExperiment):
    def __init__(
        self,
        dropout: float,
        num_layers: int,
        emb_dim: int,
        epochs: int,
        lr: float,
        device: int,
        batch_size: int,
        num_workers: int,
        num_heads: int,
        hidden_dim: int,
        num_keys: List[int],
        mem_hidden_dim: int,
        variant: str,
        lr_decay_patience: int,
        kl_period: int,
        early_stop_patience: int,
        debug: bool = False,
    ):
        super().__init__(
            "gmn-rethink",
            dropout,
            num_layers,
            emb_dim,
            epochs,
            lr,
            device,
            batch_size,
            num_workers,
            debug,
        )
        from data.datasets import get_data
        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        from gmn.GMN import GMN

        self.param_kl_period = kl_period
        self.param_variant = variant
        self.param_early_stop_patience = early_stop_patience
        self.param_heads = num_heads
        self.param_hidden_dim = hidden_dim
        self.param_num_keys = num_keys
        self.param_mem_hidden_dim = mem_hidden_dim
        self.param_variant = variant

        self.epochs_no_improve = 0
        self.epoch_stop = self.DEBUG_BATCHES if self.debug else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        res = get_data(self.DATASET_NAME, batch_size)
        train_loader, val_loader, test_loader, stats, evaluator, encode_edge = res
        self.loaders = {"train": train_loader, "test": test_loader, "valid": val_loader}
        self.model = GMN(
            stats["num_features"],
            stats["max_num_nodes"],
            stats["num_classes"],
            num_heads,
            hidden_dim,
            num_keys,
            mem_hidden_dim,
            variant=variant,
            encode_edge=encode_edge,
        ).to(self.device)
        no_keys_param_list = [
            param for name, param in self.model.named_parameters() if "keys" not in name
        ]
        self.optimizer = Adam(no_keys_param_list, lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            min_lr=1e-6,
            patience=lr_decay_patience,
        )
        self.kl_optimizer = Adam(self.model.parameters(), lr=lr)
        self.kl_scheduler = ReduceLROnPlateau(
            self.kl_optimizer,
            mode="max",
            factor=0.5,
            min_lr=1e-6,
            patience=lr_decay_patience,
        )

    @property
    def stop_early(self) -> bool:
        if self.epochs_no_improve >= self.param_early_stop_patience:
            logging.info("Hit early stopping!")
            return True
        return False

    def _train(self):
        from gmn.train import train, kl_train

        trainer = (
            kl_train
            if self.epoch % self.param_kl_period == 0 and self.param_variant == "gmn"
            else train
        )
        train_sup_loss, train_kl_loss = trainer(
            self.model,
            self.kl_optimizer,
            self.loaders["train"],
            self.device,
            self.epoch_stop,
        )
        return train_sup_loss

    def _eval(self, data_split: str):
        from gmn.train import evaluate

        loader = self.loaders[data_split]
        acc, _, _ = evaluate(
            self.model,
            loader,
            self.device,
            evaluator=self.evaluator,
            data_split=data_split,
            epoch_stop=self.epoch_stop,
        )
        if data_split == "valid":
            self.scheduler.step(acc)

            if (
                self.epoch > 2
                and self.valid_curve[-1]
                <= self.valid_curve[-2 - self.epochs_no_improve]
            ):
                self.epochs_no_improve += 1
            else:
                self.epochs_no_improve = 0

        return acc


if __name__ == "__main__":
    exp = GMNExperimentRethink(
        dropout=0.5,
        num_layers=5,
        emb_dim=300,
        epochs=1000,
        lr=1e-3,
        device=0,
        batch_size=32,
        num_workers=0,  # everything in the main process
        num_heads=5,
        hidden_dim=64,
        num_keys=[32, 1],
        mem_hidden_dim=16,
        variant="distance",
        lr_decay_patience=10,
        kl_period=5,
        early_stop_patience=50,
        debug=True,
    )

    exp.run()
