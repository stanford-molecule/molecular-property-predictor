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
from torch_geometric.data import DataLoader
from tqdm import tqdm

from gnn import GNN
from model import GMN
from dataset import DataLoaderGMN

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Experiment:
    """
    Run a single experiment (i.e. train and then evaluate on train/test/validation splits)
    given the provided parameters, and store in a pickle file for analysis.

    Most of the code is either copied from https://github.com/snap-stanford/ogb or heavily inspired by it.
    TODO: create a factory method to switch between GMN and normal
    """

    # we'll only be working on `molhiv` for this project
    DATASET_NAME = "ogbg-molhiv"
    NUM_TASKS = 1
    DEBUG_EPOCHS = 2  # makes for quick debugging
    DEBUG_VAL_BATCHES = 10

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

    def run(self):
        """
        Run the experiment + store results.
        """

        self.times["start"] = datetime.now()
        for self.epoch in range(self.param_epochs):
            logger.info("=====Epoch {}".format(self.epoch + 1))
            logger.info("Training...")
            loss = self._train()
            logger.info(f"Loss: {loss:.4f}")
            self.loss_curve.append(loss)

            logger.info("Evaluating...")
            train_perf = self._eval(data_split="train")
            valid_perf = self._eval(data_split="valid")
            test_perf = self._eval(data_split="test")
            logger.info(
                {"Train": train_perf, "Validation": valid_perf, "Test": test_perf}
            )

            self.train_curve.append(train_perf)
            self.valid_curve.append(valid_perf)
            self.test_curve.append(test_perf)

        self.times["end"] = datetime.now()
        best_val_epoch = int(np.array(self.valid_curve).argmax())
        best_train = max(self.train_curve)

        logger.info("All done!")
        logger.info(f"Best train : {best_train}")
        logger.info(f"Best valid : {self.valid_curve[best_val_epoch]}")
        logger.info(f"Best test  : {self.test_curve[best_val_epoch]}")
        self._store_results()

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
        return {k: v for k, v in self.__dict__.items() if k.startswith("param_")}

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
    ) -> DataLoader:
        dataset_split = dataset[split_idx[data_split]]
        data_loader_cls = getattr(self, 'data_loader_cls', DataLoader)
        return data_loader_cls(
            dataset_split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
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
                y_true, y_pred = self._eval_batch(batch, y_true, y_pred)
                if self.debug and idx + 1 >= self.DEBUG_VAL_BATCHES:
                    break

        return self.__eval(y_true, y_pred)


class GMNExperiment(Experiment):
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
            debug: bool = False
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
            debug
        )
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
            c_heads_pool
        )
        self.opt1, self.opt2, self.opt3 = self._get_optimizers()

    @property
    def max_nodes(self):
        return max(len(x.x) for x in self.dataset)

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
        for batch in tqdm(self.loaders["train"], desc="Training"):
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
        preds = torch.squeeze(h_prime)
        scores, _ = torch.max(preds, dim=1)
        y_true.append(label.detach().cpu())
        y_pred.append(scores.detach().cpu())
        return y_true, y_pred


if __name__ == "__main__":
    gmn = True
    if not gmn:
        exp = Experiment(
            gnn_type="gin",
            dropout=0.5,
            num_layers=5,
            emb_dim=300,
            epochs=100,
            lr=1e-3,
            device=0,
            batch_size=32,
            num_workers=0,  # everything in the main process
            debug=True,
        )
        exp.run()
    else:
        exp_gmn = GMNExperiment(
            dropout=.5,
            num_layers=5,
            emb_dim=10,
            epochs=2,
            lr=2e-3,
            device=0,
            batch_size=16,
            num_workers=0,
            alpha=.2,
            e_out=1,
            input_dim=9,  # TODO set this using the dataset
            hidden_dim=64,
            output_dim=9,  # TODO should we play with this?
            pos_dim=16,
            num_centroids=[10, 1],
            weight_decay=.5,
            decay_step=400,
            cluster_heads=5,
            learn_centroid='a',
            backward_period=5,
            clip=2,
            avg_grad=True,
            num_clusteriter=1,
            use_rwr=True,
            mask_nodes=True,
            batchnorm=True,
            c_heads_pool='conv',
            debug=False,
        )
        exp_gmn.run()
