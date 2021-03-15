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
import wandb
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch.autograd import Variable
from tqdm import tqdm

import attacks
import deeper
from dataset import DataLoaderGMN, DataLoaderGNN
from gnn import GNN, GNNFlag

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
    PROJECT_NAME = "gmn"
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
        grad_clip: float = 0.0,
        debug: bool = False,
        desc: str = "",
    ):
        self.param_gnn_type = gnn_type
        self.param_dropout = dropout
        self.param_num_layers = num_layers
        self.param_emb_dim = emb_dim
        self.param_epochs = epochs if not debug else self.DEBUG_EPOCHS
        self.param_batch_size = batch_size
        self.param_lr = lr
        self.param_grad_clip = grad_clip
        self.num_workers = num_workers
        self.debug = debug
        self.desc = desc

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

    @property
    def model_param_cnt(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def run(self):
        """
        Initialize W&B
        """
        tags = ["debug"] if self.debug else None
        name = ("debug-" if self.debug else "") + "-".join(self.desc.lower().split())
        with wandb.init(
            project=self.PROJECT_NAME, config=self.params, tags=tags, name=name
        ) as wb:
            wb.watch(self.model)

            """
            Run the experiment + store results.
            """
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
                wb.log(
                    {
                        "train_loss": loss,
                        "train_acc": train_perf,
                        "val_acc": valid_perf,
                        "test_acc": test_perf,
                    }
                )

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
    def model_cls(self):
        return GNN

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
        params = {
            k.replace("param_", ""): v
            for k, v in self.__dict__.items()
            if k.startswith("param_")
        }
        return {**params, **{"model_param_cnt": self.model_param_cnt}}

    def _get_model(self) -> GNN:
        gnn_partial = partial(
            self.model_cls,
            num_tasks=self.NUM_TASKS,
            num_layer=self.param_num_layers,
            emb_dim=self.param_emb_dim,
            drop_ratio=self.param_dropout,
            virtual_node="virtual" in self.param_gnn_type,
        )
        gnn_type = self.param_gnn_type.split("-")[0]
        return gnn_partial(gnn_type=gnn_type).to(self.device)

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
                self.optimizer.zero_grad()
                pred = self.model(batch)
                # ignore nan targets (unlabeled) when computing training loss.
                is_labeled = batch.y == batch.y
                loss_tensor = self.loss_fn(
                    pred.to(torch.float32)[is_labeled],
                    batch.y.to(torch.float32)[is_labeled],
                )
                loss += loss_tensor.item()
                loss_tensor.backward()
                if self.param_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.param_grad_clip
                    )
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

    @torch.no_grad()
    def _eval(self, data_split: str) -> float:
        y_true = []
        y_pred = []

        tqdm_desc = f"Evaluating the {data_split} split"
        with torch.no_grad():
            self.model.eval()

            for idx, batch in enumerate(tqdm(self.loaders[data_split], desc=tqdm_desc)):
                y_true, y_pred = self._eval_batch(batch, y_true, y_pred)
                if self.debug and idx + 1 >= self.DEBUG_BATCHES:
                    break

        return self.__eval(y_true, y_pred)


class GNNFLAGExperiment(GNNExperiment):
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
        m: int,
        step_size: float,
        grad_clip: float = 0,
        debug: bool = False,
        desc: str = "",
    ):
        super().__init__(
            gnn_type,
            dropout,
            num_layers,
            emb_dim,
            epochs,
            lr,
            device,
            batch_size,
            num_workers,
            grad_clip,
            debug,
            desc,
        )
        self.m = m
        self.step_size = step_size

    @property
    def model_cls(self):
        return GNNFlag

    def _train(self):
        self.model.train()
        loss = 0

        for batch in tqdm(self.loaders["train"], desc="Training"):
            batch = batch.to(self.device)

            if len(batch.x) > 1 and batch.batch[-1] > 0:
                is_labeled = batch.y == batch.y
                forward = lambda perturb: self.model(batch, perturb).to(torch.float32)[
                    is_labeled
                ]
                model_forward = (self.model, forward)
                y = batch.y.to(torch.float32)[is_labeled]
                perturb_shape = (batch.x.shape[0], self.param_emb_dim)
                loss_tensor, _ = attacks.flag(
                    model_forward,
                    perturb_shape,
                    y,
                    self.m,
                    self.step_size,
                    self.optimizer,
                    self.device,
                    self.loss_fn,
                )
                loss += loss_tensor.item()
            else:
                raise ValueError("how is this possible???")

            if self.debug:
                break

        return loss


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
        flag: bool = False,
        step_size: float = 1e-3,
        m: int = 3,
        grad_clip: float = 0,
        use_deeper: bool = False,
        debug: bool = False,
        desc: str = "",
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
            grad_clip,
            debug,
            desc,
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
        self.param_flag = flag
        self.param_step_size = step_size
        self.param_m = m
        self.param_use_deeper = use_deeper

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
            use_deeper=use_deeper,
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
            self.param_hidden_dim,
            self.epoch_stop,
            self.param_flag,
            self.param_step_size,
            self.param_m,
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


class DeeperGCNExperiment(GNNExperiment):
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
        block: str,
        conv_encode_edge: bool,
        add_virtual_node: bool,
        hidden_channels: int,
        conv: str,
        gcn_aggr: str,
        t: float,
        learn_t: bool,
        p: float,
        learn_p: bool,
        y: float,
        learn_y: bool,
        msg_norm: bool,
        learn_msg_scale: bool,
        norm: str,
        mlp_layers: int,
        graph_pooling: str,
        grad_clip: float = 0,
        debug: bool = False,
        desc: str = "",
    ):
        self.param_block = block
        self.param_conv_encode_edge = conv_encode_edge
        self.param_add_virtual_node = add_virtual_node
        self.param_hidden_channels = hidden_channels
        self.param_conv = conv
        self.param_gcn_aggr = gcn_aggr
        self.param_t = t
        self.param_learn_t = learn_t
        self.param_p = p
        self.param_learn_p = learn_p
        self.param_y = y
        self.param_learn_y = learn_y
        self.param_msg_norm = msg_norm
        self.param_learn_msg_scale = learn_msg_scale
        self.param_norm = norm
        self.param_mlp_layers = mlp_layers
        self.param_graph_pooling = graph_pooling
        super().__init__(
            "deeper-gcn",
            dropout,
            num_layers,
            emb_dim,
            epochs,
            lr,
            device,
            batch_size,
            num_workers,
            grad_clip,
            debug,
            desc,
        )

    def _get_model(self):
        return deeper.DeeperGCN(
            num_layers=self.param_num_layers,
            dropout=self.param_dropout,
            block=self.param_block,
            conv_encode_edge=self.param_conv_encode_edge,
            add_virtual_node=self.param_add_virtual_node,
            hidden_channels=self.param_hidden_channels,
            num_tasks=self.NUM_TASKS,
            conv=self.param_conv,
            gcn_aggr=self.param_gcn_aggr,
            t=self.param_t,
            learn_t=self.param_learn_t,
            p=self.param_p,
            learn_p=self.param_learn_p,
            y=self.param_y,
            learn_y=self.param_learn_y,
            msg_norm=self.param_msg_norm,
            learn_msg_scale=self.param_learn_msg_scale,
            norm=self.param_norm,
            mlp_layers=self.param_mlp_layers,
            graph_pooling=self.param_graph_pooling,
        ).to(self.device)


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
