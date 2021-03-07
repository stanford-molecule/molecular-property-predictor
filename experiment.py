from functools import partial
from typing import Dict, List
from uuid import uuid4
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm import tqdm

from gnn import GNN


class Experiment:
    """
    Run a single experiment (i.e. train and then evaluate on train/test/validation splits)
    given the provided parameters, and store in a pickle file for analysis.

    Most of the code is either copied from https://github.com/snap-stanford/ogb or heavily inspired by it.
    """

    # we'll only be working on `molhiv` for this project
    DATASET_NAME = "ogbg-molhiv"
    NUM_TASKS = 1

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
    ):
        self.param_gnn_type = gnn_type
        self.param_dropout = dropout
        self.param_num_layers = num_layers
        self.param_emb_dim = emb_dim
        self.param_epochs = epochs
        self.param_batch_size = batch_size
        self.param_lr = lr
        self.num_workers = num_workers

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

    def run(self):
        """
        Run the experiment + print and store results.
        """
        loss_curve = []
        train_curve = []
        valid_curve = []
        test_curve = []

        self.times["start"] = datetime.now()
        for epoch in range(1, self.param_epochs + 1):
            print("=====Epoch {}".format(epoch))
            print("Training...")
            loss = self._train()
            print(f"Loss: {loss:.4f}")
            loss_curve.append(loss)

            print("Evaluating...")
            train_perf = self._eval(data_split="train")
            valid_perf = self._eval(data_split="valid")
            test_perf = self._eval(data_split="test")
            print({"Train": train_perf, "Validation": valid_perf, "Test": test_perf})

            train_curve.append(train_perf)
            valid_curve.append(valid_perf)
            test_curve.append(test_perf)

        self.times["end"] = datetime.now()
        best_val_epoch = int(np.argmax(np.array(valid_curve)))
        best_train = max(train_curve)

        print("Finished training!")
        print("Best validation score: {}".format(valid_curve[best_val_epoch]))
        print("Test score: {}".format(test_curve[best_val_epoch]))
        print("Best train: {}".format(best_train))

        self._store_results(loss_curve, train_curve, valid_curve, test_curve)

    def _store_results(
        self,
        loss_curve: List[float],
        train_curve: List[float],
        valid_curve: List[float],
        test_curve: List[float],
    ) -> None:
        """
        Stores all the curves and experiment parameters in a pickle file.
        """
        results = {
            "loss": loss_curve,
            "train": train_curve,
            "valid": valid_curve,
            "test": test_curve,
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

    @staticmethod
    def _get_loader(
        dataset: PygGraphPropPredDataset,
        split_idx: Dict[str, torch.Tensor],
        data_split: str,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        dataset_split = dataset[split_idx[data_split]]
        return DataLoader(
            dataset_split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def _train(self) -> float:
        self.model.train()
        loss = 0

        for _, batch in enumerate(tqdm(self.loaders["train"], desc="Iteration")):
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

        return loss

    def _eval(self, data_split: str) -> float:
        self.model.eval()
        y_true = []
        y_pred = []

        for _, batch in enumerate(tqdm(self.loaders[data_split], desc="Iteration")):
            batch = batch.to(self.device)

            if len(batch.x) > 1:
                with torch.no_grad():
                    pred = self.model(batch)

                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        return self.evaluator.eval(input_dict)[self.eval_metric]


if __name__ == "__main__":
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
    )
    exp.run()
