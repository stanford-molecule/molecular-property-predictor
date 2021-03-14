import torch
import torch.nn.functional as F

from tqdm import tqdm

loss_fn = F.nll_loss
softmax = F.log_softmax


def _flag(model, data, device, y, step_size, m, hidden_dim):
    forward = lambda p: model(
        data.x, data.edge_index, data.batch, data.edge_attr, perturb=p
    )
    perturb_shape = (data.x.shape[0], hidden_dim)
    perturb = (
        torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    )
    perturb.requires_grad_()
    out, kl = forward(perturb)
    out = softmax(out, dim=-1)
    loss = loss_fn(out, y, reduction="mean")
    loss /= m
    kl /= m

    for _ in range(m - 1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out, kl = forward(perturb)
        out = softmax(out, dim=-1)
        loss = loss_fn(out, y)
        loss /= m
        kl /= m
    return loss, kl


def train(
    model,
    optimizer,
    loader,
    device,
    hidden_dim,
    epoch_stop=None,
    flag: bool = False,
    step_size: float = 1e-3,
    m: int = 3,
):
    model.train()
    total_ce_loss, total_kl_loss = 0, 0

    for idx, data in enumerate(tqdm(loader, desc="Training")):
        data.to(device)
        y = data.y.view(-1)

        optimizer.zero_grad()
        if flag:
            loss, kl = _flag(model, data, device, y, step_size, m, hidden_dim)
        else:
            out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
            out = softmax(out, dim=-1)
            loss = loss_fn(out, y, reduction="mean")

        loss.backward()
        optimizer.step()

        total_ce_loss += loss.item() * data.y.size(0)
        total_kl_loss += kl.item() * data.y.size(0)

        if epoch_stop and idx >= epoch_stop:
            break

    return total_ce_loss, total_kl_loss


def kl_train(
    model,
    optimizer,
    loader,
    device,
    hidden_dim,
    epoch_stop=None,
    flag: bool = False,
    step_size: float = 1e-3,
    m: int = 3,
):
    # TODO: skip FLAG on KL train?
    total_kl_loss = 0.0
    total_ce_loss = 0.0

    optimizer.zero_grad()
    for idx, data in enumerate(tqdm(loader, desc="KL train")):
        data.to(device)
        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)
        out = softmax(out, dim=-1)
        loss = loss_fn(out, data.y.view(-1), reduction="mean")
        kl.backward()

        total_kl_loss += kl.item() * data.y.size(0)
        total_ce_loss += loss.item() * data.y.size(0)

        if epoch_stop and idx >= epoch_stop:
            break

    optimizer.step()

    return total_ce_loss, total_kl_loss


@torch.no_grad()
def evaluate(model, loader, device, evaluator=None, data_split="", epoch_stop=None):
    model.eval()
    loss, kl_loss, correct = 0, 0, 0
    y_pred, y_true = [], []

    for idx, data in enumerate(tqdm(loader, desc=data_split)):
        data.to(device)
        out, kl = model(data.x, data.edge_index, data.batch, data.edge_attr)

        y_pred.append(out[:, 1])
        y_true.append(data.y)

        out = F.log_softmax(out, dim=-1)
        pred = out.max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        loss += F.nll_loss(out, data.y.view(-1), reduction="mean").item() * data.y.size(
            0
        )
        kl_loss += kl.item() * data.y.size(0)

        if epoch_stop and idx >= epoch_stop:
            break

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    if evaluator is None:
        acc = correct / len(loader.dataset)
    else:
        acc = evaluator.eval({"y_pred": y_pred.view(y_true.shape), "y_true": y_true})[
            evaluator.eval_metric
        ]

    return acc, loss, kl_loss
