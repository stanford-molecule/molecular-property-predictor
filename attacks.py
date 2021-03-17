"""
FLAG.
Copied from https://github.com/devnkong/FLAG/blob/main/ogb/attacks.py
"""

import torch


def flag(model_forward, perturb_shape, y, m, step_size, optimizer, device, criterion):
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = (
        torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    )
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= m

    for _ in range(m - 1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = criterion(out, y)
        loss /= m

    loss.backward()
    optimizer.step()

    return loss, out
