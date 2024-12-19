import torch
import torch.nn.functional as F


def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    print('model size: {:.2f} MB'.format(size_all_mb))


@torch.no_grad()
def accuracy(logits, y):
    probs = F.softmax(logits, dim=-1)
    y_pred = probs.argmax(dim=-1)
    accuracy = 100 * ((y_pred == y).sum() / y_pred.shape[0])

    return accuracy


def load_from_checkpoint(path, model, optimizer=None, lr_scheduler=None, device="cpu"):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if lr_scheduler:
        lr_scheduler = lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    return model, optimizer, lr_scheduler
