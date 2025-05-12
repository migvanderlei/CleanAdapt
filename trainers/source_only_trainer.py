import time
import torch

from utils.utils import AverageMeter, ProgressMeter, accuracy
from utils import optimizer as optim

def train_one_epoch(data_loader, model, criterion, optimizer, epoch, args, device, neptune_run=None):
    '''
    Executes one epoch of training on the train data
    '''

    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':1.2f')
    ce_loss = AverageMeter('CE', ':1.2f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, acc_cls, ce_loss],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(data_loader):
        data_time.update(time.time() - end)

        seq, targets, _ = batch
        if args.modality == 'Joint':
            seq, targets = [seq[0].to(device), seq[1].to(device)], targets.to(device)
        else:
            seq, targets = seq.to(device), targets.to(device)

        logits, _ = model(seq, args)
        pred_logits = (logits[0] + logits[1]) / 2 if args.modality == 'Joint' else logits[0]

        loss = criterion(pred_logits, targets)
        acc = accuracy(pred_logits, targets)[0]

        acc_cls.update(acc[0], targets.size(0))
        ce_loss.update(loss, targets.size(0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx)

        if neptune_run is not None:
            neptune_run["train/batch/loss"].append(loss.item())
            neptune_run["train/batch/accuracy"].append(acc[0].item())
            neptune_run["train/batch/idx"].append(batch_idx)
            neptune_run["train/batch/epoch"].append(epoch)

    if neptune_run is not None:
        neptune_run["train/epoch/loss"].append(ce_loss.avg)
        neptune_run["train/epoch/accuracy"].append(acc_cls.avg)
        neptune_run["train/epoch/idx"].append(epoch)

    return acc_cls.avg, ce_loss.avg
