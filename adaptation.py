import numpy as np
import datetime
import os
import sys
import random
import pickle
import builtins
from itertools import chain
import neptune
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings("ignore")

from parse_args import create_parser
from models.model import AdaptationModel
from dataset.get_datasets import get_data, get_weak_transforms, get_strong_transforms, get_dataloader
from utils.utils import set_seed, adjust_learning_rate, save_checkpoint, count_parameters, update_ema_variables
from trainers import adapter_trainer, validation
from utils.losses import SCELoss


idx2cls = {0: "climb", 1: "fencing", 2: "golf", 3: "kick_ball",
            4: "pullup", 5: "punch", 6: "pushup", 7: "ride_bike", 8: "ride_horse",
            9: "shoot_ball", 10: "shoot_bow", 11: "walk"}

load_dotenv()

NEPTUNE_API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed != -1:
        set_seed(args.seed)

    print("\n############################################################################\n")
    print("Experimental Configs: ", args)
    print("\n############################################################################\n")

    print("==> Training for Label Correction.. [{}]".format(args.modality))

    result_dir = os.path.join(args.save_dir, '_'.join(
        (args.source_dataset, args.target_dataset, args.adaptation_mode, args.modality)))
    log_dir = os.path.join(result_dir, 'logs')

    save_dir = os.path.join(result_dir, 'checkpoints-ts' if args.use_ema else 'checkpoints')
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    run_name = "-".join(["r-", str(args.r), args.source_dataset, args.target_dataset, args.modality])

    run = neptune.init_run(
        project="migvanderlei/cleanadapt-video-domain-adaptation",
        api_token=NEPTUNE_API_TOKEN,
        name=run_name,
        tags=["domain-adaptation", args.adaptation_mode],
        capture_stdout=False,
        capture_stderr=False
    )

    run["hyperparameters"] = vars(args)

    best_target_acc_t = 0
    best_target_acc_s = 0

    weak_transform_train = get_weak_transforms(args, 'train')
    strong_transform_train = get_strong_transforms(args, 'train')
    transform_val = get_weak_transforms(args, 'val')

    print("==> Constructing the target dataloaders..")
    target_train_dataset = get_data([weak_transform_train, strong_transform_train], args, 'train', args.target_dataset, args.pseudo_label_path)
    target_val_dataset = get_data(transform_val, args, 'val', args.target_dataset)

    target_train_loader = get_dataloader(args, 'train', target_train_dataset)
    target_val_loader = get_dataloader(args, 'val', target_val_dataset)

    print("==> Loading the {} model for label correction model..".format(args.adaptation_mode))
    model = AdaptationModel(args, device)
    model = torch.nn.parallel.DataParallel(model, device_ids=list(range(args.gpus))).to(device)

    print("==> Loading pretrained weights from {}".format(args.pretrained_weight_path))
    checkpoint = torch.load(args.pretrained_weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(model, device, args.ema_decay)
        print("Total # of trainable params in teacher N/w: {}M".format(count_parameters(ema_model.ema)/1e6))            
    else:
        ema_model = None

    print("Total # of trainable params in student N/w: {}M".format(count_parameters(model)/1e6))

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), args.lr,
                          weight_decay=args.weight_decay, momentum=args.momentum)

    for epoch in range(0, args.num_epochs):
        target_train_loader.dataset._update_video_list(select_all=True)

        if args.use_ema:
            adapter_trainer.sample_selection_step_teacher_student(target_train_loader, ema_model, args, device, r=args.r)
        else:
            adapter_trainer.sample_selection_step(target_train_loader, model, args, device, r=args.r)

        adjust_learning_rate(optimizer, epoch, args)

        train_acc, train_loss = adapter_trainer.train_one_epoch(
            target_train_loader, model, ema_model, criterion, optimizer, epoch, args, device)

        target_val_epoch_acc_s, target_val_epoch_loss = validation.validate(target_val_loader, model, epoch, args, device)

        if args.use_ema:
            target_val_epoch_acc_t, target_val_epoch_loss = validation.validate(target_val_loader, ema_model.ema, epoch, args, device)
            print("Epoch: [{}/{}] [Validation][Teacher Model] Target accuracy: {:.2f} Target loss: {:.4f}".format(
                epoch, args.num_epochs, target_val_epoch_acc_t, target_val_epoch_loss))

        print("Epoch: [{}/{}] [Validation][Student Model] Target accuracy: {:.2f} Target loss: {:.4f}".format(
            epoch, args.num_epochs, target_val_epoch_acc_s, target_val_epoch_loss))

        # Log data to Neptune
        log_data = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"]
        }

        if args.modality == "RGB":
            log_data["[Train] Accuracy - RGB"] = train_acc[0]
            log_data["[Train] Loss - RGB"] = train_loss[0]
        elif args.modality == "Flow":
            log_data["[Train] Accuracy - Flow"] = train_acc[0]
            log_data["[Train] Loss - Flow"] = train_loss[0]
        elif args.modality == "Joint":
            log_data["[Train] Accuracy - RGB"] = train_acc[0]
            log_data["[Train] Accuracy - Flow"] = train_acc[1]
            log_data["[Train] Loss - RGB"] = train_loss[0]
            log_data["[Train] Loss - Flow"] = train_loss[1]

        log_data["[Validation][Student] Accuracy"] = target_val_epoch_acc_s
        log_data["[Validation][Student] Loss"] = target_val_epoch_loss.item()

        if args.use_ema and ema_model is not None:
            log_data["[Validation][Teacher] Accuracy"] = target_val_epoch_acc_t
            log_data["[Validation][Teacher] Loss"] = target_val_epoch_loss.item()

        run.log_metrics(log_data, step=epoch)

        if args.use_ema and target_val_epoch_acc_t > best_target_acc_t:
            best_target_acc_t = target_val_epoch_acc_t
            print("[Teacher] Found target best acc {:.2f} at epoch {}.".format(best_target_acc_t, epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'i3d',
                'state_dict': ema_model.ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_target_val_acc': best_target_acc_t,
            }, False, checkpoint_dir=save_dir, epoch=epoch + 1)

        if target_val_epoch_acc_s > best_target_acc_s:
            best_target_acc_s = target_val_epoch_acc_s
            print("[Student] Found target best acc {:.2f} at epoch {}.".format(best_target_acc_s, epoch))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': 'i3d',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_target_val_acc': best_target_acc_s,
            }, False, checkpoint_dir=save_dir, epoch=epoch + 1)

    print("==> Training done!")
    if args.use_ema:
        print("==> [Target][Teacher] Best accuracy {:.2f}".format(best_target_acc_t))
    print("==> [Target][Student] Best accuracy {:.2f}".format(best_target_acc_s))

    with open("output-{}-{}-{}.txt".format(args.source_dataset, args.target_dataset, args.modality), "a") as text_file:
        if args.use_ema:
            text_file.write("lr: {}, ema: {}, r: {} [Target][Teacher] Best accuracy {:.2f}\n".format(
                args.lr, args.ema_decay, args.r, best_target_acc_t))
        text_file.write("lr: {}, ema: {}, r: {} [Target][Student] Best accuracy {:.2f}\n".format(
            args.lr, args.ema_decay, args.r, best_target_acc_s))
        


if __name__ == "__main__":

    parser = create_parser()
    
    args = parser.parse_args()
    main(args)
