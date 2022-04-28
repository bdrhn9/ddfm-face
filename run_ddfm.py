#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import modules.utils_torchvision as utils
from modules.utils import (
    evaluate_l2,
    serialize_target,
    separate_irse_bn_paras,
    get_dataloader,
)
from loss.subcluster_ddfm import subcluster_ddfm_loss
from backbones.iresnet_torch import iresnet50

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = str(3)


def train_one_epoch(
    model,
    optimizer,
    criterion_center,
    optimizer_center,
    scheduler,
    dataloader,
    device,
    epoch,
    num_subset,
    num_subcluster,
    args,
):

    model.train(), criterion_center.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "img/s", utils.SmoothedValue(window_size=10, fmt="{value:.1f}")
    )
    header = "Epoch: [{}]".format(epoch)
    for image, target in metric_logger.log_every(dataloader, args.print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        serialized_target = serialize_target(target, num_subcluster, num_subset)

        if args.distributed:
            centers = criterion_center.module.centers
        else:
            centers = criterion_center.centers

        features = model(image)

        intra, inter, triplet = criterion_center(features, serialized_target)
        loss = args.intrak * intra + args.interk * inter + args.tripletk * triplet

        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_center.step()
        scheduler.step()

        preds, acc1 = evaluate_l2(
            features, centers, target[:, 0], subcenters_available=True
        )

        batch_size = image.shape[0]
        metric_logger.update(
            loss=loss.item(),
            intra=intra.item(),
            inter=inter.item(),
            triplet=triplet.item(),
            lr=optimizer.param_groups[0]["lr"],
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    metric_logger.synchronize_between_processes()
    print(" *Train Acc@1 {top1.global_avg:.3f} ".format(top1=metric_logger.acc1))


def evaluate_majority_voting(model, criterion, dataloader, device, epoch, args):
    model.eval(), criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    all_preds_l2, all_preds_l2_norm, all_labels = [], [], []
    with torch.no_grad():
        for image, target in metric_logger.log_every(
            dataloader, args.print_freq, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            feats = model(image)

            if args.distributed:
                centers = criterion.module.centers
            else:
                centers = criterion.centers

            preds_l2, acc_l2 = evaluate_l2(
                feats, centers, target[:, 0], subcenters_available=True
            )
            preds_l2_norm, acc_l2_norm = evaluate_l2(
                F.normalize(feats),
                F.normalize(centers, dim=2),
                target[:, 0],
                subcenters_available=True,
            )
            all_preds_l2.append(preds_l2.cpu())
            all_preds_l2_norm.append(preds_l2_norm.cpu())
            all_labels.append(target.cpu())

    preds_l2_norm = torch.cat(all_preds_l2_norm, 0)
    preds_l2 = torch.cat(all_preds_l2, 0)
    labels = torch.cat(all_labels, 0)
    print(labels.shape)

    if args.distributed:
        preds_l2_dist = [None] * dist.get_world_size()
        preds_l2_norm_dist = [None] * dist.get_world_size()
        labels_dist = [None] * dist.get_world_size()
        dist.barrier()

        dist.all_gather_object(preds_l2_dist, preds_l2)
        dist.all_gather_object(preds_l2_norm_dist, preds_l2_norm)
        dist.all_gather_object(labels_dist, labels)

        preds_l2 = torch.cat(preds_l2_dist, 0)
        preds_l2_norm = torch.cat(preds_l2_norm_dist, 0)
        labels = torch.cat(labels_dist, 0)
        print(labels.shape)

    if utils.is_main_process():
        num_set = []
        for cls_ in np.unique(labels[:, 0]):
            idxs_ = labels[:, 0] == cls_
            for set_ in np.unique(labels[idxs_, 1]):
                res = np.argmax(
                    np.bincount(
                        preds_l2[(labels[:, 0] == cls_) * (labels[:, 1] == set_)]
                    )
                )
                num_set.append(res == cls_)
        test_set_acc_l2 = (float(np.count_nonzero(num_set)) / len(num_set)) * 100.0
        test_acc_l2 = (float((labels[:, 0] == preds_l2).sum()) / len(labels)) * 100.0

        num_set = []
        for cls_ in np.unique(labels[:, 0]):
            idxs_ = labels[:, 0] == cls_
            for set_ in np.unique(labels[idxs_, 1]):
                res = np.argmax(
                    np.bincount(
                        preds_l2_norm[(labels[:, 0] == cls_) * (labels[:, 1] == set_)]
                    )
                )
                num_set.append(res == cls_)
        test_set_acc_l2_norm = (float(np.count_nonzero(num_set)) / len(num_set)) * 100.0
        test_acc_l2_norm = (
            float((labels[:, 0] == preds_l2_norm).sum()) / len(labels)
        ) * 100.0

        print("Test Set_Acc_L2 %.3f" % test_set_acc_l2)
        print("Test Acc@1_L2 %.3f" % test_acc_l2)
        print("Test Set_Acc_COSINE %.3f" % test_set_acc_l2_norm)
        print("Test Acc@1_COSINE %.3f" % test_acc_l2_norm)
        return test_set_acc_l2, test_set_acc_l2_norm


def main(args):
    save_dir = os.path.join(
        "logs",
        "esogu_faces",
        "lr%.7fmargin%.2f"
        "intrak%.1finterk%.1ftripletk%.1f"
        "randomcenters%s_%s"
        % (
            args.lr,
            args.margin,
            args.intrak,
            args.interk,
            args.tripletk,
            bool(args.train_centers_path),
            args.output_note,
        ),
    )
    utils.mkdir(save_dir)
    with open(os.path.join(save_dir, "commandline_args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(12345)

    dataloader, dataloader_test = get_dataloader(args)

    print("Creating model")

    model = iresnet50(pretrained=True, make_orthonormal=args.orthonormal)
    model.to(device)

    criterion_center = subcluster_ddfm_loss(
        num_classes=dataloader.dataset.num_classes,
        num_subset=dataloader.dataset.num_subset,
        num_subcluster=dataloader.dataset.num_subcluster,
        feat_dim=model.feat_dim,
        precalc_centers=args.train_centers_path,
        margin=args.margin,
    )
    criterion_center.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
    backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(model)
    optimizer = torch.optim.Adam(
        [
            {"params": backbone_paras_wo_bn, "weight_decay": args.weight_decay},
            {"params": backbone_paras_only_bn},
        ],
        lr=args.lr,
    )
    # optimizer = torch.optim.SGD([{'params': backbone_paras_wo_bn, 'weight_decay': args.weight_decay},
    #                             {'params': backbone_paras_only_bn}], lr = args.lr, momentum = args.momentum)

    optimizer_center = torch.optim.SGD(criterion_center.parameters(), lr=0.5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader)
    )

    model_without_ddp = model
    criterion_center_without_ddp = criterion_center
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        criterion_center = torch.nn.parallel.DistributedDataParallel(
            criterion_center, device_ids=[args.gpu]
        )
        criterion_center_without_ddp = criterion_center.module

    best_test_acc1_l2, test_acc1_l2 = 0.0, 0.0
    best_test_acc1_l2_norm, test_acc1_l2_norm = 0.0, 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        criterion_center_without_ddp.load_state_dict(checkpoint["criterion_center"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_center.load_state_dict(checkpoint["optimizer_center"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        best_test_acc1_l2 = checkpoint["best_test_acc1_l2"]
        best_test_acc1_l2_norm = checkpoint["best_test_acc1_l2_norm"]

    if args.test_only:
        return evaluate_majority_voting(
            model, criterion_center, dataloader_test, device, epoch=0, args=args
        )

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dataloader.batch_sampler.set_epoch(epoch)

        train_one_epoch(
            model,
            optimizer,
            criterion_center,
            optimizer_center,
            scheduler,
            dataloader,
            device,
            epoch,
            dataloader.dataset.num_subset,
            dataloader.dataset.num_subcluster,
            args,
        )

        if epoch % args.eval_freq == 0 and epoch != 0:

            test_acc1_l2, test_acc1_l2_norm = evaluate_majority_voting(
                model, criterion_center, dataloader_test, device, epoch, args=args
            )

            is_best_l2 = test_acc1_l2 > best_test_acc1_l2
            is_best_l2_norm = test_acc1_l2_norm > best_test_acc1_l2_norm

            best_test_acc1_l2 = max(test_acc1_l2, best_test_acc1_l2)
            best_test_acc1_l2_norm = max(test_acc1_l2_norm, best_test_acc1_l2_norm)

            print(
                "L2 %.3f Set Accuracy, %.3f Best Set Acc"
                % (test_acc1_l2, best_test_acc1_l2)
            )
            print(
                "COSINE %.3f Set Accuracy, %.3f Best Set Acc"
                % (test_acc1_l2_norm, best_test_acc1_l2_norm)
            )

            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "criterion_center": criterion_center_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_center": optimizer_center.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "test_acc1": test_acc1_l2,
                "best_test_acc1": best_test_acc1_l2,
                "test_acc1_l2_norm": test_acc1_l2_norm,
                "best_test_acc1_l2_norm": best_test_acc1_l2_norm,
                "args": args,
            }

            if epoch % args.save_freq == 0:
                utils.save_on_master(
                    checkpoint, os.path.join(save_dir, "model_{}.pth".format(epoch))
                )
            if is_best_l2:
                utils.save_on_master(
                    checkpoint, os.path.join(save_dir, "best_checkpoint_l2.pth")
                )
                is_best_l2 = False
            if is_best_l2_norm:
                utils.save_on_master(
                    checkpoint, os.path.join(save_dir, "best_checkpoint_l2_norm.pth")
                )
                is_best_l2_norm = False

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=3,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=5e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--interk", default=1.0, type=float, help="decrease lr by a factor of lr-gamma"
    )
    parser.add_argument(
        "--intrak", default=1.0, type=float, help="decrease lr by a factor of lr-gamma"
    )
    parser.add_argument(
        "--tripletk",
        default=1.0,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--margin", default=24.0, type=float, help="decrease lr by a factor of lr-gamma"
    )
    parser.add_argument("--print-freq", default=1000, type=int, help="print frequency")
    parser.add_argument("--eval-freq", default=1, type=int, help="print frequency")
    parser.add_argument("--save-freq", default=5, type=int, help="print frequency")
    parser.add_argument(
        "--folder-path",
        default="./data/esogu_faces",
        help="additional note to output folder",
    )
    parser.add_argument(
        "--train-meta-path",
        default="./data/esogu_faces/train_meta_3_clustured_v3_kmeans.npy",
        help="additional note to output folder",
    )
    parser.add_argument(
        "--train-centers-path",
        default="./data/esogu_faces/train_meta_3_clustured_centers_v3_kmeans.npy",
        help="additional note to output folder",
    )
    parser.add_argument(
        "--output-note", default="subc3_kmeans", help="additional note to output folder"
    )
    parser.add_argument(
        "--test-meta-path",
        default="./data/esogu_faces/test_meta.npy",
        help="additional note to output folder",
    )
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        default=False,
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--orthonormal",
        dest="orthonormal",
        default=False,
        help="Make model orthonormal for common vector approach",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
