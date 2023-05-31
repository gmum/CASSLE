from argparse import ArgumentParser
from functools import partial
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite.distributed as idist
import wandb

from datasets import load_datasets
from models import load_backbone
from trainers import collect_features
from utils import Logger, get_engine_mock


def build_step(X, Y, classifier, optimizer, w, criterion_fn):
    def step():
        optimizer.zero_grad()
        loss = criterion_fn(classifier(X), Y, reduction='sum')
        for p in classifier.parameters():
            loss = loss + p.pow(2).sum().mul(w)
        loss.backward()
        return loss
    return step

def l1_criterion_fn(normalize_input: bool=False, normalize_target: bool=False):
    def fn(input, target, **kwargs):
        if normalize_input:
            input = nn.functional.normalize(input, dim=1)
        if normalize_target:
            target = nn.functional.normalize(target, dim=1)

        return F.l1_loss(input, target, **kwargs)

    return fn

def r2_fn(normalize_input: bool=False, normalize_target: bool=False):

    def r2_score(y, x):
        """https://github.com/ruchikachavhan/amortized-invariance-learning-ssl/blob/f832c17ce3d59c7a16cfba3caeac4438034cab23/r2score.py#L4"""
        if normalize_input:
            x = nn.functional.normalize(x, dim=1)
        if normalize_target:
            y = nn.functional.normalize(y, dim=1)

        print("r2 pre reshape", x.shape, y.shape)
        x = x.flatten().detach().cpu().numpy()
        y = y.flatten().detach().cpu().numpy()


        A = np.vstack([x, np.ones(len(x))]).T

        print("r2: x, y, A", x.shape, y.shape, A.shape)
        # Use numpy's least squares function
        m, c = np.linalg.lstsq(A, y)[0]

        # print(m, c)
        # 1.97 -0.11

        # Define the values of our least squares fit
        f = m * x + c

        # print(f)
        # [ 1.86  3.83  5.8   7.77  9.74]

        # Calculate R^2 explicitly
        yminusf2 = (y - f)**2
        sserr = sum(yminusf2)
        mean = float(sum(y)) / float(len(y))
        yminusmean2 = (y - mean)**2
        sstot = sum(yminusmean2)
        R2 = 1. -(sserr / sstot)
        return R2

    return r2_score


def compute_accuracy(X, Y, classifier, metric_name_or_fn):
    with torch.no_grad():

        preds = classifier(X)
        if metric_name_or_fn in ["top1", "class-avg"]:
            preds = preds.argmax(1)

        if metric_name_or_fn == 'top1':
            acc = (preds == Y).float().mean().item()
        elif metric_name_or_fn == 'class-avg':
            total, count = 0., 0.
            for y in range(0, Y.max().item()+1):
                masks = Y == y
                if masks.sum() > 0:
                    total += (preds[masks] == y).float().mean().item()
                    count += 1
            acc = total / count

        else:
            assert not isinstance(metric_name_or_fn, str)
            acc = metric_name_or_fn(Y, preds)

        # else:
        #     raise Exception(f'Unknown metric: {metric_name_or_fn}')
    return acc


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()


    ckpt_parents = set([Path(c).parent for c in args.ckpt])
    assert len(set(ckpt_parents)) == 1, f"Expected a single checkpoints directory but got {ckpt_parents}"
    logdir = list(ckpt_parents)[0]

    args.origin_run_name = logdir.name
    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"lin-{args.dataset}", args=args,
        job_type="eval_linear"
    )

    # DATASETS
    datasets = load_datasets(dataset=args.dataset,
                             datadir=args.datadir,
                             pretrain_data=args.pretrain_data)
    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)
    trainloader = build_dataloader(datasets['train'], drop_last=False)
    valloader   = build_dataloader(datasets['val'],   drop_last=False)
    testloader  = build_dataloader(datasets['test'],  drop_last=False)
    trainvalloader = build_dataloader(datasets["trainval"], drop_last=False)

    num_classes = datasets['num_classes']

    if args.metric in ["top1", 'class-avg']:
        criterion_fn = F.cross_entropy
        metric = args.metric
    # elif args.dataset == "celeba":
    #     criterion_fn = l1_criterion_fn(normalize_target=True)
    #     metric = r2_fn(normalize_target=True)
    # elif args.dataset == "lspose":
    #     criterion_fn = l1_criterion_fn(normalize_target=True, normalize_input=True)
    #     metric = r2_fn(normalize_target=True, normalize_input=True)
    # elif args.dataset == "300w":
    elif args.metric == "r2":
        criterion_fn = l1_criterion_fn()
        metric = r2_fn()
    else:
        raise NotImplementedError((args.dataset, args.metric))

    for ckpt_path in sorted(args.ckpt):
        engine_mock = get_engine_mock(ckpt_path=ckpt_path)

        logger.log_msg(f"Evaluating {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        
        
        backbone = load_backbone(args)
        backbone.load_state_dict(ckpt['backbone'])

        build_model = partial(idist.auto_model, sync_bn=True)
        backbone   = build_model(backbone)

        # EXTRACT FROZEN FEATURES
        logger.log_msg('collecting features ...')
        X_train, Y_train = collect_features(backbone, trainloader, device, normalize=False)
        X_val,   Y_val   = collect_features(backbone, valloader,   device, normalize=False)
        X_test,  Y_test  = collect_features(backbone, testloader,  device, normalize=False)
        X_trainval, Y_trainval  = collect_features(backbone, trainvalloader,  device, normalize=False)

        print(f"{X_train.shape=}, {Y_train.shape=}")
        print(f"{X_val.shape=}, {Y_val.shape=}")
        print(f"{X_test.shape=}, {Y_test.shape=}")
        print(f"{X_trainval.shape=}, {Y_trainval.shape=}")

        classifier = nn.Linear(args.num_backbone_features, num_classes).to(device)
        optim_kwargs = {
            'line_search_fn': 'strong_wolfe',
            'max_iter': 5000,
            'lr': 1.,
            'tolerance_grad': 1e-10,
            'tolerance_change': 0,
        }
        logger.log_msg('collecting features ... done')

        best_acc = 0.
        best_w = 0.
        best_classifier = None
        for w in torch.logspace(-6, 5, steps=45).tolist():
            optimizer = optim.LBFGS(classifier.parameters(), **optim_kwargs)

            optimizer.step(
                build_step(X_train, Y_train, classifier, optimizer, w, criterion_fn=criterion_fn))
            acc = compute_accuracy(X_val, Y_val, classifier, metric)

            if best_acc < acc:
                best_acc = acc
                best_w = w
                best_classifier = deepcopy(classifier)

            logger.log_msg(f'w={w:.4e}, acc={acc:.4f}')
            logger.log(
                engine=engine_mock, global_step=-1,
                **{
                    "w": w,
                    f"val_linear/{args.dataset}": acc
                }
            )
            if wandb.run is not None:
                wandb.log({
                    "w": w,
                    f"val_linear/{args.dataset}": acc
                })

        logger.log_msg(f'BEST: w={best_w:.4e}, acc={best_acc:.4f}')

        # X = torch.cat([X_train, X_val], 0)
        # Y = torch.cat([Y_train, Y_val], 0)
        optimizer = optim.LBFGS(best_classifier.parameters(), **optim_kwargs)
        optimizer.step(build_step(X_trainval, Y_trainval, best_classifier, optimizer, best_w, criterion_fn=criterion_fn))
        acc = compute_accuracy(X_test, Y_test, best_classifier, metric_name_or_fn=metric)
        logger.log_msg(f'test acc={acc:.4f}')
        logger.log(
            engine=engine_mock, global_step=-1,
            **{
                f"test_linear/{args.dataset}": acc
            }
        )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, nargs="+")
    parser.add_argument('--pretrain-data', type=str, default='stl10')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--metric', type=str, default='top1', choices=["top1", 'class-avg', "r2"])
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)

