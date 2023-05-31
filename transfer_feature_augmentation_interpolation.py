from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path
from tqdm import tqdm

import ignite.distributed as idist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import cosine_similarity

from datasets import load_datasets_for_cosine_sim, load_datasets_for_augm_interpolation
from resnets import load_backbone_out_blocks
from utils import Logger, get_engine_mock


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()

    logdir = Path(args.ckpt).parent

    args.origin_run_name = logdir.name

    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"feat_inv-{args.dataset}", args=args,
        job_type="eval_feature_augmentation_interpolation"

    )

    # DATASETS

    datasets = load_datasets_for_augm_interpolation(
        dataset=args.dataset,
        pretrain_data=args.pretrain_data,
        datadir=args.datadir,
        augmentation=args.augmentation
    )

    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               pin_memory=True)

    testloader  = build_dataloader(datasets['test'],  drop_last=False)
    transforms_dict = datasets["transforms"]

    ckpt_path = args.ckpt
    engine_mock = get_engine_mock(ckpt_path=ckpt_path)

    logger.log_msg(f"Evaluating {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = load_backbone_out_blocks(args)


    backbone.load_state_dict(ckpt['backbone'])

    build_model = partial(idist.auto_model, sync_bn=True)
    backbone = build_model(backbone)

    # EXTRACT FROZEN FEATURES
    logger.log_msg('collecting features ...')


    t_name_to_b_name_to_positive_sims = defaultdict(
        lambda: defaultdict(list)
    )
    t_name_to_b_name_to_negative_sims = defaultdict(
        lambda: defaultdict(list)
    )
    t_name_to_b_name_to_diff_sims = defaultdict(
        lambda: defaultdict(list)
    )

    with torch.no_grad():
        for i, (X, _) in tqdm(enumerate(testloader)):
            X_transformed = {
                t_name: t(X) for (t_name, t) in transforms_dict.items()
            }
            X_norm = X_transformed.pop("identity")

            bs = X_norm.shape[0]

            feats_norm = backbone(X_norm.to(device))

            for t_name, X_t in X_transformed.items():
                feats_t = backbone(X_t.to(device))
                assert feats_norm.keys() == feats_t.keys()

                for block_name, fn in feats_norm.items():
                    ft = feats_t[block_name]
                    fn_r = fn.reshape(bs, -1)
                    ft_r = ft.reshape(bs, -1)
                    positive_sim = cosine_similarity(fn_r, ft_r).mean().item()
                    negative_sim = cosine_similarity(
                        fn_r,
                        torch.flip(ft_r, [0])
                    ).mean().item()

                    t_name_to_b_name_to_positive_sims[t_name][block_name].append(positive_sim)
                    t_name_to_b_name_to_negative_sims[t_name][block_name].append(negative_sim)
                    t_name_to_b_name_to_diff_sims[t_name][block_name].append(positive_sim - negative_sim)

                    if (i+1) % args.print_freq == 0:
                        logger.log_msg(
                            f'{i + 1:3d} | {block_name} | {t_name} | pos: {np.mean(t_name_to_b_name_to_positive_sims[t_name][block_name]):.4f} | neg: {np.mean(t_name_to_b_name_to_negative_sims[t_name][block_name]):.4f})'
                        )

        metrics = dict()
        for (sim_kind, sim_dict) in [
            ("positive", t_name_to_b_name_to_positive_sims),
            ("negative", t_name_to_b_name_to_negative_sims),
            ("diff", t_name_to_b_name_to_diff_sims)
        ]:
            for t_name, b_name_to_sim in sim_dict.items():
                for block_name, sims in b_name_to_sim.items():
                    mean_sim = np.mean(sims)
                    std_sim = np.std(sims)
                    logger.log_msg(f'{sim_kind} {args.dataset} invariance of {block_name} to {t_name}: {mean_sim:.4f}±{std_sim:.4f}')
                    metrics[f"test_feature_augmentation_interpolation/{args.dataset}/{block_name}/{t_name}/{sim_kind}"] = mean_sim

        logger.log(
            engine=engine_mock, global_step=i,
            **metrics
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pretrain-data', type=str, default='stl10')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--datadir', type=str, default='/data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--augmentation', type=str, default='colorjitter')
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)
