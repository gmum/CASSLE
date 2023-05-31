from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import ignite.distributed as idist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.decomposition import PCA

from datasets import load_datasets_for_cosine_sim
from resnets import load_backbone_out_blocks
from utils import Logger, get_engine_mock
from sklearn.metrics import r2_score

def stringer_get_powerlaw(ss, trange):
    # COPIED FROM Stringer+Pachitariu 2018b github repo! (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    ''' fit exponent to variance curve'''
    logss = np.log(np.abs(ss))
    y = logss[trange][:, np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:, np.newaxis], np.ones((nt, 1))), axis=1)
    w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, ss.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:, np.newaxis], np.ones((ss.size, 1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))
    alpha = b[0]
    max_range = 500 if len(ss) >= 512 else len(
        ss) - 10  # subtracting 10 here arbitrarily because we want to avoid the last tail!
    fit_R2 = r2_score(y_true=logss[trange[0]:max_range], y_pred=np.log(np.abs(ypred))[trange[0]:max_range])
    try:
        fit_R2_100 = r2_score(y_true=logss[trange[0]:100], y_pred=np.log(np.abs(ypred))[trange[0]:100])
    except:
        fit_R2_100 = None
    return alpha, ypred, fit_R2, fit_R2_100


def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()

    logdir = Path(args.ckpt).parent

    args.origin_run_name = logdir.name

    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"feat_pca-{args.dataset}", args=args,
        job_type="eval_pca"

    )

    datasets = load_datasets_for_cosine_sim(
        dataset=args.dataset,
        pretrain_data=args.pretrain_data,
        datadir=args.datadir,
    )

    build_dataloader = partial(idist.auto_dataloader,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False,
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

    latents = []

    with torch.no_grad():
        for i, (X, _) in enumerate(testloader):
            X_transformed = {
                t_name: t(X) for (t_name, t) in transforms_dict.items()
            }
            X_norm = X_transformed.pop("identity")

            feats_norm = backbone(X_norm.to(device))["backbone_out"].detach().cpu().numpy()

            latents.append(feats_norm)

    latents = np.concatenate(latents)

    pca = PCA().fit(latents)
    exp_var =  pca.explained_variance_ratio_
    
    cum = np.cumsum(exp_var)

    for i, (e, c) in enumerate(zip(exp_var, cum)):
        logger.log(
            engine=engine_mock, global_step=-1,
            component=i,
            **{
                f"test_pca/exp_variance/{args.dataset}": e,
                f"test_pca/cum_variance/{args.dataset}": c,
            }
        )
        
    alpha, _, R2, r2_range = stringer_get_powerlaw(
        exp_var, np.arange(5, 50)
    )
    logger.log(
            engine=engine_mock, global_step=-1,
            **{
                f"test_pca/a-req/alpha/{args.dataset}": alpha,
                f"test_pca/a-req/r2/{args.dataset}": R2,
                f"test_pca/a-req/r2_100/{args.dataset}": r2_range,
            }
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
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)
