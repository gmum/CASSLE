from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from pathlib import Path

import ignite.distributed as idist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import cosine_similarity

from cond_utils import AugProjector, AUG_STRATEGY, AUG_HN_TYPES, AUG_INJECTION_TYPES
from datasets import load_datasets_for_cosine_sim
from resnets import load_backbone_out_blocks
from transforms import extract_aug_descriptors, RandomResizedCrop
from utils import Logger, get_engine_mock
from models import load_backbone, load_mlp, load_ss_predictor
import torch.nn.functional as F

PROJ_OUT = "projector_out"
PROJ_OUT_MIXED = "projector_out_mixed_descriptors"

BKB_OUT = "backbone_out"
MLP = "mlp"
AUG_COND = "aug_cond"

proj_sims = defaultdict(list)

def load_projector(args, ckpt):
    projector_type = None
    projector_kwargs = None
    if "moco-" in args.origin_run_name:
        projector_kwargs = dict(
            n_in=args.num_backbone_features,
            n_hidden=args.num_backbone_features,
            n_out=128,
            num_layers=2,
            last_bn=False
        )
    elif "mocov3" in args.origin_run_name:
        projector_kwargs = dict(
            n_in=args.num_backbone_features,
            n_hidden=4096,
            n_out=256,
            num_layers=3,
            last_bn=True,
            last_bn_affine=False
        )
    elif "simsiam" in args.origin_run_name:
        projector_kwargs = dict(
            n_in=args.num_backbone_features,
            n_hidden=2048,
            n_out=2048,
            num_layers=3,
            last_bn=True
        )

    elif "simclr" in args.origin_run_name:
        projector_kwargs = dict(
            n_in=args.num_backbone_features,
            n_hidden=args.num_backbone_features,
            n_out=128,
            num_layers=2,
            last_bn=False
        )
    elif "byol" in args.origin_run_name:
        projector_kwargs = dict(
            n_in=args.num_backbone_features,
            n_hidden=4096,
            n_out=256,
            num_layers=2,
            last_bn=False
        )
    elif "barlow_twins" in args.origin_run_name:
        projector_kwargs = dict(
            n_in=args.num_backbone_features,
            n_hidden=8192,
            n_out=8192,
            num_layers=3,
            last_bn=True,
            last_bn_affine=False,
        )


    aug_projector_kwargs = None
    if projector_kwargs is not None:
        aug_projector_kwargs = dict(
            args=args,
            proj_out_dim=projector_kwargs["n_out"],
            proj_depth=projector_kwargs.get("num_layers", 2),
            proj_hidden_dim=projector_kwargs.get("n_hidden"),
            projector_last_bn=projector_kwargs.get("last_bn", False),
            projector_last_bn_affine=projector_kwargs.get("last_bn_affine", False)
        )

    try:
        try:
            projector = load_mlp(**projector_kwargs)
            projector.load_state_dict(ckpt["projector"])
            projector_type = MLP


        except Exception as e:
            print(f"Could not load raw mlp projector bc of {e}. Trying AugProjector.")
            # TODO load args from wandb or args.json
            framework, architecture, *rest = args.origin_run_name.split("-")
            print(f"Parsing #1: {framework=}, {architecture=}, {rest=}")

            try:
                dataset, aug_treatment, depth, width, inj_type, *rest = "-".join(rest).split("_")
                print(f"Parsing #2: {dataset=}, {aug_treatment=}, {depth=}, {width=}, {inj_type=}, {rest=}")
            except:
                dataset, aug_treatment, depth, width, inj_type = "imagenet100", "mlp", 6, 64, "proj-none"
                print(f"Parsing #2 failed, trying defaults: {dataset=}, {aug_treatment=}, {depth=}, {width=}, {inj_type=}")

            args.aug_treatment = aug_treatment
            args.aug_hn_type = AUG_HN_TYPES.mlp
            args.aug_nn_depth = int(depth)
            args.aug_nn_width = int(width)
            args.aug_cond = ["crop", "color", "color_diff", "flip", "blur", "grayscale"]
            args.aug_inj_type = inj_type

            projector = AugProjector(**aug_projector_kwargs)
            projector.load_state_dict(ckpt["projector"])
            projector_type = AUG_COND


        build_model = partial(idist.auto_model, sync_bn=True)
        projector = build_model(projector)

        print(f"loaded projector", projector_type)
        return projector_type, projector

    except Exception as e:
        print(f"Could not load any projector bf of {e}")
        return projector_type, None

def infonce_loss(ft_1, ft_2, device, T: float = 0.2):
    fn_r = F.normalize(ft_1)
    ft_r = F.normalize(ft_2)
    logits = torch.einsum('nc,mc->nm', [fn_r, ft_r]) / T
    N = logits.shape[0]
    labels = torch.arange(N, dtype=torch.long).to(device)
    return F.cross_entropy(logits, labels) * (2 * T)

def self_distill_loss(ft_1, ft_2, device=None):
    return F.cosine_similarity(ft_1, ft_2.detach(), dim=-1).mean().mul(-1)

def cca_loss(
        ft_1, ft_2,
        device=None,
        bt_lambda: float = 0.0051,
):
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    c = ft_1.T @ ft_2

    c = c / len(ft_1)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    return on_diag + bt_lambda * off_diag

def main(local_rank, args):
    cudnn.benchmark = True
    device = idist.device()

    logdir = Path(args.ckpt).parent

    args.origin_run_name = logdir.name

    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"feat_inv-{args.dataset}", args=args,
        job_type="eval_feature_invariance"

    )

    # DATASETS

    datasets = load_datasets_for_cosine_sim(
        dataset=args.dataset,
        pretrain_data=args.pretrain_data,
        datadir=args.datadir,
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

    projector_type, projector = load_projector(args, ckpt)

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
    t_name_to_b_name_to_infonce = defaultdict(lambda : defaultdict(list))
    t_name_to_b_name_to_self_distill = defaultdict(lambda : defaultdict(list))
    t_name_to_b_name_to_cca = defaultdict(lambda : defaultdict(list))

    rrc = RandomResizedCrop(224, scale=(0.2, 1.0))
    identity = transforms_dict["identity"]

    with torch.no_grad():
        for i, (X, _) in enumerate(testloader):
            X_transformed = {
                t_name: t(X) for (t_name, t) in transforms_dict.items()
            }
            X_crops_and_params = [rrc(x) for x in X]
            crop_params = torch.stack([p for (_, p) in X_crops_and_params])
            X_crop = torch.stack([x for (x, _) in X_crops_and_params])
            X_crop = identity(X_crop) #norm
            X_transformed["crop"] = X_crop

            X_norm = X_transformed.pop("identity")

            bs = X_norm.shape[0]

            feats_norm = backbone(X_norm.to(device))

            if projector_type is not None:
                if projector_type == MLP:
                    feats_norm[PROJ_OUT] = F.normalize(projector(feats_norm[BKB_OUT]))
                elif projector_type == AUG_COND:
                    fake_crop_params = torch.cat([torch.zeros(bs, 2), torch.ones(bs, 2)], dim=1)
                    aug_desc = dict()

                    for (t_name, t) in transforms_dict.items():
                        assert t_name != "crop"
                        aug_desc.update(
                            extract_aug_descriptors(
                                t,
                                fake_crop_params
                            )
                        )

                    computed_crop_params = extract_aug_descriptors([], crop_params)["crop"]

                    aug_desc["flip"] = torch.ones_like(aug_desc["flip"])

                    aug_keys = sorted(aug_desc.keys())
                    t_to_aug_descriptors = dict()

                    for t_name in transforms_dict.keys():

                        augs_to_search_in = ["crop", t_name] if t_name != "color" else ["crop", t_name, "color_diff"]
                        t_to_aug_descriptors[t_name] = torch.cat(
                            [
                                (aug_desc[k] if k in augs_to_search_in else torch.zeros_like(aug_desc[k]))
                                for k in aug_keys
                            ],
                            dim=1
                        ).to(device)

                        augs_mixed_to_search_in = augs_to_search_in
                        if t_name in ["flip", "grayscale"]:
                            augs_mixed_to_search_in = ["crop"]

                        t_to_aug_descriptors[f"{t_name}_mixed"] = torch.cat(
                            [
                                (aug_desc[k] if k in augs_mixed_to_search_in else torch.zeros_like(aug_desc[k]))
                                for k in aug_keys
                            ],
                            dim=1
                        ).to(device)

                    t_to_aug_descriptors["crop"] = t_to_aug_descriptors["crop_mixed"]= torch.cat(
                            [
                                (computed_crop_params if k =="crop" else torch.zeros_like(aug_desc[k]))
                                for k in aug_keys
                            ],
                            dim=1
                        ).to(device)

                    feats_norm[PROJ_OUT] = F.normalize(
                        projector(feats_norm[BKB_OUT], t_to_aug_descriptors["identity"])
                    )
                    feats_norm[PROJ_OUT_MIXED] = F.normalize(
                        projector(feats_norm[BKB_OUT], t_to_aug_descriptors["identity"])
                    )
                else:
                    raise NotImplementedError(projector_type)

            for t_name, X_t in X_transformed.items():
                feats_t = backbone(X_t.to(device))

                if projector_type == MLP:
                    feats_t[PROJ_OUT] = F.normalize(projector(feats_t[BKB_OUT]))
                elif projector_type == AUG_COND:
                    feats_t[PROJ_OUT] = F.normalize(
                        projector(feats_t[BKB_OUT], t_to_aug_descriptors[t_name])
                    )
                    feats_t[PROJ_OUT_MIXED] = F.normalize(
                        projector(feats_t[BKB_OUT], torch.flip(t_to_aug_descriptors[f"{t_name}_mixed"], [0]))
                    )


                assert feats_norm.keys() == feats_t.keys()

                for block_name, fn in feats_norm.items():
                    ft = feats_t[block_name]
                    fn_r = fn.reshape(bs, -1)
                    ft_r = ft.reshape(bs, -1)
                    positive_sim = cosine_similarity(fn_r, ft_r).mean().item()

                    if block_name in [PROJ_OUT, PROJ_OUT_MIXED]:
                        proj_sims[f"{block_name}/{t_name}"].extend(cosine_similarity(fn_r, ft_r).detach().cpu().numpy().reshape(-1))

                    negative_sim = cosine_similarity(
                        fn_r,
                        torch.flip(ft_r, [0])
                    ).mean().item()

                    t_name_to_b_name_to_positive_sims[t_name][block_name].append(positive_sim)
                    t_name_to_b_name_to_negative_sims[t_name][block_name].append(negative_sim)
                    t_name_to_b_name_to_diff_sims[t_name][block_name].append(positive_sim - negative_sim)

                    infonce = infonce_loss(fn_r, ft_r, device)
                    t_name_to_b_name_to_infonce[t_name][block_name].append(infonce.item())
                    self_distill = self_distill_loss(fn_r, ft_r)
                    t_name_to_b_name_to_self_distill[t_name][block_name].append(self_distill.item())

                    try:
                        cca = cca_loss(fn_r, ft_r)
                        t_name_to_b_name_to_cca[t_name][block_name].append(cca.item())
                    except:
                        pass

                    if (i+1) % args.print_freq == 0:
                        logger.log_msg(
                            f'{i + 1:3d} | {block_name} | {t_name} | pos: {np.mean(t_name_to_b_name_to_positive_sims[t_name][block_name]):.4f} | neg: {np.mean(t_name_to_b_name_to_negative_sims[t_name][block_name]):.4f})'
                        )

        metrics = dict()
        for (sim_kind, sim_dict) in [
            ("positive", t_name_to_b_name_to_positive_sims),
            ("negative", t_name_to_b_name_to_negative_sims),
            ("diff", t_name_to_b_name_to_diff_sims),
            ("infonce", t_name_to_b_name_to_infonce),
            ("self-distill", t_name_to_b_name_to_self_distill),
            ("cca", t_name_to_b_name_to_cca),
        ]:
            for t_name, b_name_to_sim in sim_dict.items():
                for block_name, sims in b_name_to_sim.items():
                    mean_sim = np.mean(sims)
                    std_sim = np.std(sims)
                    logger.log_msg(f'{sim_kind} {args.dataset} invariance of {block_name} to {t_name}: {mean_sim:.4f}±{std_sim:.4f}')
                    metrics[f"test_feature_invariance/{args.dataset}/{block_name}/{t_name}/{sim_kind}"] = mean_sim

                    if block_name in [PROJ_OUT, PROJ_OUT_MIXED]:
                        metrics[f"test_feature_invariance/{args.dataset}/{block_name}/{t_name}/positive_sims"] = np.array(proj_sims[f"{block_name}/{t_name}"])

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
    parser.add_argument('--print-freq', type=int, default=100)
    parser.add_argument('--distributed', action='store_true')
    args = parser.parse_args()
    args.backend = 'nccl' if args.distributed else None
    args.num_backbone_features = 512 if args.model.endswith('resnet18') else 2048
    with idist.Parallel(args.backend) as parallel:
        parallel.run(main, args)

