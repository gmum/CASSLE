from typing import Optional

import torch
from torch import nn as nn

from models import load_mlp

AUG_DESC_SIZE_CONFIG = {
    "crop": 4,
    "color": 4,
    "flip": 1,
    "blur": 1,
    # "rot": 4,
    # "sol": 1,
    "grayscale": 1,
    "color_diff": 3
}


class AUG_STRATEGY:
    raw = "raw"
    mlp = "mlp" # TODO -> mlp_proj_cat
    hn = "hn"


class AUG_HN_TYPES:
    mlp = "mlp"
    mlp_bn = "mlp-bn"


class AUG_INJECTION_TYPES:
    proj_cat = "proj-cat" # concatenate raw/mlp_outputs before projector
    proj_mul = "proj-mul" # add proj input to mlp output
    proj_add = "proj-add" # multiply proj input by mlp output
    proj_none = "proj-none" # don't inject anything to projector
    img_cat = "img-cat" # concatenate raw/mlp_outputs to image channels

class AUG_CNT_LOSS_TYPES:
    absolute = "abs" # contrast f_n' with theta_n'
    relative = "relative" # contrast (f_n', f_n'') with (theta_n' - theta_n'')

class AUG_DESC_TYPES:
    absolute = "abs"
    relative = "rel"


class AugProjector(nn.Module):
    def __init__(
            self,
            args, proj_out_dim: int, proj_depth: int = 2,
            proj_hidden_dim: Optional[int] = None,
            projector_last_bn: bool = False,
            projector_last_bn_affine: bool = True,
    ):
        super().__init__()
        self.num_backbone_features = args.num_backbone_features
        self.aug_treatment = args.aug_treatment
        self.aug_hn_type = args.aug_hn_type
        self.aug_nn_depth = args.aug_nn_depth
        self.aug_nn_width = args.aug_nn_width
        self.aug_cond = args.aug_cond or []
        self.aug_subset_sizes = {k: v for (k, v) in AUG_DESC_SIZE_CONFIG.items() if k in self.aug_cond}
        self.aug_inj_type = args.aug_inj_type
        self.projector_last_bn = projector_last_bn

        print("Projector aug strategy:", self.aug_treatment)
        print("Conditioning projector on augmentations:", self.aug_subset_sizes)

        if self.aug_treatment == AUG_STRATEGY.raw or self.aug_inj_type==AUG_INJECTION_TYPES.proj_none:
            self.num_aug_features = sum(self.aug_subset_sizes.values())

            self.aug_processor = nn.Identity()

        elif self.aug_treatment == AUG_STRATEGY.mlp:
            self.num_aug_features = self.aug_nn_width

            self.aug_processor_out = (
                self.aug_nn_width
                if self.aug_inj_type in [AUG_INJECTION_TYPES.proj_cat]
                else args.num_backbone_features
            )

            args.aug_processor_out = self.aug_processor_out

            self.aug_processor = load_mlp(
                n_in=sum(self.aug_subset_sizes.values()),
                n_hidden=self.aug_nn_width,
                n_out=self.aug_processor_out,
                num_layers=self.aug_nn_depth
            )
            print(self.aug_processor)


        elif self.aug_treatment == AUG_STRATEGY.hn:
            num_weights_to_generate = 0

            layer_in_size = args.num_backbone_features
            layer_out_size = args.num_backbone_features

            self.layers_config = []
            for i in range(proj_depth):
                if i == (proj_depth - 1):
                    layer_out_size = proj_out_dim

                print(f"HN for proj layer #{i}")
                print(f"{layer_in_size=}")
                print(f"{layer_out_size=}")
                print(f"weight_params: {layer_in_size * layer_out_size}")
                print(f"bias_params: {layer_out_size}")
                print("-------")

                layer_cfg = dict(
                    n_in=layer_in_size,
                    n_out=layer_out_size,
                    weight_range=(
                        num_weights_to_generate,
                        num_weights_to_generate + layer_in_size * layer_out_size
                    ),
                    bias_range=(
                        num_weights_to_generate + layer_in_size * layer_out_size,
                        num_weights_to_generate + (layer_in_size * layer_out_size) + layer_out_size
                    )
                )
                self.layers_config.append(layer_cfg)

                num_weights_to_generate += layer_in_size * layer_out_size  # weights
                num_weights_to_generate += layer_out_size  # biases

                layer_in_size = layer_out_size

            self.projector_hn = load_mlp(
                n_in=sum(self.aug_subset_sizes.values()),
                n_hidden=self.aug_nn_width,
                n_out=num_weights_to_generate,
                num_layers=self.aug_nn_depth
            )
            print("HN layer config")
            from pprint import pprint
            pprint(self.layers_config)

            if self.aug_hn_type == AUG_HN_TYPES.mlp_bn:
                self.projector_bns = nn.ModuleList([
                    nn.BatchNorm1d(l_cfg["n_out"])
                    for l_cfg in self.layers_config[:-1]
                ])
                print("Projector Batchnorms")
                print(self.projector_bns)

        if self.aug_treatment in [AUG_STRATEGY.raw, AUG_STRATEGY.mlp]:

            projector_in = (
                args.num_backbone_features + self.num_aug_features
                if self.aug_inj_type == AUG_INJECTION_TYPES.proj_cat
                else args.num_backbone_features
            )
            self.projector = load_mlp(
                projector_in,
                proj_hidden_dim or args.num_backbone_features,
                proj_out_dim,
                num_layers=proj_depth,
                last_bn=projector_last_bn,
                last_bn_affine=projector_last_bn_affine,
            )
            print(self.projector)


    def forward(self, x: torch.Tensor, aug_desc: torch.Tensor):

        if self.aug_treatment in [AUG_STRATEGY.mlp, AUG_STRATEGY.raw]:
            aug_desc = self.aug_processor(aug_desc)

            # print(f"pre {x.shape=}, {aug_desc.shape=}")
            if self.aug_inj_type == AUG_INJECTION_TYPES.proj_cat:
                x = torch.cat([x, aug_desc], dim=1)

            elif self.aug_inj_type == AUG_INJECTION_TYPES.proj_add:
                assert aug_desc.shape == x.shape, (x.shape, aug_desc.shape)
                x = x + aug_desc

            elif self.aug_inj_type == AUG_INJECTION_TYPES.proj_mul:
                assert aug_desc.shape == x.shape, (x.shape, aug_desc.shape)
                x = x * aug_desc

            elif self.aug_inj_type == AUG_INJECTION_TYPES.proj_none:
                x = x

            else:
                raise NotImplementedError(self.aug_inj_type)

            # print(f"post {x.shape=}")
            return self.projector(x)


        elif self.aug_treatment == AUG_STRATEGY.hn:
            generated_weights = self.projector_hn(aug_desc)

            b_s, x_s = x.shape

            x_proc = x.reshape(b_s, 1, x_s)
            for l, l_cfg in enumerate(self.layers_config):
                w_r_s, w_r_e = l_cfg["weight_range"]
                b_r_s, b_r_e = l_cfg["bias_range"]

                w_in = l_cfg["n_in"]
                w_out = l_cfg["n_out"]

                w_weights = generated_weights[:, w_r_s:w_r_e].reshape(b_s, w_in, w_out)
                b_weights = generated_weights[:, b_r_s:b_r_e].reshape(b_s, 1, w_out)

                x_proc = torch.bmm(x_proc, w_weights)
                x_proc = x_proc + b_weights
                # TODO - batchnorm?

                if l != (len(self.layers_config) - 1):

                    if self.aug_hn_type == AUG_HN_TYPES.mlp_bn:
                        b_s, _, x_p_s = x_proc.shape
                        x_proc = self.projector_bns[l](x_proc.reshape(b_s, x_p_s)).reshape(b_s, 1, x_p_s)

                    x_proc = torch.relu(x_proc)

            # print(f"{x_proc.shape=}")
            return x_proc.reshape(b_s, -1)


class AugSSPredictor(nn.Module):
    def __init__(self, args, out_dim: int, predictor_depth: int = 2):
        super().__init__()
        assert args.aug_treatment == AUG_STRATEGY.mlp, args.aug_treatment
        assert args.ss_aug_inj_type == AUG_INJECTION_TYPES.proj_cat, args.ss_aug_inj_type

        self.aug_treatment = args.aug_treatment
        self.aug_inj_type = args.ss_aug_inj_type

        self.num_aug_features = args.aug_nn_width
        self.aug_cond = args.aug_cond or []
        self.aug_nn_depth = args.aug_nn_depth
        self.aug_nn_width = args.aug_nn_width
        self.aug_subset_sizes = {k: v for (k, v) in AUG_DESC_SIZE_CONFIG.items() if k in self.aug_cond}


        self.aug_processor_out = args.aug_nn_width
        args.aug_processor_out = self.aug_processor_out

        self.aug_processor = load_mlp(
            n_in=sum(self.aug_subset_sizes.values()),
            n_hidden=self.aug_nn_width,
            n_out=self.aug_processor_out,
            num_layers=self.aug_nn_depth
        )

        projector_in = (
                out_dim + self.num_aug_features
                if self.aug_inj_type == AUG_INJECTION_TYPES.proj_cat
                else out_dim
        )
        self.predictor = load_mlp(
                projector_in,
                out_dim // 4,
                out_dim,
                num_layers=predictor_depth,
                last_bn=False
            )

        print("Predictor aug strategy:", self.aug_treatment)
        print("Conditioning predictor on augmentations:", self.aug_subset_sizes)
        print(self.aug_processor)
        print(self.predictor)
    def forward(self, x: torch.Tensor, aug_desc: torch.Tensor):

        if self.aug_treatment in [AUG_STRATEGY.mlp, AUG_STRATEGY.raw]:

            if self.aug_inj_type == AUG_INJECTION_TYPES.proj_cat:
                aug_desc = self.aug_processor(aug_desc)
                x = torch.concat([x, aug_desc], dim=1)
            elif self.aug_inj_type == AUG_INJECTION_TYPES.proj_none:
                x = x
            else:
                raise NotImplementedError(self.aug_inj_type)
            return self.predictor(x)
