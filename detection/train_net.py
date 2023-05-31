#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import sys
sys.path.append(".")
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, Res5ROIHeads
import wandb
from utils import Logger, get_engine_mock
from pathlib import Path
import json

@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def _build_res5_block(self, cfg):
        seq, out_channels = super()._build_res5_block(cfg)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, out_channels)
        seq.add_module("norm", norm)
        return seq, out_channels


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if "coco" in dataset_name:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert "voc" in dataset_name
            return PascalVOCDetectionEvaluator(dataset_name)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):

    out_dir_i = args.opts.index("OUTPUT_DIR")
    out_dir = Path(args.opts[out_dir_i+1])
    logdir = out_dir.parent
    
    assert logdir.exists(), logdir

    # args.opts.extend(["origin_run_name", logdir.name])

    cfg = setup(args)

    cfg_dict = dict(cfg)
    cfg_dict["origin_run_name"] = logdir.name
    
    test_ds_name = '-'.join(cfg.DATASETS.TEST)
    logger = Logger(
        logdir=logdir, resume=True, wandb_suffix=f"det_{test_ds_name}", args=cfg,
        job_type="eval_detection"
    )
    engine_mock = get_engine_mock(ckpt_path=str((logdir / "ckpt-500.pth")))


    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

    
    else:
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        res = trainer.train()
    
    
    with (out_dir / "metrics.json").open("r") as f:
        lines = f.readlines()
        final_metrics = json.loads(lines[-1])
        
    print(final_metrics)
    
    logger.log(engine=engine_mock, global_step=-1,
       **{
            f"test_detection/{test_ds_name}/{k}": v
            for (k,v) in final_metrics.items()
        }
    )

    return res


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)


    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    
    
