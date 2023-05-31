import logging
import os
from pathlib import Path
from typing import List, Optional
import matplotlib
from matplotlib import pyplot as plt

import ignite.distributed as idist
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


def maybe_setup_wandb(logdir, args=None, run_name_suffix=None, **init_kwargs):

    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    if wandb_entity is None or wandb_project is None:
        print(f"{wandb_entity=}", f"{wandb_project=}")
        print("Not initializing WANDB")
        return

    origin_run_name = Path(logdir).name

    api = wandb.Api()

    name_runs = list(api.runs(f'{wandb_entity}/{wandb_project}', filters={'display_name': origin_run_name}))
    group_runs = list(api.runs(f'{wandb_entity}/{wandb_project}', filters={'group': origin_run_name}))

    print(f'Retrieved {len(name_runs)} for run_name: {origin_run_name}')

    assert len(name_runs) <= 1, f'retrieved_runs: {len(name_runs)}'

    new_run_name = origin_run_name if len(name_runs) == 0 else f"{origin_run_name}_{len(group_runs)}"

    if run_name_suffix is not None:
        new_run_name = f"{new_run_name}_{run_name_suffix}"

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config=args,
        name=new_run_name,
        dir=logdir,
        resume="never",
        group=origin_run_name,
        **init_kwargs
    )

    print("WANDB run", wandb.run.id, new_run_name, origin_run_name)

def get_engine_mock(ckpt_path: str):
    print("Mocking engine from", ckpt_path)
    try:
        epoch_no = int(
            Path(ckpt_path).name.replace(".pth", "").replace("ckpt-", "")
        )
    except Exception as e:
        print("Epoch inference error", e)
        epoch_no = -1

    print(f"Engine mock inferred {epoch_no=}")
    class engine:
        class state:
            epoch = epoch_no
            iteration = epoch_no

    return engine

def get_first_free_port(start_port: int=2222, n_ports_to_check: int =100) -> int:
    """
    A shitfix for two distributed trainings on one device, see:
    https://github.com/pytorch/ignite/issues/2312
    Solution based on:
    https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python
    """
    def is_port_in_use(port: int) -> bool:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    for port in range(start_port, start_port + n_ports_to_check):
        print(f"checking {port=}")
        if not is_port_in_use(port):
            print(f"{port=} seems to be free")
            return port

    raise ConnectionError(f"Free port not found with {start_port=} and {n_ports_to_check=}")

class Logger(object):

    def __init__(self, logdir, resume=None, args=None, wandb_suffix=None, **wandb_kwargs):
        assert logdir is not None

        self.logdir = logdir
        self.rank = idist.get_rank()

        handlers = [logging.StreamHandler(os.sys.stdout)]
        if logdir is not None and self.rank == 0:
            if resume is None:
                os.makedirs(logdir)
            maybe_setup_wandb(logdir=logdir, args=args, run_name_suffix=wandb_suffix, **wandb_kwargs)
            handlers.append(logging.FileHandler(os.path.join(logdir, 'log.txt')))
            self.writer = SummaryWriter(log_dir=logdir)
        else:
            self.writer = None

        logging.basicConfig(format=f"[%(asctime)s ({self.rank})] %(message)s",
                            level=logging.INFO,
                            handlers=handlers)
        logging.info(' '.join(os.sys.argv))

    def log_msg(self, msg):
        if idist.get_rank() > 0:
            return
        logging.info(msg)

    def log(self, engine, global_step, print_msg=True, **kwargs):
        msg = f'[epoch {engine.state.epoch}] [iter {engine.state.iteration}]'

        kwargs["epoch"] = engine.state.epoch
        kwargs["iteration"] = engine.state.iteration
        kwargs["log_step"] = global_step

        wandb_log = dict()
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is float:
                msg += f' [{k} {v:.4f}]'
            elif type(v) in [matplotlib.lines.Line2D, matplotlib.patches.Rectangle] \
                or (type(v) is list and type(v[0]) is matplotlib.lines.Line2D):
                wandb.log({f"plot {k}": v})
                continue
            else:
                msg += f' [{k} {v}]'

            if self.writer is not None:
                try:
                    self.writer.add_scalar(k, v, global_step)
                except:
                    pass
            wandb_log[k] = v

        if wandb.run is not None:
            wandb.log(wandb_log)


        if print_msg:
            logging.info(msg)

    def save(self, engine, override_name: Optional[str]=None, **kwargs):
        if idist.get_rank() > 0:
            return

        state = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.nn.parallel.DistributedDataParallel):
                v = v.module

            if hasattr(v, 'state_dict'):
                state[k] = v.state_dict()

            if type(v) is list and hasattr(v[0], 'state_dict'):
                state[k] = [x.state_dict() for x in v]

            if type(v) is dict and k == 'ss_predictor':
                state[k] = { y: x.state_dict() for y, x in v.items() }

        filename = override_name or  f'ckpt-{engine.state.epoch}.pth'
        torch.save(state, os.path.join(self.logdir, filename))

