import os
from pathlib import Path
import argparse
import time
import datetime
import typing
import random
import argparse

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import numpy as np

from audiocraft.models import MusicGen, LMModel
from audiocraft.models import builders
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
from audiocraft.models.loaders import load_compression_model, _get_state_dict
from datasets import MusicGenDataset
from omegaconf import OmegaConf


def dataset_collate_fn(batch):
    from torch.utils.data.dataloader import default_collate

    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def run_cmd(cmd_list: list):
    import subprocess

    try:
        ret = subprocess.check_output(
            " ".join(cmd_list), shell=True, stderr=subprocess.STDOUT
        )
        print(ret.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print(e.returncode, e.output)
    except Exception as e:
        print(e)


def get_config(path: Path) -> dict:
    import json

    with open(path, mode="r") as rf:
        result = json.load(rf)
    return result


def init_configs(config_path: str) -> tuple:
    config = get_config(Path(config_path))
    training_config: dict = config["training_config"]
    evaluation_config: dict = config.get("evaluation_config", None)
    dataset_config: dict = config["dataset_config"]
    model_config: dict = config["model_config"]
    dataset_config.update(target_sample_rate=model_config["sample_rate"])
    return training_config, evaluation_config, dataset_config, model_config


def init_dataset(dataset_config: dict) -> MusicGenDataset:
    from mp_data import get_numpy_filelist

    np_list, np_addr_list = get_numpy_filelist(
        filelist_path=dataset_config["file_list_path"], blacklist_path=None
    )
    dataset = MusicGenDataset(
        root=dataset_config["root"],
        np_list=np_list,
        np_addr_list=np_addr_list,
        target_duration=dataset_config["target_duration"],
        target_sample_rate=dataset_config["target_sample_rate"],
    )
    return dataset


def init_model(model_config: dict, device="cpu") -> MusicGen:
    model_name = model_config["name"]
    compression_model = load_compression_model(model_name, device=device)
    pkg = _get_state_dict(model_name, filename="state_dict.bin")
    cfg = OmegaConf.create(pkg["xp.cfg"])
    cfg.device = str(device)
    if cfg.device == "cpu":
        cfg.transformer_lm.memory_efficient = False
        cfg.transformer_lm.custom = True
        cfg.dtype = "float32"
    else:
        cfg.dtype = "float16"
    lm = builders.get_lm_model(cfg)
    model = MusicGen(model_name, compression_model=compression_model, lm=lm)
    model.set_generation_params(use_sampling=True, duration=model_config["duration"])
    return model


def load_checkpoint(checkpoint_config: dict, lm: LMModel):
    if not checkpoint_config:
        return


def init_optimizers(model: torch.nn.Module) -> tuple:
    # optimizer
    opt_config = {"lr": 3e-4, "weight_decay": 0.01}
    model_opt = torch.optim.AdamW(model.parameters(), **opt_config)
    model_scheduler = torch.optim.lr_scheduler.LinearLR(
        model_opt, start_factor=1e-7, end_factor=1.0, total_iters=3000
    )
    return model_opt, model_scheduler


def get_cfg_conditions(model: MusicGen, conditions):
    two_step_cfg = model.generation_params["two_step_cfg"]
    lm = model.lm
    if conditions:
        null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
        if two_step_cfg:
            cfg_conditions = (
                lm.condition_provider(lm.condition_provider.tokenize(conditions)),
                lm.condition_provider(lm.condition_provider.tokenize(null_conditions)),
            )
        else:
            conditions = conditions + null_conditions
            tokenized = lm.condition_provider.tokenize(conditions)
            cfg_conditions = lm.condition_provider(tokenized)
    else:
        cfg_conditions = {}
    return cfg_conditions


def train_base(args):
    """single node single gpu"""
    config_path: str = args.config_path
    training_config, evaluation_config, dataset_config, model_config = init_configs(
        config_path
    )
    model_dir = Path(training_config["model_dir"])
    assert model_dir.exists() and model_dir.is_dir(), f"invalid {model_dir=}"
    summary_writer = SummaryWriter(log_dir=Path(training_config["summarywriter_root"]))
    # dataloader
    dataset = init_dataset(dataset_config)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=dataset_collate_fn,
    )
    # device
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    training_seed = 0 if training_config["constant_seed"] else time.time()
    torch.manual_seed(training_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(training_seed)
    # model
    model = init_model(model_config, device=device)
    # load checkpoint
    checkpoint_config: dict = training_config.get("checkpoint_config", None)
    load_checkpoint(checkpoint_config, model.lm)
    # loss
    model_opt, model_scheduler = init_optimizers(model.lm)
    start_epoch_idx = checkpoint_config.get("epoch_idx") if checkpoint_config else 0
    start_step_idx = checkpoint_config.get("step_idx") if checkpoint_config else 0
    max_gen_len = model.generation_params["max_gen_len"]
    for epoch_idx in range(
        start_epoch_idx, start_epoch_idx + training_config["num_epochs"]
    ):
        model.lm.train()
        epoch_time = time.time()
        for step_idx, batch in enumerate(dataloader, start=1):
            B = batch.size(0)
            batch = batch.to(device)
            global_step = start_step_idx
            attributes = model._prepare_attributes(batch)
            cfg_conditions = get_cfg_conditions(model, attributes)
            prompt = torch.zeros(
                (B, model.lm.num_codebooks, 0), dtype=torch.long, device=device
            )
            B, K, start_offset = prompt.size()
            pattern = model.lm.pattern_provider.get_pattern(max_gen_len)
            unknown_token = -1
            gen_codes = torch.full(
                (B, K, max_gen_len), unknown_token, dtype=torch.long, device=device
            )
            # filling the gen_codes with the prompt if needed
            gen_codes[..., :start_offset] = prompt
            output = model.lm.compute_predictions(
                gen_codes, None, condition_tensors=cfg_conditions
            )
            # calc
            print(batch.shape)


# def train_ddp(args):
#     """multi-node multi-gpu"""
#     import torch.distributed as torch_dist
#     from torch.nn.parallel import DistributedDataParallel
#     from torch.utils.data import DistributedSampler

#     def init_distributed_dataset(
#         dataset_config: dict, world_size: int, rank: int, local_rank: int
#     ):
#         from mp_data import get_distributed_shared_filelist

#         np_list, np_addr_list = get_distributed_shared_filelist(
#             world_size,
#             rank,
#             local_rank,
#             filelist_path=dataset_config["file_list_path"],
#             blacklist_path=dataset_config.get("blacklist_path", None),
#         )
#         gain_list = np.array(dataset_config["gain_list"])
#         dataset = EncodecDataset(
#             root=dataset_config["root"],
#             np_list=np_list,
#             np_addr_list=np_addr_list,
#             target_duration=dataset_config["target_duration"],
#             target_sample_rate=dataset_config["target_sample_rate"],
#             gain_list=gain_list,
#         )
#         return dataset

#     # set up env
#     assert torch_dist.is_nccl_available(), f"nccl is not supported"
#     backend = "nccl"
#     device_str = "cuda"
#     torch_dist.init_process_group(backend=backend)
#     rank = torch_dist.get_rank()
#     # device_id = rank % torch.cuda.device_count()
#     local_rank = int(os.environ["LOCAL_RANK"])
#     world_size = torch_dist.get_world_size()
#     device = torch.device(device_str, local_rank)
#     # NOTE: manually set device for broadcast
#     torch.cuda.set_device(local_rank)
#     print(f"{rank=} {local_rank=} {world_size=} {device=}")

#     # config
#     config_path = args.config_path
#     training_config, evaluation_config, dataset_config, model_config = init_configs(
#         config_path
#     )
#     model_dir = Path(training_config["model_dir"])
#     assert model_dir.exists() and model_dir.is_dir(), f"invalid {model_dir=}"
#     if rank == 0:
#         summary_writer = SummaryWriter(
#             log_dir=Path(training_config["summarywriter_root"])
#         )
#     training_seed = time.time()
#     if training_config["constant_seed"]:
#         training_seed = training_config["constant_seed"]
#     torch.manual_seed(training_seed)
#     torch.cuda.manual_seed_all(training_seed)

#     # dataloader
#     dataset = init_distributed_dataset(dataset_config, world_size, rank, local_rank)
#     sampler = DistributedSampler(dataset)
#     num_workers = 16
#     dataloader = DataLoader(
#         dataset,
#         sampler=sampler,
#         batch_size=training_config["batch_size"],
#         collate_fn=dataset_collate_fn,
#         # shuffle=True,
#         drop_last=True,
#         num_workers=num_workers,
#         persistent_workers=True,
#         pin_memory=True,
#     )
#     # model
#     model = init_model(model_config)
#     # discriminator
#     discriminator = init_discriminator(model_config)
#     # load checkpoint
#     checkpoint_config: dict = training_config.get("checkpoint_config", None)
#     load_checkpoint(checkpoint_config, model, discriminator)
#     # loss
#     time_loss, freq_loss, feat_loss, adver_loss, disc_loss = init_losses(model_config)
#     # balancer
#     balancer = init_balancer()
#     # optimizer
#     model_opt, discriminator_opt = init_optimizers(model, discriminator)
#     # ddp and device
#     model.to(device)
#     model = DistributedDataParallel(model)
#     discriminator.to(device)
#     discriminator = DistributedDataParallel(discriminator)
#     time_loss.to(device)
#     freq_loss.to(device)
#     feat_loss.to(device)
#     adver_loss.to(device)
#     disc_loss.to(device)
#     # train
#     start_epoch_idx = checkpoint_config.get("epoch_idx") if checkpoint_config else 0
#     for epoch_idx in range(
#         start_epoch_idx, start_epoch_idx + training_config["num_epochs"]
#     ):
#         # set mode
#         model.train()
#         discriminator.train()
#         # update sampler
#         sampler.set_epoch(epoch_idx)
#         # sync every epoch
#         torch_dist.barrier()
#         #
#         if rank == 0:
#             epoch_time = time.time()
#         for step_idx, input in enumerate(dataloader, start=1):
#             global_step = epoch_idx * len(dataloader) + step_idx
#             # sync batch_bandwidth
#             if rank == 0:
#                 batch_bandwidth_t = torch.tensor(
#                     random.choice(model.module.target_bandwidths)
#                 ).to(device)
#             else:
#                 batch_bandwidth_t = torch.tensor(-1.0).to(device)
#             torch_dist.broadcast(batch_bandwidth_t, 0, async_op=False)
#             batch_bandwidth = batch_bandwidth_t.item()
#             print(f"{rank=} {local_rank=} {batch_bandwidth=}")
#             model.module.set_target_bandwidth(batch_bandwidth)
#             # calc
#             input = input.to(device)
#             print(f"{rank=} {local_rank=} {epoch_idx=} {step_idx=} {input.size()}")
#             output, quantized_loss = model(input)
#             input_logits, input_feat_maps = discriminator(input)
#             output_logits, output_feat_maps = discriminator(output)
#             lt = time_loss(output, input)
#             lf = freq_loss(output, input)
#             lfeat = feat_loss(output_feat_maps, input_feat_maps)
#             lg = adver_loss(output_logits)
#             ld = disc_loss(input_logits, output_logits)
#             loss_dict = {"lt": lt, "lf": lf, "lfeat": lfeat, "lg": lg}
#             update_discriminator = (
#                 step_idx % training_config["steps_update_discriminator"] == 0
#             )
#             # print(f"{rank=} {local_rank=} {epoch_idx=} {step_idx=} {loss_dict=}")
#             # TODO: use only corresponding discriminator for given bandwidth
#             # print(f"{rank=} {device_id=} {batch_bandwidth=} done forwarding")
#             if update_discriminator:
#                 discriminator_opt.zero_grad()
#                 ld.backward()
#                 discriminator_opt.step()
#                 # print(
#                 #     f"{rank=} {device_id=} {batch_bandwidth=} done discriminator opt step"
#                 # )
#             else:
#                 model_opt.zero_grad()
#                 balancer.backward(loss_dict, output)
#                 quantized_loss.backward()
#                 model_opt.step()
#                 # print(f"{rank=} {device_id=} {batch_bandwidth=} done model opt step")
#             # print(f"{rank=} {device_id=} {batch_bandwidth=} done backwarding")
#             # summary
#             if rank == 0:
#                 summary_writer.add_scalar("lt", lt.item(), global_step=global_step)
#                 summary_writer.add_scalar("lf", lf.item(), global_step=global_step)
#                 summary_writer.add_scalar(
#                     "lw", quantized_loss.item(), global_step=global_step
#                 )
#                 summary_writer.add_scalar(
#                     "lfeat", lfeat.item(), global_step=global_step
#                 )
#                 summary_writer.add_scalar("lg", lg.item(), global_step=global_step)
#                 summary_writer.add_scalar("ld", ld.item(), global_step=global_step)
#                 if step_idx % 10 == 0:
#                     loss_dict.update(ld=ld.item(), lw=quantized_loss.item())
#                     log_str = " ".join([f"{k}={v:.4f}" for k, v in loss_dict.items()])
#                     print(
#                         f"{datetime.datetime.now()}: {rank=} {epoch_idx=} {step_idx=} {log_str}"
#                     )
#             print(f"{rank=} {local_rank=} {epoch_idx=} {step_idx=} done")
#         # save and evaluate
#         if rank == 0:
#             print(f"epoch_time={(time.time() - epoch_time):.4f}s")
#             # save models
#             saved_epoch_idx = epoch_idx + 1
#             saved_model_name = f"encodec_model_epoch{saved_epoch_idx}.pth"
#             torch.save(model.module.state_dict(), model_dir.joinpath(saved_model_name))
#             saved_discriminator_name = (
#                 f"encodec_discriminator_epoch{saved_epoch_idx}.pth"
#             )
#             torch.save(
#                 discriminator.module.state_dict(),
#                 model_dir.joinpath(saved_discriminator_name),
#             )
#             print(
#                 f"{datetime.datetime.now()}: done saving model and discriminator at {saved_epoch_idx=}"
#             )

#             # evaluate
#             # if not evaluation_config:
#             #     continue
#             # model.eval()
#             # audio_path_list: list = evaluation_config["audio_path_list"]
#             # for audio_path in audio_path_list:
#             #     print(f"{datetime.datetime.now()}: evaluating {audio_path} ... ")
#             #     # run encodec
#             #     encodec_path = eval_encodec(audio_path, model)
#             #     # run visqol
#             #     eval_visqol(
#             #         evaluation_config["visqol_path"],
#             #         audio_path,
#             #         encodec_path.as_posix(),
#             #         evaluation_config["visqol_arguments"],
#             #     )
#             #     # run sisnr
#             #     sisnr_result = eval_sisnr(audio_path, encodec_path.as_posix())
#             #     print(f"{sisnr_result=}")
#     # clean up
#     torch_dist.destroy_process_group()
#     if rank == 0:
#         print(f"{datetime.datetime.now()}: all done")


# def train_deepspeed():
#     import deepspeed

#     def get_arguments():
#         parser = argparse.ArgumentParser()
#         parser.add_argument(
#             "--local_rank",
#             type=int,
#             default=-1,
#             help="local rank passed from distributed launcher",
#         )
#         parser = deepspeed.add_config_arguments(parser)
#         args = parser.parse_args()
#         return args

#     args = get_arguments()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--method", type=str, default=None, help="function name to be run"
    )
    arg_parser.add_argument(
        "--config_path", type=str, default=None, help="path to configuration.json"
    )
    args = arg_parser.parse_args()
    if args.method == "train_base":
        train_base(args)
    # elif args.method == "train_ddp":
    #     train_ddp(args)
    else:
        raise ValueError(f"unsupported method name {args.method}")
