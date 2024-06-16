import torch
import random
import json

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from models.vlmo.dist_utils import all_gather
from models.vlmo.objectives import compute_irtr_recall, compute_irtr_recall_with_rerank
from models.vlmo.my_metrics import Accuracy, VQAScore, Scalar
# from pytorch_lightning.utilities.distributed import rank_zero_info
import os

def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.config["loss_names"].items():
            if v < 1:
                continue
            if k == "vqa":
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2":
                if split == "train":
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"dev_{k}_loss", Scalar())
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    setattr(pl_module, f"test_{k}_loss", Scalar())
            elif k == "irtr":
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())

            elif k == "itm":
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itc":
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())

                setattr(pl_module, f"{split}_{k}_vl_i2t_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_vl_t2i_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_vl_logit_scale", Scalar())
            else:
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    if pl_module.config["get_recall_metric"] and not pl_module.training:
        (val_ir_r1, val_ir_r5, val_ir_r10, val_tr_r1, val_tr_r5, val_tr_r10) = compute_irtr_recall(pl_module, split="val")
        val_avg = (val_ir_r1.item() + val_ir_r5.item() + val_ir_r10.item() + val_tr_r1.item() + val_tr_r5.item() + val_tr_r10.item()) / 6.0
        pl_module.logger.experiment.add_scalar(
            "recalls/val_avg", val_avg, pl_module.global_step
        )

        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module, split="test")
        test_avg = (ir_r1.item() + ir_r5.item() + ir_r10.item() + tr_r1.item() + tr_r5.item() + tr_r10.item()) / 6.0
        pl_module.logger.experiment.add_scalar(
            "recalls/test_avg", test_avg, pl_module.global_step
        )

        print("val_avg:{}, test_avg:{}".format(val_avg, test_avg))
        print("test ir_r1:{}, ir_r5:{}, ir_r10:{}, tr_r1:{}, tr_r5:{}, tr_r10:{}".format(ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10))
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r1", ir_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r5", ir_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/ir_r10", ir_r10, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r1", tr_r1, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r5", tr_r5, pl_module.global_step
        )
        pl_module.logger.experiment.add_scalar(
            "recalls/tr_r10", tr_r10, pl_module.global_step
        )
        the_metric += val_avg


    for loss_name, v in pl_module.config["loss_names"].items():
        if v < 1:
            continue

        value = 0

        if loss_name == "vqa":
            value = getattr(pl_module, f"{phase}_{loss_name}_score").compute()
            pl_module.log(f"{loss_name}/{phase}/score_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_score").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "nlvr2":
            if phase == "train":
                value = getattr(pl_module, f"train_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/train/accuracy_epoch", value)
                getattr(pl_module, f"train_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/train/loss_epoch",
                    getattr(pl_module, f"train_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"train_{loss_name}_loss").reset()
            else:
                value_dev = getattr(pl_module, f"dev_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/dev/accuracy_epoch", value_dev)
                getattr(pl_module, f"dev_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/dev/loss_epoch",
                    getattr(pl_module, f"dev_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"dev_{loss_name}_loss").reset()

                value_test = getattr(pl_module, f"test_{loss_name}_accuracy").compute()
                pl_module.log(f"{loss_name}/test/accuracy_epoch", value_test)
                getattr(pl_module, f"test_{loss_name}_accuracy").reset()
                pl_module.log(
                    f"{loss_name}/test/loss_epoch",
                    getattr(pl_module, f"test_{loss_name}_loss").compute(),
                )
                getattr(pl_module, f"test_{loss_name}_loss").reset()
                value = value_dev
        elif loss_name == "irtr":
            value_i2t = getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/i2t_accuracy_epoch", value_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").reset()
            
            value_t2i = getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/t2i_accuracy_epoch", value_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").reset()

            value = value_i2t + value_t2i
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itm":
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
        elif loss_name == "itc":
            value_i2t = getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/i2t_accuracy_epoch", value_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_i2t_accuracy").reset()
            
            value_t2i = getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/t2i_accuracy_epoch", value_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_t2i_accuracy").reset()
            
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

            value_vl_i2t = getattr(pl_module, f"{phase}_{loss_name}_vl_i2t_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/vl_i2t_accuracy_epoch", value_vl_i2t)
            getattr(pl_module, f"{phase}_{loss_name}_vl_i2t_accuracy").reset()
            
            value_vl_t2i = getattr(pl_module, f"{phase}_{loss_name}_vl_t2i_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/vl_t2i_accuracy_epoch", value_vl_t2i)
            getattr(pl_module, f"{phase}_{loss_name}_vl_t2i_accuracy").reset()

            value = value_i2t + value_t2i
        else:
            value = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()
            pl_module.log(
                f"{loss_name}/{phase}/loss_epoch",
                getattr(pl_module, f"{phase}_{loss_name}_loss").compute(),
            )
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.config["loss_names"].items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.config["learning_rate"]
    wd = pl_module.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.config["lr_mult"]
    end_lr = pl_module.config["end_lr"]
    decay_power = pl_module.config["decay_power"]
    optim_type = pl_module.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps==-1:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.config["warmup_steps"]
    if isinstance(pl_module.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)
    print("Warmup_steps:{} \t Max_steps:{}".format(warmup_steps, max_steps))

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


# distributed training functions

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args['rank'] = int(os.environ["RANK"])
        args['world_size'] = int(os.environ['WORLD_SIZE'])
        args['local_rank'] = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args['cluster'] = False
        return

    args['cluster']  = True

    torch.cuda.set_device(args['local_rank'])
    args['dist_backend'] = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args['rank'], args['dist_url'], args['local_rank']), flush=True)

    torch.distributed.init_process_group(backend=args['dist_backend'], init_method=args['dist_url'],
                                         world_size=args['world_size'], rank=args['rank'])
    torch.distributed.barrier()
    setup_for_distributed(args['rank'] == 0)

