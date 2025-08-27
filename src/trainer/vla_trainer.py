import os
import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from typing import List

class QwenVLATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(QwenVLATrainer, self).__init__(*args, **kwargs)

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model  # 可能是 PeftModel 外壳（LoRA）

        if self.optimizer is None:
            # names 用于做 decay/no-decay 切分
            decay_names = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_names = [n for n in decay_names if "bias" not in n]

            # ====== 收集不同模块的“名字清单” ======
            def names_if(pred):
                return [n for n, p in opt_model.named_parameters() if pred(n, p)]

            # 视觉、merger（你原来的）
            visual_names = names_if(lambda n, p: ("visual" in n and "merger" not in n))
            merger_names = names_if(lambda n, p: ("merger" in n))

            # ====== 新增：action 相关 ======
            action_head_names = names_if(lambda n, p: ("action_head" in n))
            action_token_names = names_if(lambda n, p: ("action_token_embeds" in n))

            # 只训练 requires_grad=True 的参数
            def params_from(names: List[str], decay: bool):
                name_set = set(names)
                if decay:
                    return [p for n, p in opt_model.named_parameters()
                            if (n in name_set) and (n in decay_names) and p.requires_grad]
                else:
                    return [p for n, p in opt_model.named_parameters()
                            if (n in name_set) and (n not in decay_names) and p.requires_grad]

            # 汇总“特殊学习率”的名字，便于从默认组里排除
            special_lr_names = set(visual_names + merger_names + action_head_names + action_token_names)

            # ====== 先放入“默认学习率”的两组（排除所有特殊学习率参数） ======
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if (n in decay_names) and (n not in special_lr_names) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in opt_model.named_parameters()
                               if (n not in decay_names) and (n not in special_lr_names) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]

            # ====== 视觉独立 LR（保持你原有逻辑） ======
            if self.args.vision_lr is not None and len(visual_names) > 0:
                optimizer_grouped_parameters.extend([
                    {
                        "params": params_from(visual_names, decay=True),
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_lr,
                    },
                    {
                        "params": params_from(visual_names, decay=False),
                        "weight_decay": 0.0,
                        "lr": self.args.vision_lr,
                    },
                ])

            # ====== merger 独立 LR（保持你原有逻辑） ======
            if self.args.merger_lr is not None and len(merger_names) > 0:
                optimizer_grouped_parameters.extend([
                    {
                        "params": params_from(merger_names, decay=True),
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.merger_lr,
                    },
                    {
                        "params": params_from(merger_names, decay=False),
                        "weight_decay": 0.0,
                        "lr": self.args.merger_lr,
                    },
                ])

            # ====== 新增：action 独立 LR ======
            # 1) 统一学习率（action_lr）
            action_lr = self.args.action_lr
            # 2) 细分：优先使用 head/token 的单独 lr
            action_head_lr = self.args.action_head_lr or action_lr
            action_token_lr = self.args.action_token_lr or action_lr
            action_wd = self.args.action_weight_decay if (self.args.action_weight_decay is not None) else self.args.weight_decay

            if (action_head_lr is not None) and len(action_head_names) > 0:
                optimizer_grouped_parameters.extend([
                    {
                        "params": params_from(action_head_names, decay=True),
                        "weight_decay": action_wd,
                        "lr": action_head_lr,
                    },
                    {
                        "params": params_from(action_head_names, decay=False),
                        "weight_decay": 0.0,
                        "lr": action_head_lr,
                    },
                ])

            if (action_token_lr is not None) and len(action_token_names) > 0:
                optimizer_grouped_parameters.extend([
                    {
                        "params": params_from(action_token_names, decay=True),
                        "weight_decay": action_wd,
                        "lr": action_token_lr,
                    },
                    {
                        "params": params_from(action_token_names, decay=False),
                        "weight_decay": 0.0,
                        "lr": action_token_lr,
                    },
                ])

            # ====== 构建优化器 ======
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # 8-bit Adam 处理 Embedding（保留你原逻辑）
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

            # （可选）打印一下各组规模，便于 sanity-check
            total = sum(p.numel() for g in self.optimizer.param_groups for p in g["params"])
            trainable = sum(p.numel() for n,p in opt_model.named_parameters() if p.requires_grad)
            logger.info(f"[VLA OPT] param_groups={len(self.optimizer.param_groups)}  trainable={trainable:,}  in_groups={total:,}")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super(QwenVLATrainer, self)._save_checkpoint(model, trial)

    # def training_step(self, model, inputs):
    #     for name, param in model.named_parameters():
    #         if 'visual' in name and param.requires_grad:
    #             print(f"Training parameter {name}")
    # 
    #     return super().training_step(model, inputs)