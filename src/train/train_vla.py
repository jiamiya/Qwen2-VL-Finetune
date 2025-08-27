
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ["CUDA_VISIBLE_DEVICES"] = '5' 
# os.environ["TOKENIZERS_PARALLELISM"] = "false"  
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser, Qwen2_5_VLForConditionalGeneration
from src.model.qwen2_5_vla import QwenVLAWrapper
from src.trainer import QwenSFTTrainer, QwenVLATrainer
from src.dataset import make_supervised_vla_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    use_liger = training_args.use_liger
    if "Qwen2.5" in model_args.model_id:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)
    

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual", "lm_head"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    # ==========================================================
    # 1) 构建包装后的模型 —— 直接用 from_pretrained（无需自己实现）
    #    这里把 action 相关的超参从 ModelArguments 里取（若无则用默认）
    # ==========================================================
    wrapper_kwargs = dict(
        num_action_tokens=getattr(model_args, "num_action_tokens", 10),   # 预测步数 N
        action_dim=getattr(model_args, "action_dim", 2),                  # 动作维度 A（如 [speed, curvature]）
        # action_low=getattr(model_args, "action_low", [-10.0, -1.0]),
        # action_high=getattr(model_args, "action_high", [10.0, 1.0]),
        # lm_loss_weight=getattr(model_args, "lm_loss_weight", 0.0),        # 只训 action 时设为 0
        # action_loss_weight=getattr(model_args, "action_loss_weight", 1.0),
    )

    model = QwenVLAWrapper.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa",
        **bnb_model_from_pretrained_args,
        **wrapper_kwargs,
    )

    # 训练时禁用 KV cache
    model.config.use_cache = False

    # 按你原逻辑配置 LLM 与 vision
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    # k-bit 训练前准备
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else
            (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # ==========================================================
    # 2) 应用 LoRA；只训练 LoRA + action_*，其余冻结
    # ==========================================================
    if training_args.lora_enable:
        # 确保 action_head 不会被 LoRA 化（全参训练它）
        lora_namespan_exclude = list(set(list(training_args.lora_namespan_exclude) + ["action_head"]))

        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )

        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        # LoRA 包装后，某些参数（视觉/merger）会被重新冻结；按原逻辑处理
        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True
        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True

        # —— 关键：显式解冻 action_token_embeds 与 action_head（参与训练）
        for name, p in model.named_parameters():
            if ("action_token_embeds" in name) or ("action_head" in name):
                p.requires_grad = True

    # 处理量化/精度下的模块 dtype（沿用原逻辑，并确保 action_head dtype 正确）
    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

        # 把 action_head 调到一致 dtype
        act_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
        for name, module in model.named_modules():
            if name.startswith("action_head") or ".action_head" in name:
                try:
                    module.to(act_dtype)
                except Exception:
                    pass

    # check trainable parameters
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    trainable_info = [(n, p.requires_grad, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad]


    # Optional: print debug info
    # for name, requires_grad, shape in trainable_info:
    #     print(f"{name}: requires_grad={requires_grad}, shape={shape}")
    print(f"#trainable params: {sum(p.numel() for p in trainable_params):,}")

    # ==========================================================
    # 3) 数据与 Trainer（保持原逻辑）
    #    —— 前提： Dataset/Collator 已经产出 action_targets / action_mask
    # ==========================================================
    processor = AutoProcessor.from_pretrained(model_args.model_id)

    data_module = make_supervised_vla_data_module(
        model_id=model_args.model_id,
        processor=processor,
        data_args=data_args
    )

    trainer = QwenVLATrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module
    )
    
    import math
    dl = trainer.get_train_dataloader()
    num_batches = len(dl)
    gas = training_args.gradient_accumulation_steps
    updates_per_epoch = math.ceil(num_batches / gas)

    print(f"[DEBUG] len(dataset)={len(trainer.train_dataset)}")
    print(f"[DEBUG] num_batches_per_epoch={num_batches}")
    print(f"[DEBUG] gradient_accumulation_steps={gas}")
    print(f"[DEBUG] num_update_steps_per_epoch={updates_per_epoch}")
    print(f"[DEBUG] num_train_epochs={training_args.num_train_epochs}  max_steps={training_args.max_steps}")


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    # 训练结束恢复 use_cache（按原逻辑）
    model.config.use_cache = True

    # ==========================================================
    # 4) 保存（LoRA & 非 LoRA 可训练参数分开保存）—— 保持原逻辑
    #     action_* 参数会进入 non_lora_state_dict.bin
    # ==========================================================
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    print(f"Current GPU: {torch.cuda.current_device()}") 
    train()
