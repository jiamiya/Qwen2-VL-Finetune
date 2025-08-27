import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
from dataclasses import dataclass
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, List, Tuple, Union

from .waypoint_encoder import WaypointDiffNormCoder, WaypointRangeNormCoder

@dataclass
class VLAOutputWithPast(CausalLMOutputWithPast):
    rope_deltas: Optional[torch.Tensor] = None
    action_hidden_states: Optional[torch.FloatTensor] = None  # [B, N, H]
    # Added
    action_preds: Optional[torch.Tensor] = None            # [B, N, A]
    action_loss: Optional[torch.Tensor] = None              # 标量
    lm_loss: Optional[torch.Tensor] = None       

class QwenVLAWrapper(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config, num_action_tokens: int = 10, action_dim: int = 2, return_action_states: bool = True, use_waypoint_encoder: bool = True):
        super().__init__(config)
        self.num_action_tokens = num_action_tokens
        self.action_dim = action_dim
        
        mlp_hidden = 1024
        self.action_head = nn.Sequential(
            nn.Linear(self.model.embed_tokens.embedding_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, self.action_dim),
        )
        
        self.action_use_tanh = True

        self.lm_loss_weight = 1.0
        self.action_loss_weight = 1.0
        self.action_loss_type = "mse"
        
        self.return_action_states = return_action_states
        if num_action_tokens > 0:
            hidden_size = self.model.embed_tokens.embedding_dim
            self.action_token_embeds = nn.Parameter(
                torch.randn(num_action_tokens, hidden_size) * 0.02
            )
        self._last_action_states = None  # 便于 generate 增量阶段外部取用
        
        self.use_waypoint_encoder = use_waypoint_encoder
        if self.use_waypoint_encoder:
            # self.waypoint_encoder = WaypointDiffNormCoder()
            self.waypoint_encoder = WaypointRangeNormCoder()

    @torch.no_grad()
    def _build_inputs_embeds(
        self,
        input_ids,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        """
        复刻官方前半段：把 token embedding 算出来，并用视觉 encoder 的特征替换 image/video 占位 token。
        之所以放到 wrapper，是因为我们要往后面再拼 action embeddings，所以必须自己算好 inputs_embeds 再传给 super()。
        """
        inputs_embeds = self.model.embed_tokens(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)  # [#img_tokens, H]
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            if n_image_tokens != image_embeds.shape[0]:
                raise ValueError(f"Image features and image tokens do not match: tokens={n_image_tokens}, feats={image_embeds.shape[0]}")
            mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask.to(inputs_embeds.device), image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)  # [#vid_tokens, H]
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            if n_video_tokens != video_embeds.shape[0]:
                raise ValueError(f"Video features and video tokens do not match: tokens={n_video_tokens}, feats={video_embeds.shape[0]}")
            mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask.to(inputs_embeds.device), video_embeds.to(inputs_embeds.device, inputs_embeds.dtype))

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        return inputs_embeds, attention_mask

    def _append_action_tokens(
        self,
        inputs_embeds,
        attention_mask,
        labels,
        input_ids,
        should_append: bool,
    ):
        """
        把 [B, T, H] 的 inputs_embeds 末尾拼上 N 个可学习 action；同步扩展 mask/labels/input_ids。
        注意：为让 RoPE/位置对齐，**必须**把 input_ids 也扩到 T+N（这里用 pad_token_id 占位）。
        """
        if not should_append or self.num_action_tokens <= 0:
            return inputs_embeds, attention_mask, labels, input_ids, 0

        B = inputs_embeds.size(0)
        N = self.num_action_tokens
        H = inputs_embeds.size(-1)

        action_embeds = self.action_token_embeds.unsqueeze(0).expand(B, N, H).to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, action_embeds], dim=1)

        if attention_mask is not None:
            pad_mask = torch.ones((B, N), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        if labels is not None:
            neg = -100 * torch.ones((B, N), dtype=labels.dtype, device=labels.device)
            labels = torch.cat([labels, neg], dim=1)

        if input_ids is not None:
            pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
            tail = torch.full((B, N), pad_id, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, tail], dim=1)

        return inputs_embeds, attention_mask, labels, input_ids, N

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        action_targets: Optional[torch.FloatTensor] = None,  # [B, N, A]
        action_mask: Optional[torch.BoolTensor] = None,      # [B, N]，True=有监督
        compute_lm_loss: bool = False,  
        **kwargs,
    ):
        # 0) 取出我们用的，丢弃 Trainer 注入的
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("return_loss", None)
        kwargs.pop("ignore_keys", None)
        kwargs.pop("output_orig", None)
        kwargs.pop("inputs", None)

        # —— 1) 如果没给 inputs_embeds，就先把 embed + 图像/视频替换算好
        if inputs_embeds is None:
            # embed + 用 visual 特征替换 <image>/<video> 占位（略）
            inputs_embeds, attention_mask = self._build_inputs_embeds(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

        # —— 2) 只在 prefill 阶段（KV 空）追加 action tokens；增量阶段不追加
        is_prefill = (past_key_values is None) or (getattr(past_key_values, "get_seq_length", lambda: 0)() == 0)
        inputs_embeds, attention_mask, labels, input_ids, appended = self._append_action_tokens(
            inputs_embeds, attention_mask, labels, input_ids, should_append=is_prefill
        )

        # —— 3) 调用父类 forward（它会基于 input_ids 的长度计算 position_ids/RoPE；我们已把长度同步到 T+N 只传白名单字段
        allowed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels if compute_lm_loss else None,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
            "pixel_values":None,               # 已经手工处理过视觉特征
            "pixel_values_videos":None,
            "image_grid_thw":image_grid_thw,
            "video_grid_thw":video_grid_thw,
            "rope_deltas": rope_deltas,
            "cache_position": cache_position,
            "second_per_grid_ts": second_per_grid_ts,
        }
        safe_kwargs = {k: v for k, v in allowed.items() if v is not None}

        out = super().forward(**safe_kwargs)

        # —— 4) 提取 action hidden states（只在拼接过的这次可直接从最后 N 个取）
        action_states = None
        if appended and self.num_action_tokens > 0:
            # out.logits/hidden_states 的第一维是 batch，第二维是序列
            action_states = out.logits.new_empty(0)  # 占位防 torchscript 报错
            # hidden 是模型最后一层输出：out.logits = lm_head(hidden)
            hidden = out.logits.new_empty(0)  # 仅为安全初始化
            # 有些实现 out.hidden_states 可能是每层的 list；为稳妥用 out.logits 逆推不方便
            # 更直接：父类 outputs[0] 即最后层 hidden；但这里 return_dict=True 时可用 out.logits 和 lm_head 关系：
            # 我们直接复用父类内部的最后隐层：Qwen 的 `model.forward` 返回的第一个张量就是 last_hidden_state。
            # 为保证，我们再算一次：通过 self.lm_head 的输入是 last_hidden_state。
            # 这里可以从父类未公开的字段拿不太稳，所以更保险：重新前向一次拿 outputs[0] 代价大。
            # 折中做法：让父类也返回 hidden_states=True。
            if not output_hidden_states:
                # 需要的话，重新跑一遍很小代价（通常 prefill 只有一次）
                out2 = super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=None,
                    use_cache=use_cache,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    pixel_values=None,
                    pixel_values_videos=None,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                    second_per_grid_ts=second_per_grid_ts,
                )
                last_hidden = out2[0]  # [B, T+N, H]
            else:
                # return_dict=True 且 output_hidden_states=True：out.hidden_states 是每层 list
                last_hidden = out.hidden_states[-1]  # [B, T+N, H]

            action_states = last_hidden[:, -self.num_action_tokens:, :]  # [B, N, H]
            self._last_action_states = action_states.detach()
            
            # —— 解码到数值动作 —— #
            action_preds = None
            action_loss  = None

            if self.num_action_tokens > 0 and action_states is not None:
                # [B, N, H] -> [B, N, A]
                action_preds = self.action_head(action_states)

                if self.action_use_tanh:
                    action_preds = torch.tanh(action_preds)

                # —— 动作监督（可选） —— #
                if action_targets is not None:
                    # 对齐 dtype/device
                    action_targets = action_targets.to(action_preds.device, action_preds.dtype)

                    if action_mask is None:
                        # 默认所有位置都有监督
                        action_mask = torch.ones(action_targets.shape[:2], dtype=torch.bool, device=action_preds.device)

                    # [B, N, 1] 用于按时间步屏蔽
                    mask_t = action_mask.unsqueeze(-1)

                    if self.action_loss_type == "mse":
                        if not self.use_waypoint_encoder:
                            per_elem = (action_preds - action_targets) ** 2
                        else:
                            action_preds_normed = action_preds # we expect the network output is the action value between [-1, 1]
                            action_targets_normed = self.waypoint_encoder.encode(action_targets)
                            # print('action normed: ', action_preds_normed)
                            # print('action_targets_normed: ', action_targets_normed)
                            per_elem = (action_preds_normed - action_targets_normed) ** 2 # L2 loss on normalized action
                            action_preds = self.waypoint_encoder.decode(action_preds_normed) # denomarlize the action for real action
                    elif self.action_loss_type == "l1":
                        if not self.use_waypoint_encoder:
                            per_elem = (action_preds - action_targets).abs()
                        else:
                            action_preds_normed = action_preds # we expect the network output is the action value between [-1, 1]
                            action_targets_normed = self.waypoint_encoder.encode(action_targets)
                            per_elem = (action_preds_normed - action_targets_normed).abs() # L1 loss on normalized action
                            action_preds = self.waypoint_encoder.decode(action_preds_normed) # denomarlize the action for real action
                    else:  # huber
                        delta = 1.0
                        if not self.use_waypoint_encoder:
                            diff = (action_preds - action_targets).abs()
                            per_elem = torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
                        else:
                            action_preds_normed = action_preds # we expect the network output is the action value between [-1, 1]
                            action_targets_normed = self.waypoint_encoder.encode(action_targets)
                            diff = (action_preds_normed - action_targets_normed).abs()
                            per_elem = torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))
                            action_preds = self.waypoint_encoder.decode(action_preds_normed) # denomarlize the action for real action
                            
                            

                    # 只对 mask=True 的时间步求均值；若不同维度重要性不同可加权
                    per_elem = per_elem * mask_t  # [B,N,A]
                    # print('mask_t ', mask_t)
                    # print('perelem: ', per_elem)
                    denom = mask_t.sum() * action_preds.shape[-1] + 1e-8
                    # print('denom',denom)
                    action_loss = per_elem.sum() / denom
                
                else: # action_target is None, for inference only
                    action_preds = self.waypoint_encoder.decode(action_preds)
                    action_loss = None


        lm_loss = out.loss if compute_lm_loss else None

        if (lm_loss is not None) and (action_loss is not None):
            total_loss = self.lm_loss_weight * lm_loss + self.action_loss_weight * action_loss
        elif action_loss is not None:
            total_loss = self.action_loss_weight * action_loss
        else:
            total_loss = lm_loss  # 可能是 None

        return VLAOutputWithPast(
            loss=total_loss,
            logits=out.logits if compute_lm_loss else None,
            past_key_values=out.past_key_values,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
            rope_deltas=getattr(out, "rope_deltas", None),
            action_hidden_states=action_states if self.return_action_states else None,
            # 新增字段
            action_preds=action_preds,            # [B, N, A]
            action_loss=action_loss,              # 标量
            lm_loss=lm_loss,                      # 标量或 None
        )

    def get_action_states(self):
        """在 generate 的增量阶段拿不到当步的 N 个 hidden 时，可从这里取到 prefill 存下来的那份。"""
        return self._last_action_states
