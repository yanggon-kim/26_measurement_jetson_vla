"""T016 step 3, GR00T N1.7 rows: `keep1` and `kvonce` as harness-level overrides (VARIANTS, never baseline).

keep1   Qwen3Backbone.forward calls Qwen3VLForConditionalGeneration with the default logits_to_keep=0,
        so lm_head runs over every prefix token and the logits are discarded (only hidden_states[-1]
        is used). The override passes logits_to_keep=1: the head runs on the last token only. The
        hidden states are computed exactly as before -> expected bitwise-equal actions.
kvonce  The DiT's even (cross-attention) blocks apply attn1.to_k / attn1.to_v to the SAME
        encoder_hidden_states (the per-frame VLM features) at every denoising step. The override
        computes each even block's K and V ONCE per frame, before the loop, with that block's own
        to_k / to_v, packs them as cat([K, V], -1), and hands each block its packed tensor as its
        encoder_hidden_states; the block's to_k / to_v are swapped for exact slices of that tensor
        (.contiguous(), so the attention sees the same layout). The rest of diffusers'
        AttnProcessor2_0 and every other op run unchanged. Expected bitwise-equal actions (same GEMM,
        same shape, same inputs -- computed once instead of N times).
The Isaac-GR00T sources are not edited: the loop and the DiT forward are mirrored here, line for line,
for the non-RTC path only (an RTC request raises).
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_keep1(model) -> str:
    bb = model.backbone

    def forward(self, vl_input):
        from transformers.feature_extraction_utils import BatchFeature                 # noqa: PLC0415
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}
        outputs = self.model(**vl_input, output_hidden_states=True, logits_to_keep=1)   # <- keep1
        outputs = outputs.hidden_states[-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(data={"backbone_features": outputs,
                                  "backbone_attention_mask": attention_mask,
                                  "image_mask": image_mask})
    bb.forward = types.MethodType(forward, bb)
    return "backbone.forward: Qwen3VL call with logits_to_keep=1 (lm_head on the last token only)"


class _Half(nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return x[..., self.lo:self.hi].contiguous()


def apply_kvonce(model) -> str:
    ah = model.action_head
    dit = ah.model
    blocks = dit.transformer_blocks
    kvp = {}                                         # even block idx -> (orig to_k, orig to_v)
    for idx, blk in enumerate(blocks):
        if idx % 2 == 0:
            a = blk.attn1
            if getattr(a, "norm_cross", None):
                raise SystemExit("kvonce: attn.norm_cross set -- the K/V input is not the raw features")
            inner = a.to_k.out_features
            kvp[idx] = (a.to_k, a.to_v)
            a.to_k, a.to_v = _Half(0, inner), _Half(inner, 2 * inner)

    def dit_forward_kv(self, hidden_states, enc_list, timestep=None, image_mask=None,
                       backbone_attention_mask=None):
        # mirror of AlternateVLDiT.forward (dit.py:349-415), encoder input per block
        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        image_attention_mask = image_mask & backbone_attention_mask
        non_image_attention_mask = (~image_mask) & backbone_attention_mask
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                hidden_states = block(hidden_states, attention_mask=None, encoder_hidden_states=None,
                                      encoder_attention_mask=None, temb=temb)
            else:
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    m = non_image_attention_mask
                else:
                    m = image_attention_mask
                hidden_states = block(hidden_states, attention_mask=None,
                                      encoder_hidden_states=enc_list[idx], encoder_attention_mask=m,
                                      temb=temb)
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)
    dit.forward_kv = types.MethodType(dit_forward_kv, dit)

    def gawf(self, backbone_features, state_features, embodiment_id, backbone_output, action_input,
             options=None):
        # mirror of get_action_with_features (gr00t_n1d7.py:326-440), non-RTC path
        from transformers.feature_extraction_utils import BatchFeature                 # noqa: PLC0415
        if "action" in action_input:
            raise SystemExit("kvonce mirror covers the non-RTC path only")
        vl_embeds = backbone_features
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(size=(batch_size, self.config.action_horizon, self.action_dim),
                              dtype=vl_embeds.dtype, device=device)
        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)
        enc = vl_embeds.contiguous()
        enc_list = {i: torch.cat([k(enc), v(enc)], dim=-1) for i, (k, v) in kvp.items()}   # <- once
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)
            if not self.config.use_alternate_vl_dit:
                raise SystemExit("kvonce mirror assumes use_alternate_vl_dit")
            model_output = self.model.forward_kv(hidden_states=sa_embs, enc_list=enc_list,
                                                 timestep=timesteps_tensor,
                                                 image_mask=backbone_output.image_mask,
                                                 backbone_attention_mask=backbone_output.backbone_attention_mask)
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity * vel_strength
        return BatchFeature(data={"action_pred": actions, "backbone_features": vl_embeds,
                                  "state_features": state_features})
    ah.get_action_with_features = types.MethodType(gawf, ah)
    return (f"action_head: K/V of {len(kvp)} cross-attention blocks computed once per frame (packed, "
            f"to_k/to_v -> exact slices); get_action_with_features + DiT forward mirrored")
