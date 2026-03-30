import numpy as np
import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers import AutoTokenizer

from pi05_infer import (
    matmul_abT_scale,
    matmul_k8_n_256,
    matmul_k_32_1024_bias,
    matmul_k_1024_2560_qkv_rope,
    matmul_k_2048_1024_gate,
    matmul_k_4096_1024_gate,
    matmul_small_bias_res,
    matmul_small_gate,
    softmax_kernel_prefix_suffix,
    transformer_encoder,
    vision_encoder,
)

def _swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def _posemb_sincos_torch(
    t: torch.Tensor, embedding_dim: int = 1024, min_period: float = 4e-3, max_period: float = 4.0
) -> torch.Tensor:
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, dtype=torch.float32, device=t.device)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = t[:, None] * (1.0 / period)[None, :] * (2.0 * np.pi)
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)

@triton.jit
def adarms_norm_kernel_rowwise(
    x_ptr,
    adarms_mod_ptr,
    normed_x_ptr,
    gate_ptr,
    seq_len: tl.constexpr,
    features: tl.constexpr,
    adarms_mod_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for i in range(pid, seq_len, psize):
        row_x_offset = i * features
        sum_sq = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += x_val * x_val

        rms_factor = tl.rsqrt(tl.sum(sum_sq) / features + 1e-6)
        row_adarms_mod_offset = i * adarms_mod_stride
        for j in range(0, features, BLOCK_SIZE):
            cols = j + tl.arange(0, BLOCK_SIZE)
            mask = cols < features
            x_val = tl.load(x_ptr + row_x_offset + cols, mask=mask, other=0.0).to(tl.float32)
            x_norm = x_val * rms_factor
            s_scale = tl.load(adarms_mod_ptr + row_adarms_mod_offset + cols, mask=mask, other=0.0).to(tl.float32)
            s_shift = tl.load(adarms_mod_ptr + row_adarms_mod_offset + features + cols, mask=mask, other=0.0).to(tl.float32)
            s_gate = tl.load(adarms_mod_ptr + row_adarms_mod_offset + 2 * features + cols, mask=mask, other=0.0).to(tl.float32)
            output_val = x_norm * (1.0 + s_scale) + s_shift
            tl.store(normed_x_ptr + row_x_offset + cols, output_val.to(tl.bfloat16), mask=mask)
            tl.store(gate_ptr + row_x_offset + cols, s_gate.to(tl.bfloat16), mask=mask)

def adarms_norm_mod_proj_rowwise(
    x: torch.Tensor, adarms_mod: torch.Tensor, x_normed: torch.Tensor, gate: torch.Tensor
) -> None:
    seq_len = x.shape[0]
    adarms_norm_kernel_rowwise[(seq_len,)](
        x,
        adarms_mod,
        x_normed,
        gate,
        seq_len=seq_len,
        features=1024,
        adarms_mod_stride=3072,
        BLOCK_SIZE=512,
    )

def adarms_matmul_k_1024_32_bias_res_rowwise(
    x: torch.Tensor,
    x_normed: torch.Tensor,
    gate: torch.Tensor,
    adarms_mod: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    res: torch.Tensor,
) -> None:
    adarms_norm_mod_proj_rowwise(x, adarms_mod, x_normed, gate)
    seq_len = x.shape[0]
    matmul_small_bias_res[((seq_len + 15) // 16) * (32 // 16),](
        x_normed,
        weight,
        out,
        bias,
        res,
        seq_len=seq_len,
        features=1024,
        hidden=32,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_K=256,
    )

def _apply_prefill_mask(buffers: dict[str, torch.Tensor]) -> None:
    buffers["diffusion_noise"].mul_(buffers["prefill_inv_mask"])
    buffers["diffusion_noise"].add_(buffers["prefill_actions"] * buffers["prefill_mask"])

def transformer_decoder_rtc(weights, buffers, encoder_seq_len, num_steps=10):
    for step in range(num_steps):
        matmul_k_32_1024_bias(
            buffers["diffusion_noise"],
            weights["decoder_action_in_proj_w"],
            weights["decoder_action_in_proj_b"],
            buffers["decoder_x"],
        )
        seq_len = buffers["decoder_x"].shape[0]
        for i in range(18):
            adarms_norm_mod_proj_rowwise(
                buffers["decoder_x"],
                buffers["decoder_adarms_mod_attn"][step, i],
                buffers["x_normed_buf"],
                buffers["gate_buf"],
            )
            matmul_k_1024_2560_qkv_rope(
                buffers["x_normed_buf"],
                weights["decoder_attn_qkv_w"][i],
                buffers["decoder_rope_weights"],
                buffers["decoder_q_buf"],
                buffers["encoder_K"][i, encoder_seq_len : encoder_seq_len + seq_len],
                buffers["encoder_V"][i, encoder_seq_len : encoder_seq_len + seq_len],
            )
            total_queries = buffers["decoder_q_buf"].shape[0]
            prefix_keys = encoder_seq_len
            suffix_keys = seq_len
            total_keys = prefix_keys + suffix_keys

            matmul_abT_scale[(((total_queries + 31) // 32) * ((total_keys + 31) // 32),)](
                buffers["decoder_q_buf"],
                buffers["encoder_K"][i, : encoder_seq_len + seq_len],
                buffers["decoder_logits_buf"],
                total_queries,
                total_keys,
                256,
                256**-0.5,
                BLOCK_SIZE_M=32,
                BLOCK_SIZE_N=32,
                BLOCK_SIZE_K=64,
            )
            softmax_kernel_prefix_suffix[((total_queries + 3) // 4,)](
                buffers["decoder_logits_buf"],
                total_queries,
                prefix_keys,
                suffix_keys,
                buffers["valid_encoder_len"],
                buffers["decoder_attn_buf"],
                BLOCK_SIZE_M=4,
                BLOCK_SIZE=1024,
            )
            matmul_k8_n_256(
                buffers["decoder_attn_buf"],
                buffers["encoder_V"][i, : encoder_seq_len + seq_len],
                buffers["decoder_q_buf"],
            )
            matmul_k_2048_1024_gate(
                buffers["decoder_q_buf"].view(-1, 2048),
                weights["decoder_attn_o_w"][i],
                buffers["decoder_x"],
                buffers["gate_buf"],
            )
            adarms_norm_mod_proj_rowwise(
                buffers["decoder_x"],
                buffers["decoder_adarms_mod_ffn"][step, i],
                buffers["x_normed_buf"],
                buffers["gate_buf"],
            )
            seq_len = buffers["decoder_x"].shape[0]
            matmul_small_gate[((seq_len + 127) // 128, (4096 + 63) // 64)](
                buffers["x_normed_buf"],
                weights["decoder_ffn_gate_w"][i],
                weights["decoder_ffn_up_w"][i],
                buffers["decoder_hidden"],
                seq_len,
                1024,
                4096,
            )
            matmul_k_4096_1024_gate(
                buffers["decoder_hidden"],
                weights["decoder_ffn_down_w"][i],
                buffers["decoder_x"],
                buffers["gate_buf"],
            )

        adarms_matmul_k_1024_32_bias_res_rowwise(
            buffers["decoder_x"],
            buffers["x_normed_buf"],
            buffers["gate_buf"],
            buffers["decoder_adarms_mod_final"][step],
            weights["decoder_action_out_proj_w"],
            weights["decoder_action_out_proj_b"],
            buffers["diffusion_noise"],
            buffers["diffusion_noise"],
        )
        _apply_prefill_mask(buffers)

def pi05rtc_model(weights, buffers, num_views, encoder_seq_len, num_steps=10):
    vision_encoder(weights, buffers, num_views)
    transformer_encoder(weights, buffers, encoder_seq_len)
    transformer_decoder_rtc(weights, buffers, encoder_seq_len, num_steps=num_steps)

class Pi05RTCInference:
    def __init__(
        self,
        checkpoint,
        num_views,
        chunk_size,
        tokenizer_path: str | None = None,
        max_tokenize_len: int = 200,
        discrete_state_input: bool = True,
        max_prompt_text: str | None = None,
        state_dim_for_max_prompt: int | None = None,
    ):
        self.discrete_state_input = discrete_state_input
        self.tokenizer_path = tokenizer_path
        self.checkpoint = checkpoint
        self.num_views = num_views
        self.chunk_size = chunk_size
        self.max_tokenize_len = int(max_tokenize_len)
        if discrete_state_input:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if max_prompt_text is not None and state_dim_for_max_prompt is not None:
                self.max_prompt_len = self.estimate_max_prompt_len(
                    tokenizer=self.tokenizer,
                    task_prompt=max_prompt_text,
                    state_dim=int(state_dim_for_max_prompt),
                    max_tokenize_len=self.max_tokenize_len,
                    state_token_value=255,
                )
            else:
                self.max_prompt_len = self.max_tokenize_len
        else:
            self.max_prompt_len = len(checkpoint['language_embeds'])
        encoder_seq_len = num_views * 256 + self.max_prompt_len
        decoder_seq_len = chunk_size

        self.weights = {
            "vision_patch_embedding_w":           torch.empty(14, 14, 3, 1152,        dtype = torch.bfloat16, device = "cuda"),
            "vision_patch_embedding_b":           torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_position_embedding":          torch.empty(256, 1152,              dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_w":                  torch.empty(27, 1152, 3 * 1152,     dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_qkv_b":                  torch.empty(27, 3 * 1152,           dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_w":                    torch.empty(27, 1152, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_attn_o_b":                    torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_w":                    torch.empty(27, 1152, 4304,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_up_b":                    torch.empty(27, 4304,               dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_w":                  torch.empty(27, 4304, 1152,         dtype = torch.bfloat16, device = "cuda"),
            "vision_ffn_down_b":                  torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_w":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_attn_norm_b":             torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_w":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_pre_ffn_norm_b":              torch.empty(27, 1152,               dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_w":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),
            "vision_final_norm_b":                torch.empty(1152,                   dtype = torch.bfloat16, device = "cuda"),

            "encoder_multi_modal_projector_w":    torch.empty(1152, 2048,             dtype = torch.bfloat16, device = "cuda"),
            "encoder_multi_modal_projector_b":    torch.empty(2048,                   dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_qkv_w":                 torch.empty(18, 2048, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_attn_o_w":                   torch.empty(18, 2048, 2048,         dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_gate_w":                 torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_up_w":                   torch.empty(18, 2048, 16384,        dtype = torch.bfloat16, device = "cuda"),
            "encoder_ffn_down_w":                 torch.empty(18, 16384, 2048,        dtype = torch.bfloat16, device = "cuda"),

            "decoder_time_embeds":                torch.zeros(10, 1024,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_in_w":              torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_in_b":              torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_w":             torch.empty(1024, 1024,             dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_mlp_out_b":             torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_w":           torch.empty(32, 1024,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_in_proj_b":           torch.empty(1024,                   dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_attn_norm_mod_w":        torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"), 
            "decoder_pre_attn_norm_mod_b":        torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_pre_ffn_norm_mod_w":         torch.empty(18, 1024, 3 * 1024,     dtype = torch.bfloat16, device = "cuda"), 
            "decoder_pre_ffn_norm_mod_b":         torch.empty(18, 3 * 1024,           dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_qkv_w":                 torch.empty(18, 1024, 2560,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_attn_o_w":                   torch.empty(18, 2048, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_gate_w":                 torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_up_w":                   torch.empty(18, 1024, 4096,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_ffn_down_w":                 torch.empty(18, 4096, 1024,         dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_w":          torch.empty(1024, 32,               dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_out_proj_b":          torch.empty(32,                     dtype = torch.bfloat16, device = "cuda"),
            "decoder_final_norm_mod_w":           torch.empty(1024, 3 * 1024,         dtype = torch.bfloat16, device = "cuda"), 
            "decoder_final_norm_mod_b":           torch.empty(3 * 1024,               dtype = torch.bfloat16, device = "cuda"), 
            "language_embeds":                    torch.empty(len(checkpoint['language_embeds']), 2048,  dtype = torch.bfloat16, device = "cuda"),
        }

        self.buffers = {
            "observation_images_normalized":      torch.empty(num_views, 224, 224, 3,               dtype = torch.bfloat16, device = "cuda"),
            "diffusion_noise":                    torch.empty(decoder_seq_len, 32,                  dtype = torch.bfloat16, device = "cuda"),
            "vision_x":                           torch.empty(num_views, 256, 1152,                 dtype = torch.bfloat16, device = "cuda"),
            "vision_x_norm":                      torch.empty(num_views, 256, 1152,                 dtype = torch.bfloat16, device = "cuda"),
            "vision_QKV":                         torch.empty(num_views, 256, 3 * 1152,             dtype = torch.bfloat16, device = "cuda"),
            "vision_hidden":                      torch.empty(num_views, 256, 4304,                 dtype = torch.bfloat16, device = "cuda"),
            "vision_x_split_k_buf":               torch.empty((num_views * 256 * 1152 * 4,),        dtype = torch.float32,  device = "cuda"),
            "encoder_rope_weights":               torch.empty(encoder_seq_len, 256,                 dtype = torch.bfloat16, device = "cuda"),
            "encoder_x":                          torch.empty(encoder_seq_len, 2048,                dtype = torch.bfloat16, device = "cuda"),
            "encoder_x_norm":                     torch.empty(encoder_seq_len, 2048,                dtype = torch.bfloat16, device = "cuda"),
            "encoder_K":                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256, dtype= torch.bfloat16, device = "cuda"),
            "encoder_V":                          torch.empty(18, encoder_seq_len + decoder_seq_len, 256, dtype= torch.bfloat16, device = "cuda"),
            "encoder_Q":                          torch.empty(encoder_seq_len * 8, 256,             dtype = torch.bfloat16, device = "cuda"),
            "encoder_hidden":                     torch.empty(encoder_seq_len, 16384,               dtype = torch.bfloat16, device = "cuda"),
            "valid_encoder_len":                  torch.empty((1,),                                 dtype = torch.int32, device = "cuda"),
            "encoder_logits_buf":                 torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype = torch.float32, device = "cuda"),
            "encoder_attn_buf":                   torch.empty((encoder_seq_len * 8, encoder_seq_len), dtype = torch.bfloat16, device = "cuda"),
            "encoder_ctx_buf":                    torch.empty((encoder_seq_len * 8, 256),           dtype = torch.bfloat16, device = "cuda"),
            "decoder_rope_weights":               torch.empty(decoder_seq_len, 256,                 dtype = torch.bfloat16, device = "cuda"),
            "decoder_x":                          torch.empty((decoder_seq_len, 1024),              dtype = torch.bfloat16, device = "cuda"),
            "decoder_x_buf":                      torch.empty((decoder_seq_len, 1024),              dtype = torch.bfloat16, device = "cuda"),
            "decoder_action_buf":                 torch.empty((decoder_seq_len, 32),                dtype = torch.bfloat16, device = "cuda"),
            "decoder_time_emb":                   torch.empty((10, decoder_seq_len, 1024),          dtype = torch.bfloat16, device = "cuda"),
            "decoder_adarms_mod_attn":            torch.empty((10, 18, decoder_seq_len, 1024 * 3),  dtype = torch.bfloat16, device = "cuda"),
            "decoder_adarms_mod_ffn":             torch.empty((10, 18, decoder_seq_len, 1024 * 3),  dtype = torch.bfloat16, device = "cuda"),
            "decoder_adarms_mod_final":           torch.empty((10, decoder_seq_len, 1024 * 3),      dtype = torch.bfloat16, device = "cuda"),
            "decoder_q_buf":                      torch.empty((decoder_seq_len * 8, 256),           dtype = torch.bfloat16, device = "cuda"),
            "decoder_logits_buf":                 torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len), dtype = torch.float32, device = "cuda"),
            "decoder_attn_buf":                   torch.empty((decoder_seq_len * 8, encoder_seq_len + decoder_seq_len), dtype = torch.bfloat16, device = "cuda"),
            "decoder_hidden":                     torch.empty((decoder_seq_len, 4096),              dtype = torch.bfloat16, device = "cuda"),
            "x_normed_buf":                       torch.empty((decoder_seq_len, 1024),              dtype = torch.bfloat16, device = "cuda"),
            "gate_buf":                           torch.empty((decoder_seq_len, 1024),              dtype = torch.bfloat16, device = "cuda"),
            "prefill_actions":                    torch.zeros((decoder_seq_len, 32),                dtype = torch.bfloat16, device = "cuda"),
            "prefill_mask":                       torch.zeros((decoder_seq_len, 1),                 dtype = torch.bfloat16, device = "cuda"),
            "prefill_inv_mask":                   torch.ones((decoder_seq_len, 1),                  dtype = torch.bfloat16, device = "cuda"),
        }

        prefix_alloc = self.num_views * 256 + self.max_prompt_len
        max_pos = (self.num_views * 256 + self.max_prompt_len - 1) + self.chunk_size
        position_ids = torch.arange(max_pos + 1, device="cuda")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 256, 2, dtype=torch.float32, device="cuda") / 256))
        k_phase = inv_freq[None, :] * position_ids[:, None]
        k_cos = torch.cos(k_phase).to(torch.bfloat16)
        k_sin = torch.sin(k_phase).to(torch.bfloat16)
        self._rope_table = torch.cat([k_cos[:, :, None], k_sin[:, :, None]], 2).view(-1, 256)
        self.buffers["encoder_rope_weights"].copy_(self._rope_table[:prefix_alloc])
        self.buffers["valid_encoder_len"].fill_(self.num_views * 256 + 1)
        for k, v in checkpoint.items():
            if k != "embedding_weight":
                self.weights[k].copy_(v)

        self.weights["decoder_action_out_proj_w"] *= -1.0 / 10
        self.weights["decoder_action_out_proj_b"] *= -1.0 / 10

        self.prompt_embedding = None
        self._prompt_embed_scale = None
        if self.discrete_state_input:
            if "embedding_weight" not in checkpoint:
                raise KeyError("checkpoint must contain 'embedding_weight' when discrete_state_input=True")
            emb_w = checkpoint["embedding_weight"]
            if isinstance(emb_w, np.ndarray):
                emb_w = torch.from_numpy(emb_w)
            emb_w = emb_w.to(device="cuda", dtype=torch.bfloat16, non_blocking=True)
            self.prompt_embedding = nn.Embedding(
                num_embeddings=emb_w.shape[0],
                embedding_dim=emb_w.shape[1],
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                self.prompt_embedding.weight.copy_(emb_w)
            self._prompt_embed_scale = float(emb_w.shape[1] ** 0.5)

        self.encoder_seq_len = encoder_seq_len
        self._build_adarms_mod_bases()
        self._update_runtime_adarms_mods(prefill_len=0)

        self.infer_graph = torch.cuda.CUDAGraph()
        self.record_infer_graph()

    def estimate_max_prompt_len(
        self,
        tokenizer: AutoTokenizer,
        task_prompt: str,
        state_dim: int,
        max_tokenize_len: int = 200,
        state_token_value: int = 255,
    ) -> int:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join([str(int(state_token_value))] * int(state_dim))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=int(max_tokenize_len),
            padding=False,
        )["input_ids"][0]
        return int(token_ids.shape[0])

    def build_prompt_embeds(
        self,
        task_prompt: str,
        state_tokens: np.ndarray
    ) -> tuple[torch.Tensor, int]:
        task_prompt = task_prompt.strip().replace("_", " ")
        state_str = " ".join(map(str, state_tokens.tolist()))
        full_prompt = f"Task: {task_prompt}, State: {state_str};\nAction: "
        token_ids = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokenize_len,
            padding=False,
        )["input_ids"][0].to(device="cuda", non_blocking=True)
        embeds = self.prompt_embedding(token_ids) * self._prompt_embed_scale
        return embeds, int(embeds.shape[0])

    def get_decoder_rope_weights(self, prompt_len: int) -> torch.Tensor:
        start = self.num_views * 256 + prompt_len - 1
        end = start + self.chunk_size
        return self._rope_table[start:end]

    def _build_adarms_mod_bases(self) -> None:
        t_embeds = []
        for step in range(10):
            t0 = self.weights["decoder_time_embeds"][step].float()
            t1 = _swish(torch.matmul(t0, self.weights["decoder_time_mlp_in_w"].float()) + self.weights["decoder_time_mlp_in_b"].float())
            t2 = _swish(torch.matmul(t1, self.weights["decoder_time_mlp_out_w"].float()) + self.weights["decoder_time_mlp_out_b"].float())
            t_embeds.append(t2)
        self._time_step_emb = torch.stack(t_embeds, dim=0).to(torch.float32)

        t_zero = _posemb_sincos_torch(torch.zeros(1, dtype=torch.float32, device="cuda"))[0]
        t_zero = _swish(
            torch.matmul(t_zero, self.weights["decoder_time_mlp_in_w"].float()) + self.weights["decoder_time_mlp_in_b"].float()
        )
        t_zero = _swish(
            torch.matmul(t_zero, self.weights["decoder_time_mlp_out_w"].float()) + self.weights["decoder_time_mlp_out_b"].float()
        )
        self._time_zero_emb = t_zero.to(torch.float32)

        attn_w = self.weights["decoder_pre_attn_norm_mod_w"].float()
        attn_b = self.weights["decoder_pre_attn_norm_mod_b"].float()
        ffn_w = self.weights["decoder_pre_ffn_norm_mod_w"].float()
        ffn_b = self.weights["decoder_pre_ffn_norm_mod_b"].float()
        final_w = self.weights["decoder_final_norm_mod_w"].float()
        final_b = self.weights["decoder_final_norm_mod_b"].float()

        self._base_adarms_mod_attn_vec = (
            torch.einsum("sd,ldh->slh", self._time_step_emb, attn_w) + attn_b[None, :, :]
        ).to(
            torch.bfloat16
        )
        self._base_adarms_mod_ffn_vec = (
            torch.einsum("sd,ldh->slh", self._time_step_emb, ffn_w) + ffn_b[None, :, :]
        ).to(
            torch.bfloat16
        )
        self._base_adarms_mod_final_vec = (torch.matmul(self._time_step_emb, final_w) + final_b[None, :]).to(
            torch.bfloat16
        )

        self._base_adarms_mod_attn_t0 = (torch.einsum("d,ldh->lh", self._time_zero_emb, attn_w) + attn_b).to(
            torch.bfloat16
        )
        self._base_adarms_mod_ffn_t0 = (torch.einsum("d,ldh->lh", self._time_zero_emb, ffn_w) + ffn_b).to(
            torch.bfloat16
        )
        self._base_adarms_mod_final_t0 = (torch.matmul(self._time_zero_emb, final_w) + final_b).to(torch.bfloat16)

    def _update_runtime_adarms_mods(
        self,
        prefill_len: int,
    ) -> None:
        self.buffers["decoder_adarms_mod_attn"][:] = self._base_adarms_mod_attn_vec[:, :, None, :]
        self.buffers["decoder_adarms_mod_ffn"][:] = self._base_adarms_mod_ffn_vec[:, :, None, :]
        self.buffers["decoder_adarms_mod_final"][:] = self._base_adarms_mod_final_vec[:, None, :]

        if prefill_len > 0:
            self.buffers["decoder_adarms_mod_attn"][:, :, :prefill_len, :] = self._base_adarms_mod_attn_t0[
                None, :, None, :
            ]
            self.buffers["decoder_adarms_mod_ffn"][:, :, :prefill_len, :] = self._base_adarms_mod_ffn_t0[
                None, :, None, :
            ]
            self.buffers["decoder_adarms_mod_final"][:, :prefill_len, :] = self._base_adarms_mod_final_t0[
                None, None, :
            ]

    def _prepare_prefill(
        self,
        diffusion_noise: torch.Tensor,
        prefill_actions,
        action_prefill_len,
    ) -> int:
        prefill_len = int(action_prefill_len or 0)
        prefill_len = max(0, min(prefill_len, self.chunk_size))

        self.buffers["prefill_actions"].zero_()
        self.buffers["prefill_mask"].zero_()
        self.buffers["prefill_inv_mask"].fill_(1)
        if prefill_len == 0:
            return 0
        if prefill_actions is None:
            raise ValueError("action_prefill_len > 0 requires prefill_actions.")
        pref = torch.from_numpy(prefill_actions).to(device="cuda", dtype=torch.bfloat16)
        copy_len = min(int(pref.shape[0]), self.chunk_size)
        self.buffers["prefill_actions"][:copy_len].copy_(pref[:copy_len])
        self.buffers["prefill_mask"][:prefill_len].fill_(1)
        self.buffers["prefill_inv_mask"][:prefill_len].fill_(0)
        diffusion_noise[:prefill_len].copy_(self.buffers["prefill_actions"][:prefill_len])
        return prefill_len

    def record_run(self):
        pi05rtc_model(self.weights, self.buffers, self.num_views, self.encoder_seq_len)

    def record_infer_graph(self):
        for _ in range(3):
            self.record_run()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.infer_graph.capture_begin()
            self.record_run()
            self.infer_graph.capture_end()

    def forward(
        self,
        observation_images_normalized: torch.Tensor,
        diffusion_noise: torch.Tensor,
        task_prompt: str = None,
        state_tokens: np.ndarray = None,
        action_prefill_len = None,
        prefill_actions = None,
    ) -> torch.Tensor:
        if self.discrete_state_input:
            prompt_embeds, prompt_len = self.build_prompt_embeds(task_prompt=task_prompt, state_tokens=state_tokens)
        else:
            prompt_embeds = self.weights["language_embeds"]
            prompt_len = self.weights["language_embeds"].shape[0]

        start = self.num_views * 256
        self.buffers["encoder_x"][start : start + prompt_len].copy_(prompt_embeds)
        self.buffers["valid_encoder_len"].fill_(start + prompt_len)
        self.buffers["decoder_rope_weights"].copy_(self.get_decoder_rope_weights(prompt_len))

        self.buffers["observation_images_normalized"].copy_(observation_images_normalized)
        self.buffers["diffusion_noise"].copy_(diffusion_noise)

        prefill_len = self._prepare_prefill(self.buffers["diffusion_noise"], prefill_actions, action_prefill_len)
        self._update_runtime_adarms_mods(prefill_len)

        self.infer_graph.replay()
        return self.buffers["diffusion_noise"]