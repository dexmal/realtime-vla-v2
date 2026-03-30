import argparse
import pickle
import torch
from convert_from_jax_pi05 import (
    convert_weights_pi05,
    load_jax_weights,
    prepare_adarms_cond,
    prepare_prompt,
)

def build_empty_weights(prompt_len: int):
    return {
        "embedding_weight": torch.zeros(257152, 2048, dtype=torch.bfloat16, device="cpu"),
        "vision_patch_embedding_w": torch.zeros(14, 14, 3, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_patch_embedding_b": torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),
        "vision_position_embedding": torch.zeros(256, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_w": torch.zeros(27, 1152, 3 * 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_b": torch.zeros(27, 3 * 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_w": torch.zeros(27, 1152, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_b": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_w": torch.zeros(27, 1152, 4304, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_b": torch.zeros(27, 4304, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_w": torch.zeros(27, 4304, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_b": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_w": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_b": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_w": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_b": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_w": torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_b": torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),
        "encoder_multi_modal_projector_w": torch.zeros(1152, 2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_multi_modal_projector_b": torch.zeros(2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_qkv_w": torch.zeros(18, 2048, 2560, dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_o_w": torch.zeros(18, 2048, 2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_gate_w": torch.zeros(18, 2048, 16384, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_up_w": torch.zeros(18, 2048, 16384, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_down_w": torch.zeros(18, 16384, 2048, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_embeds": torch.zeros(10, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_in_w": torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_in_b": torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_out_w": torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_time_mlp_out_b": torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_attn_norm_mod_w": torch.zeros(18, 1024, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_attn_norm_mod_b": torch.zeros(18, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_ffn_norm_mod_w": torch.zeros(18, 1024, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_pre_ffn_norm_mod_b": torch.zeros(18, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_final_norm_mod_w": torch.zeros(1024, 3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_final_norm_mod_b": torch.zeros(3 * 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_qkv_w": torch.zeros(18, 1024, 2560, dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_o_w": torch.zeros(18, 2048, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_gate_w": torch.zeros(18, 1024, 4096, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_up_w": torch.zeros(18, 1024, 4096, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_down_w": torch.zeros(18, 4096, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_w": torch.zeros(32, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_b": torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_out_proj_w": torch.zeros(1024, 32, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_out_proj_b": torch.zeros(32, dtype=torch.bfloat16, device="cpu"),
        "language_embeds": torch.zeros(prompt_len, 2048, dtype=torch.bfloat16, device="cpu"),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert JAX Pi0.5 RTC weights to Triton Pi05RTC format.")
    # parser.add_argument("--jax_path", type=str, required=True)
    # parser.add_argument("--output", type=str, required=True)
    # parser.add_argument("--prompt", type=str, required=True)
    # parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--jax_path", type=str, default="/self_group/yc/ckpts/pi05_ft_dosw1_qpos_foldclothes_merging_stage1/quant_baseline/49999")
    parser.add_argument("--output", type=str, default="/mlp_vepfs/share/myc/weight/openpi/pi05_ft_dosw1_qpos_foldclothes_merging_stage1_quant_baseline_49999_rtc2.pkl")
    parser.add_argument("--prompt", type=str, default="fold the cloth")
    parser.add_argument("--tokenizer_path", type=str, default="/mlp_vepfs/share/myc/weight/openpi/paligemma-3b-pt-224")
    args = parser.parse_args()

    dump_weights = load_jax_weights(args.jax_path)
    embedding_weight = dump_weights["PaliGemma"]["llm"]["embedder"]["input_embedding"]["value"]

    language_embeds, prompt_len = prepare_prompt(args.prompt, embedding_weight, args.tokenizer_path)
    time_embeds = prepare_adarms_cond(num_steps=10)

    weights = build_empty_weights(prompt_len)
    convert_weights_pi05(weights, dump_weights)

    weights["embedding_weight"].copy_(torch.as_tensor(embedding_weight, dtype=torch.bfloat16, device="cpu"))
    weights["language_embeds"].copy_(language_embeds.to(dtype=torch.bfloat16, device="cpu"))
    weights["decoder_time_embeds"].copy_(time_embeds.to(dtype=torch.bfloat16, device="cpu"))

    with open(args.output, "wb") as f:
        pickle.dump(weights, f)
    print(f"Successfully converted Pi05 RTC weights to {args.output}")

if __name__ == "__main__":
    main()
