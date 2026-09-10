"""Add quantization_config to model_merged config for pre-quantized BNB NF4 weights."""
import json

config_path = "/root/autodl-tmp/model_merged/config.json"
config = json.load(open(config_path))

# Add BNB NF4 quantization config (matching the pre-quantized weights)
config["quantization_config"] = {
    "_load_in_4bit": True,
    "_load_in_8bit": False,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_storage": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": False,
    "llm_int8_enable_fp32_cpu_offload": False,
    "llm_int8_has_fp16_weight": False,
    "llm_int8_skip_modules": None,
    "llm_int8_threshold": 6.0,
    "load_in_4bit": True,
    "load_in_8bit": False,
    "quant_method": "bitsandbytes"
}

json.dump(config, open(config_path, "w"), indent=2)
print("quantization_config added")
print(json.dumps(config["quantization_config"], indent=2))
