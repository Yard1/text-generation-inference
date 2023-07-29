import torch


def get_bnb_kwargs(quantize: str, dtype: torch.dtype) -> dict:
    bnb_kwargs = {}
    if quantize == "bitsandbytes":
        bnb_kwargs["load_in_8bit"] = True
    elif quantize == "bitsandbytes-fp4":
        bnb_kwargs["load_in_4bit"] = True
        bnb_kwargs["bnb_4bit_quant_type"] = "fp4"
        bnb_kwargs["bnb_4bit_compute_dtype"] = dtype
    elif quantize == "bitsandbytes-nf4":
        bnb_kwargs["load_in_4bit"] = True
        bnb_kwargs["bnb_4bit_quant_type"] = "nf4"
        bnb_kwargs["bnb_4bit_compute_dtype"] = dtype
    return bnb_kwargs