#!/usr/bin/env python3
import struct
from pathlib import Path
import argparse

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed")
    exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: NumPy not installed")
    exit(1)


CACTUS_MAGIC = b"CACT"
CACTUS_ALIGNMENT = 32
FLAG_TRANSPOSED = 1 << 2


def align_offset(offset, alignment):
    remainder = offset % alignment
    return offset if remainder == 0 else offset + (alignment - remainder)


def compute_padding(current_offset, alignment):
    aligned = align_offset(current_offset, alignment)
    return b"\x00" * (aligned - current_offset)


def save_tensor_fp16(tensor, output_path, transpose=False):
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)

    shape = list(data.shape)
    if transpose and len(shape) == 2:
        data = data.T
        shape = [shape[1], shape[0]]

    data = data.astype(np.float16).flatten()

    with open(output_path, "wb") as f:
        ndim = len(shape)
        flags = FLAG_TRANSPOSED if transpose else 0

        f.write(CACTUS_MAGIC)
        f.write(struct.pack("<I", flags))
        f.write(struct.pack("<I", CACTUS_ALIGNMENT))
        f.write(struct.pack("<I", ndim))

        for i in range(4):
            f.write(struct.pack("<Q", shape[i] if i < ndim else 0))

        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", data.size * 2))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<Q", shape[0] if ndim >= 1 else 0))

        f.write(compute_padding(84, CACTUS_ALIGNMENT))
        f.write(data.tobytes())


def convert_silero_vad_weights(model, output_dir, precision="FP16"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()

    stft_basis = state_dict["_model.stft.forward_basis_buffer"]
    n_fft_bins, _, window_size = stft_basis.shape

    lstm_weight_ih = state_dict["_model.decoder.rnn.weight_ih"]
    lstm_hidden_size = lstm_weight_ih.shape[1]

    encoder_channels = []
    for i in range(4):
        key = f"_model.encoder.{i}.reparam_conv.weight"
        if key in state_dict:
            weight = state_dict[key]
            out_ch, in_ch, kernel = weight.shape
            encoder_channels.append((in_ch, out_ch, kernel))

    config = {
        "model_type": "silero_vad",
        "sampling_rate": 16000,
        "window_size": int(window_size),
        "n_fft_bins": int(n_fft_bins),
        "num_encoder_blocks": len(encoder_channels),
        "lstm_hidden_size": int(lstm_hidden_size),
        "model_variant": "default",
        "precision": "FP16",
    }

    save_tensor_fp16(stft_basis, output_dir / "stft_basis.weights")

    for i in range(config["num_encoder_blocks"]):
        save_tensor_fp16(
            state_dict[f"_model.encoder.{i}.reparam_conv.weight"],
            output_dir / f"encoder_block_{i}_conv_weight.weights",
        )
        save_tensor_fp16(
            state_dict[f"_model.encoder.{i}.reparam_conv.bias"],
            output_dir / f"encoder_block_{i}_conv_bias.weights",
        )

    lstm_weights = [
        ("_model.decoder.rnn.weight_ih", "lstm_weight_ih.weights"),
        ("_model.decoder.rnn.weight_hh", "lstm_weight_hh.weights"),
        ("_model.decoder.rnn.bias_ih", "lstm_bias_ih.weights"),
        ("_model.decoder.rnn.bias_hh", "lstm_bias_hh.weights"),
    ]
    for key, filename in lstm_weights:
        save_tensor_fp16(state_dict[key], output_dir / filename)

    save_tensor_fp16(
        state_dict["_model.decoder.decoder.2.weight"],
        output_dir / "output_conv_weight.weights",
    )
    save_tensor_fp16(
        state_dict["_model.decoder.decoder.2.bias"],
        output_dir / "output_conv_bias.weights",
    )

    with open(output_dir / "config.txt", "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert Silero-VAD weights to Cactus format"
    )
    parser.add_argument("output_dir", nargs="?", default="../weights/silero-vad")
    parser.add_argument("--precision", choices=["FP16"], default="FP16")
    args = parser.parse_args()

    print("Loading Silero-VAD from torch.hub...")
    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)

    print(f"Converting to {args.output_dir}...")
    convert_silero_vad_weights(model, args.output_dir, args.precision)

    print("✅ Done")


if __name__ == "__main__":
    main()
