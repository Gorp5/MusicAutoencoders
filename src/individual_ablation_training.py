import argparse
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim

from models.Myna import Myna
from data.data_utils import MemmapDataset
from info_nce import InfoNCE

mp.set_start_method("spawn", force=True)
torch.multiprocessing.set_sharing_strategy("file_system")

def build_model(mask_ratio, chunk_length, embedding_params, device):
    if os.path.exists("/mnt/ssd/output/GPU-0/model_0.pt"):
        return torch.load("/mnt/ssd/output/GPU-0/model_0.pt")["model"]
    else:
        return Myna(
            image_size=(128, chunk_length),
            channels=1,
            patch_size=(16, 16),
            latent_space=128,
            d_model=384,
            depth=12,
            heads=6,
            mlp_dim=1536,
            mask_ratio=mask_ratio,
            latent_projection_method="cls",
            use_sinusoidal_x=embedding_params.get("sinusoidal_x", False),
            use_sinusoidal_y=embedding_params.get("sinusoidal_y", False),
            use_sinusoidal_raster=embedding_params.get("sinusoidal_raster", False),
            use_learned_encoding_x=embedding_params.get("learned_x", False),
            use_learned_encoding_y=embedding_params.get("learned_y", False),
            use_rope_x=embedding_params.get("rope_x", False),
            use_rope_y=embedding_params.get("rope_y", False),
            use_alibi_x=embedding_params.get("alibi_x", False),
            use_alibi_y=embedding_params.get("alibi_y", False),
            use_learned_alibi_slopes=embedding_params.get("alibi_learned_slopes", False),
            rope_base=8192,
            device=device
        )


def build_dataloader(dataset_path, batch_size, chunk_length):
    dataset = MemmapDataset(dataset_path, split="train", views=2, chunk_size=chunk_length)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=1
    )


def gpu_worker(args, params):
    print(f"Training Starting starting")

    device = torch.device("cuda")

    models, optimizers = [], []

    model = build_model(
        params["mask_ratio"],
        params["chunk_length"],
        params["embedding_params"],
        device
    ).to(device)

    model.train()

    epochs_done = 0
    save_dir = os.path.join(args.save_dir, f"output")
    save_file = os.path.join(save_dir, f"model.pt")

    if os.path.exists(save_file):
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        saved = torch.load(save_file)

        optimizer.load_state_dict(saved["optimizer_state_dict"])
        epochs_done = saved["epoch"]
        total_losses = saved["loss"]
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        total_losses = []

    criterion = InfoNCE()

    dataloader = build_dataloader(
        args.train_data_dir,
        args.batch_size,
        args.chunk_length
    )


    os.makedirs(save_dir, exist_ok=True)

    i = 0

    # Training
    for epoch in range(epochs_done, args.epochs):
        losses = 0.0

        pbar = tqdm(dataloader, desc=f"GPU {gpu_id} Epoch {epoch}")

        for batch in pbar:
            _, inputs, _ = batch
            inputs = inputs.to(device, non_blocking=True)

            chunk_len = params["chunk_length"]
            sliced = inputs[:, :, :, :chunk_len]
            B, _, T, F = sliced.shape

            # Flatten views
            stacked = sliced.view(B * 2, T, F).unsqueeze(1)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z = model(stacked, mask=None, checkpointing=False).squeeze(1).view(B, 2, -1)

            optimizer.zero_grad()

            loss = criterion(z[:, 0], z[:, 1])
            loss.backward()
            optimizer.step()

            losses += loss.item()

        # Saving
        final_epoch_loss = losses[i] / len(dataloader)
        total_losses.append(final_epoch_loss)

        torch.save({
            "epoch": epoch,
            "model": model,
            "optimizer_state_dict": optimizers[i].state_dict(),
            "loss": total_losses
        }, os.path.join(save_dir, f"model_{i}.pt"))

def determine_based_on_id(id):
    masking_ratio_array = [0.25, 0.5, 0.75, 0.9]
    training_chunk_length_array = [128, 256, 512, 1024, 2048]

    embedding_configs = [
        dict(name="alibi_2d_learned", alibi_x=True, alibi_y=True, alibi_learned_slopes=True),
        dict(name="alibi_2d", alibi_x=True, alibi_y=True),
        dict(name="rope_2d", rope_x=True, rope_y=True),
        dict(name="alibi_1d", alibi_x=True),
        dict(name="rope_1d", rope_x=True),
        dict(name="sinusoidal_raster", sinusoidal_raster=True),
        dict(name="learned_x", learned_x=True, learned_y=True),
        dict(name="none"),
        dict(name="sinusoidal_xy", sinusoidal_x=True, sinusoidal_y=True),
        dict(name="rope_double_frequency", rope_x=True, rope_y=True),
    ]

    config = embedding_configs[id // 20]

    return (
        masking_ratio_array[id % len(masking_ratio_array)],
        training_chunk_length_array[(id // 4) % len(training_chunk_length_array)],
        config
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--chunk_length", type=int, required=True)

    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--num_models", type=int, required=True)

    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()

    models_per_gpu = [[] for _ in range(args.num_gpus)]

    i = args.id
    gpu_id = i % args.num_gpus

    mask_ratio, chunk_length, params = determine_based_on_id(i)

    config = {
        "mask_ratio": mask_ratio,
        "chunk_length": chunk_length,
        "embedding_params": params
    }

    gpu_worker(args, config)