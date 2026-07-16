import argparse
import torch

from contrastive_training import train_contrastive
from data.data_utils import StreamViewDataset, MemmapDataset
from models.Myna import Myna
from utils.Config import Config

masking_ratios = [0.25, 0.5, 0.75, 0.9]
training_chunk_lengths = [128, 256, 512, 1024, 2048]
embedding_strategy = ["alibi_2d_learned", "alibi_1d", "alibi_2d", "rope_1d", "rope_2d", "sinusoidal_raster", "learned_x", "none", "sinusoidal_xy", "rope_double_frequency"]

BASE_CONFIG = dict(
    alibi_x=False,
    alibi_y=False,
    alibi_learned_slopes=False,
    rope_x=False,
    rope_y=False,
    sinusoidal_raster=False,
    sinusoidal_x=False,
    sinusoidal_y=False,
    learned_x=False,
    learned_y=False,
)

embedding_configs = [
    dict(name="alibi_2d_learned", alibi_x=True, alibi_y=True, alibi_learned_slopes=True),
    dict(name="alibi_2d", alibi_x=True, alibi_y=True),
    dict(name="rope_2d", rope_x=True, rope_y=True),
    dict(name="alibi_1d",         alibi_x=True),
    dict(name="rope_1d",          rope_x=True),
    dict(name="sinusoidal_raster", sinusoidal_raster=True),
    dict(name="learned_x",        learned_x=True, learned_y=True),
    dict(name="none"),
    dict(name="sinusoidal_xy",    sinusoidal_x=True, sinusoidal_y=True),
    dict(name="rope_double_frequency", rope_x=True, rope_y=True),
]

import os

from info_nce import InfoNCE
from loss.loss_utils import *
from datasets import tqdm
from torch import optim


def train_contrastive(model, test_dataloader, train_dataloader, config, start_epoch=0):
    # Training setup
    file_path = f"{config.save_path}\\Config.pt"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(config, file_path)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    criterion = InfoNCE()
    device = "cuda"
    model = model.to(device=device, dtype=config.dtype)
    torch.autograd.set_detect_anomaly(True)

    if start_epoch == 0:
        f = open(f"{config.save_path}\\Loss.txt", "w")
        f.close()

    # Training loop
    step = 1
    for epoch in range(start_epoch, config.num_epochs):
        batch_steps = 0
        epoch_same_song_contrastive_loss = 0

        batches = len(train_dataloader)
        pbar = tqdm(train_dataloader)
        for batch in pbar:
            indicies, inputs, masks = batch

            B, _, T, F = inputs.shape

            inputs = inputs.to(device=device)

            # forward pass
            stacked = inputs.view(B * 2, T, F).unsqueeze(1)

            stacked = model(stacked, mask=None)
            z_list = stacked.squeeze(1).view(B, 2, -1)

            contrastive_loss = 0

            for index in range(1, len(z_list)):
                contrastive_loss += criterion(z_list[0], z_list[index])

            contrastive_loss = contrastive_loss / (len(z_list) - 1)
            loss = contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_same_song_contrastive_loss += contrastive_loss.item()

            step += 1
            batch_steps += 1

            term = f"Contrastive Loss [{batch_steps}/{batches}]: {contrastive_loss.item():.4f}"

            with open(f"{config.save_path}\\Loss.txt", "a") as f:
                term += "\n"
                f.write(term)

        same_song_contrastive_loss = evaluate_contrastive(model, test_dataloader, config)

        term = f"[Epoch {epoch}] Train: Same Song Contrastive Loss = {epoch_same_song_contrastive_loss / batch_steps:.4f}"

        term += "\n"
        term += f"Test: Same Song Contrastive Loss = {same_song_contrastive_loss:.4f}"

        term += "\n"

        print(term)

        torch.save(model, f".\\{config.save_path}\\Epoch-{epoch}.pt")


def evaluate_contrastive(model, dataloader, config):
    song_contrastive_loss_total = 0

    criterion = InfoNCE()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            indicies, inputs, masks = batch

            B, _, T, F = inputs.shape

            inputs = inputs.to(device=config.device)

            # forward pass
            stacked = inputs.view(B * 2, T, F).unsqueeze(1)

            stacked = model(stacked, mask=None)
            z_list = stacked.squeeze(1).view(B, 2, -1)

            contrastive_loss = 0
            for index in range(1, len(z_list)):
                contrastive_loss += criterion(z_list[0], z_list[index])

            contrastive_loss = contrastive_loss / (len(z_list) - 1)
            song_contrastive_loss_total += contrastive_loss.item()

    return song_contrastive_loss_total / len(dataloader)

embedding_strategy_params = []
def determine_based_on_id(id):
    masking_ratio_index = id % 4
    training_length_index = (id // 4) % 5
    type_index = (id // 20) % 10

    config = BASE_CONFIG.copy()
    config.update(embedding_configs[type_index])

    return masking_ratios[masking_ratio_index], training_chunk_lengths[training_length_index], config

import sys
import os

if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--latent_projection_method", type=str, required=False, default="cls")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--rope_base", type=int, required=False, default=4096)
    parser.add_argument("--chunk_length", type=int, required=False, default=256)
    parser.add_argument("--use_time_chunking", type=bool, required=False, default=False)

    args = parser.parse_args()

    id = args.id
    mask_ratio, training_chunk_lengths, params = determine_based_on_id(id)

    use_alibi_x = params["alibi_x"]
    use_alibi_y = params["alibi_y"]
    use_learned_alibi_slopes = params["alibi_learned_slopes"]
    use_rope_x = params["rope_x"]
    use_rope_y = params["rope_y"]
    use_sinusoidal_raster = params["sinusoidal_raster"]
    use_sinusoidal_x = params["sinusoidal_x"]
    use_sinusoidal_y = params["sinusoidal_y"]
    use_learned_x = params["learned_x"]
    use_learned_y = params["learned_y"]
    name = params["name"]

    print(f"Running task {id}: {name}:{id % 20}")

    per_gpu_batch = args.batch_size

    config = Config(
        save_path=args.save_dir,
        num_epochs=args.epochs,
        learning_rate=3e-4,
        weight_decay=1e-4,
        num_workers=2,
        batch_size=per_gpu_batch,
        eval_batch_size=per_gpu_batch,
        dtype=torch.float32,
        device="cuda"
    )

    patch_size = (16, 16)
    if args.use_time_chunking:
        patch_size = (128, 1)

    # Determines which model is trained
    model = Myna(
        image_size=(128, args.chunk_length),
        channels=1,
        patch_size=(16, 16),
        latent_space=128,
        d_model=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        mask_ratio=mask_ratio,
        latent_projection_method=args.latent_projection_method,
        use_sinusoidal_x=use_sinusoidal_x,
        use_sinusoidal_y=use_sinusoidal_y,
        use_sinusoidal_raster=use_sinusoidal_raster,
        use_learned_encoding_y=use_learned_y,
        use_learned_encoding_x=use_learned_x,
        use_rope_x=use_rope_x,
        use_rope_y=use_rope_y,
        use_rope_double_frequency=False,
        use_learned_alibi_slopes=use_learned_alibi_slopes,
        use_alibi_x=use_alibi_x,
        use_alibi_y=use_alibi_y,
        rope_base=8192
    )

    train_dataset = MemmapDataset(args.train_data_dir, split="train", views=2, chunk_size=args.chunk_length)
    test_dataset = MemmapDataset(args.test_data_dir, split="test", views=2, chunk_size=args.chunk_length)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=3,
        pin_memory=True,
        shuffle=True,
        # prefetch_factor=1,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=3,
        # prefetch_factor=1
    )

    train_contrastive(model, test_dataloader, train_dataloader, config)