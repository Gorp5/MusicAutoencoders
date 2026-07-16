import lmdb
import pickle
import torch
import os
import random

import numpy as np
from torch.utils.data import Dataset


class StreamViewDataset(Dataset):
    def __init__(self, lmdb_root, split="train", num_shards=4,
                 chunk_size=256, views=2):
        self.lmdb_root = lmdb_root
        self.split = split
        self.num_shards = num_shards
        self.chunk_size = chunk_size
        self.view_count = views

        # Load IDs instantly from numpy file (no LMDB scanning!)
        id_file = os.path.join(lmdb_root, f"{split}_ids.npy")
        self.track_ids = np.load(id_file)

        # Open envs lazily per-worker to prevent fork/bloat issues
        self.envs = None

    def _init_envs(self):
        """Lazy init - called in worker process, not main process"""
        if self.envs is None:
            self.envs = [lmdb.open(
                f"{self.lmdb_root}/{self.split}_shard{i}.lmdb",
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=4
            ) for i in range(self.num_shards)]

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        self._init_envs()  # Open LMDB in worker, not main process

        track_id = int(self.track_ids[idx])
        shard = track_id % self.num_shards

        with self.envs[shard].begin() as txn:
            data_bytes = txn.get(f"{track_id:06d}".encode())

        # Process outside the transaction - allows LMDB to free the read snapshot
        np_arr = pickle.loads(data_bytes)
        spec = torch.from_numpy(np_arr.copy()).float()

        del np_arr, data_bytes

        # Generate views
        views, masks = [], []
        for _ in range(self.view_count):
            size = self.chunk_size

            if spec.shape[1] <= size:
                # Pad if too short
                pad = size - spec.shape[1]
                view = torch.nn.functional.pad(spec, (0, pad))
                mask = torch.cat([torch.ones(spec.shape[1]), torch.zeros(pad)]).bool()
            else:
                start = random.randint(0, spec.shape[1] - size)
                view = spec[:, start:start + size]
                mask = torch.ones(size, dtype=torch.bool)

            views.append(view)
            masks.append(mask)

        return track_id, torch.stack(views), torch.stack(masks)

class MemmapDataset:
    def __init__(
        self,
        memmap_root,
        split="train",
        chunk_size=256,
        views=2,
        min_chunk=-1,
        max_chunk=-1,
        dtype=np.float16,
        freq_bins=128,
    ):
        self.memmap_root = memmap_root
        self.split = split
        self.chunk_size = chunk_size
        self.view_count = views
        self.stochastic = min_chunk > 0 and max_chunk > 0
        self.min_chunk, self.max_chunk = min_chunk, max_chunk
        self.dtype = dtype
        self.freq_bins = freq_bins

        index_path = os.path.join(memmap_root, "index.npy")
        full_index = np.load(index_path)

        self.index = full_index[full_index["split"] == split]

        self.track_ids = self.index["track_id"]
        self.memmaps = {}

    def __len__(self):
        return len(self.index)

    def _get_memmap(self, file_id):
        if file_id not in self.memmaps:
            path = os.path.join(
                self.memmap_root,
                f"{self.split}_{file_id}.bin"   # FIXED
            )

            mm = np.memmap(
                path,
                dtype=self.dtype,
                mode="r"
            )

            mm = mm.reshape(-1, self.freq_bins)
            self.memmaps[file_id] = mm

        return self.memmaps[file_id]

    def __getitem__(self, idx):
        row = self.index[idx]

        track_id = int(row["track_id"])
        file_id = int(row["file_id"])
        start = int(row["start"])
        length = int(row["length"])

        mm = self._get_memmap(file_id)
        spec = mm[start:start + length]

        spec = torch.from_numpy(spec.copy()).float().T

        views, masks = [], []

        for _ in range(self.view_count):
            size = self.chunk_size

            if spec.shape[1] <= size:
                pad = size - spec.shape[1]
                view = torch.nn.functional.pad(spec, (0, pad))
                mask = torch.cat([
                    torch.ones(spec.shape[1]),
                    torch.zeros(pad)
                ]).bool()

            else:
                start_idx = random.randint(0, spec.shape[1] - size)
                view = spec[:, start_idx:start_idx + size]
                mask = torch.ones(size, dtype=torch.bool)

            views.append(view)
            masks.append(mask)

        return track_id, torch.stack(views), torch.stack(masks)