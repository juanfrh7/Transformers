import torch
import numpy as np
import random
import pandas as pd

### Language modeling ###
@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        data = train_data if split == "train" else val_data

        for k in range(eval_iters):
            X, Y = get_lm_batch(
                data,
                block_size=block_size,
                batch_size=batch_size
            )

            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()

    return out

def get_lm_batch(data, block_size, batch_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


### Image classification ###
def get_image_batch(X, y, batch_size=64, device=None):

    N = X.shape[0]

    indices = np.random.choice(N, size=batch_size, replace=False)

    X_batch = torch.tensor(X[indices], dtype=torch.float32, device=device)
    y_batch = torch.tensor(y[indices], dtype=torch.float32, device=device)

    return X_batch, y_batch

def get_image_batches(X, y, batch_size=64, device=None):
    N = X.shape[0]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        X_batch = torch.tensor(
            X[start:end],
            dtype=torch.float32,
            device=device
        )

        y_batch = torch.tensor(
            y[start:end],
            dtype=torch.float32,
            device=device
        )

        yield X_batch, y_batch

def patchify_images(X, num_patches=16):
    """
    X: array of shape (N, H, W)
    Returns: patches of shape (N, num_patches, patch_size, patch_size)
    """
    N, H, W = X.shape

    patch_size = int(H / (num_patches ** 0.5))
    assert H % patch_size == 0, "Image height must be divisible by patch_size"
    assert W % patch_size == 0, "Image width must be divisible by patch_size"

    n_patches_h = H // patch_size
    n_patches_w = W // patch_size

    patches = X.reshape(
        N,
        n_patches_h,
        patch_size,
        n_patches_w,
        patch_size)

    patches = patches.transpose(0, 1, 3, 2, 4)
    patches = patches.reshape(N, -1,patch_size*patch_size)

    return patches


def make_balanced_subset(X, y, num_per_class, seed=42):
    rng = np.random.default_rng(seed)

    zero_indices = np.where(y == 0)[0]
    one_indices = np.where(y == 1)[0]

    selected_zero = rng.choice(zero_indices, size=num_per_class, replace=False)
    selected_one = rng.choice(one_indices, size=num_per_class, replace=False)

    selected_indices = np.concatenate([selected_zero, selected_one])
    rng.shuffle(selected_indices)

    return X[selected_indices], y[selected_indices]

@torch.no_grad()
def evaluate(model, X, y, batch_size=64, device=None):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for x_batch, y_batch in get_image_batches(
        X,
        y,
        batch_size=batch_size,
        device=device
    ):

        logits, loss = model(x_batch, y_batch)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_correct += (preds == y_batch).sum().item()
        total_samples += y_batch.numel()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples

    return avg_loss, accuracy

## Utilities for training and evaluation of vision transformer models with workspace tokens
class FeedFoward(torch.nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, emb_dim, ffn_dim, layer = 'relu', dropout = 0.1):
        super().__init__()

        if layer == 'relu':
            l = torch.nn.ReLU()

        elif layer == 'gelu':
            l = torch.nn.GELU()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, ffn_dim),
            l,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(ffn_dim, emb_dim),
            torch.nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

### Saving results
def merge_and_overwrite(existing_path, new_df, key_cols):
    """
    Merge new_df with an existing CSV file.

    If existing_path exists:
        - load existing rows
        - remove rows whose key appears in new_df
        - append new_df
        - save merged result

    If existing_path does not exist:
        - save new_df directly
    """

    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)

        existing_keys = existing_df[key_cols].astype(str).agg("_".join, axis=1)
        new_keys = new_df[key_cols].astype(str).agg("_".join, axis=1)

        existing_df = existing_df[~existing_keys.isin(set(new_keys))]

        merged_df = pd.concat([existing_df, new_df], ignore_index=True)

    else:
        merged_df = new_df.copy()

    merged_df = merged_df.sort_values(key_cols).reset_index(drop=True)
    merged_df.to_csv(existing_path, index=False)

    return merged_df