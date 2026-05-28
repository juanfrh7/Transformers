import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import sys
import pandas as pd
import json

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.vision_transformer import VisionTransformer
from models.utils import (
    set_seed,
    evaluate,
    get_image_batches,
    make_balanced_subset,
    patchify_images,
    merge_and_overwrite,
)


if __name__ == "__main__":

    # -------------------------
    # Paths
    # -------------------------

    project_root = Path(__file__).resolve().parents[3]

    config_path = (
        project_root
        / "src"
        / "experiments"
        / "equilateral_triangles"
        / "local_results"
        / "config_1.json"
    )

    data_path = (
        project_root
        / "data"
        / "equilateral_triangles"
    )

    output_path = (
        project_root
        / "src"
        / "experiments"
        / "equilateral_triangles"
        / "local_results"
    )

    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load dataset
    # -------------------------

    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")

    X_val = np.load(data_path / "X_val.npy")
    y_val = np.load(data_path / "y_val.npy")

    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")

    # -------------------------
    # Load config
    # -------------------------

    with open(config_path, "r") as f:
        config = json.load(f)

    seeds = config["seeds"]
    experiment_num = config["experiment_num"]

    raw_results = []
    training_history = []

    print(
        f"Running ViT baseline with "
        f"emb_dim: {config['emb_dim']}, "
        f"num_heads: {config['num_heads']}, "
        f"num_layers: {config['num_layers']}, "
        f"head_size_k: {config['head_size_k']}, "
        f"head_size_v: {config['head_size_v']}, "
        f"num_patches: {config['num_patches']}, "
        f"patch_size: {config['patch_size']}x{config['patch_size']}, "
        f"num_train_per_class: {config['num_train_per_class']}, "
        f"num_val_per_class: {config['num_val_per_class']}, "
        f"num_test_per_class: {config['num_test_per_class']}"
    )

    # -------------------------
    # Main experiment loop
    # -------------------------

    for seed in seeds:

        set_seed(seed)

        reduced_x_train, reduced_y_train = make_balanced_subset(
            X_train,
            y_train,
            num_per_class=config["num_train_per_class"],
            seed=seed,
        )

        reduced_x_val, reduced_y_val = make_balanced_subset(
            X_val,
            y_val,
            num_per_class=config["num_val_per_class"],
            seed=seed + 100,
        )

        reduced_x_test, reduced_y_test = make_balanced_subset(
            X_test,
            y_test,
            num_per_class=config["num_test_per_class"],
            seed=seed + 200,
        )

        patched_train = patchify_images(
            reduced_x_train,
            num_patches=config["num_patches"],
        )

        patched_val = patchify_images(
            reduced_x_val,
            num_patches=config["num_patches"],
        )

        patched_test = patchify_images(
            reduced_x_test,
            num_patches=config["num_patches"],
        )

        set_seed(seed)

        model = VisionTransformer(
            emb_dim=config["emb_dim"],
            ffn_dim=config["ffn_dim"],
            patch_size=config["patch_size"],
            num_patches=config["num_patches"],
            num_heads=config["num_heads"],
            head_size_k=config["head_size_k"],
            head_size_v=config["head_size_v"],
            dropout=config["dropout"],
            num_layers=config["num_layers"],
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
        )

        for epoch in tqdm(
            range(config["epochs"]),
            desc=f"Seed {seed}",
        ):

            model.train()

            for x_batch, y_batch in get_image_batches(
                patched_train,
                reduced_y_train,
                batch_size=config["batch_size"],
            ):
                logits, loss = model(x_batch, y_batch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            scheduler.step()

            train_loss, train_acc = evaluate(
                model,
                patched_train,
                reduced_y_train,
                batch_size=config["batch_size"],
            )

            val_loss, val_acc = evaluate(
                model,
                patched_val,
                reduced_y_val,
                batch_size=config["batch_size"],
            )

            training_history.append(
                {
                    "model": "ViT",
                    "seed": seed,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

        test_loss, test_acc = evaluate(
            model,
            patched_test,
            reduced_y_test,
            batch_size=config["batch_size"],
        )

        raw_results.append(
            {
                "model": "ViT",
                "seed": seed,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "final_train_loss": train_loss,
                "final_train_acc": train_acc,
                "final_val_loss": val_loss,
                "final_val_acc": val_acc,
            }
        )

    # -------------------------
    # Convert new results to DataFrames
    # -------------------------

    new_raw_df = pd.DataFrame(raw_results)
    new_history_df = pd.DataFrame(training_history)

    raw_path = output_path / f"vit_raw_results_{experiment_num}.csv"
    history_path = output_path / f"vit_training_history_{experiment_num}.csv"
    summary_path = output_path / f"vit_summary_results_{experiment_num}.csv"

    # -------------------------
    # Merge with existing files or create new files
    # -------------------------

    merged_raw_df = merge_and_overwrite(
        existing_path=raw_path,
        new_df=new_raw_df,
        key_cols=["model", "seed"],
    )

    merged_history_df = merge_and_overwrite(
        existing_path=history_path,
        new_df=new_history_df,
        key_cols=["model", "seed", "epoch"],
    )

    # -------------------------
    # Recompute summary from merged raw results
    # -------------------------

    summary_df = (
        merged_raw_df
        .groupby(["model"], dropna=False)
        .agg(
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_loss=("test_loss", "mean"),
            std_test_loss=("test_loss", "std"),
            mean_val_acc=("final_val_acc", "mean"),
            std_val_acc=("final_val_acc", "std"),
            mean_val_loss=("final_val_loss", "mean"),
            std_val_loss=("final_val_loss", "std"),
            num_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["model"])
    )

    summary_df.to_csv(summary_path, index=False)

    print(f"Saved raw results to: {raw_path}")
    print(f"Saved training history to: {history_path}")
    print(f"Saved summary results to: {summary_path}")