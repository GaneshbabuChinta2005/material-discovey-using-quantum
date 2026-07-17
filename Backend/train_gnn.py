"""
Train GNN — Complete training pipeline for Graph Neural Network models.

Trains TWO models:
  1. Formation Energy (eV/atom)
  2. Band Gap (eV)

Pipeline:
  1. Load mp_dataset.csv
  2. Convert formulas → PyG graphs
  3. Split 80/10/10 (train/val/test)
  4. Train with AdamW + cosine annealing LR
  5. Early stopping on validation MAE
  6. Save best checkpoints as .pt files
  7. Report R², MAE, RMSE on test set

Usage:
    python train_gnn.py
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from torch_geometric.loader import DataLoader
from app.utils.graph_constructor import formula_to_graph
from app.models.gnn_model import MaterialGNN


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
CONFIG = {
    # Data
    "dataset_path": os.path.join(BASE_DIR, "Dataset", "real_materials.csv"),
    "model_dir": os.path.join(BASE_DIR, "model"),

    # Architecture
    "hidden_dim": 128,
    "n_conv_layers": 3,
    "dropout": 0.15,
    "neighbor_padding": 0.45,
    "structure_augmentations": 0,  # disabled for faster CPU processing
    "jitter_range": (0.02, 0.08),
    "density_scale_sigma": 0.05,

    # Training
    "batch_size": 64,
    "max_epochs": 200,
    "learning_rate": 1e-3,
    "min_lr": 1e-5,
    "weight_decay": 1e-4,
    "patience": 30,       # early stopping patience

    # Split
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
    "seed": 42,
}


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: str) -> pd.DataFrame:
    """Load and validate the materials dataset."""
    df = pd.read_csv(path)
    print(f"[Data] Loaded {len(df)} compounds from {os.path.basename(path)}")
    print(f"[Data] Columns: {list(df.columns)}")

    # Ensure required columns exist
    required = ["formula", "formation_energy_eV_per_atom", "band_gap_eV", "e_above_hull"]
    
    # Try finding an ehull column if not exactly named
    if "e_above_hull" not in df.columns:
        if "ehull" in df.columns:
            df["e_above_hull"] = df["ehull"]
        elif "energy_above_hull" in df.columns:
            df["e_above_hull"] = df["energy_above_hull"]

    for col in required:
        if col not in df.columns:
            print(f"[Data] WARNING: Missing column: {col}")

    # Drop rows with missing energy or bandgap
    before = len(df)
    df = df.dropna(subset=["formation_energy_eV_per_atom", "band_gap_eV"])
    if len(df) < before:
        print(f"[Data] Dropped {before - len(df)} rows with missing targets")

    return df


def prepare_graphs(df: pd.DataFrame, config: dict) -> list:
    """Convert all formulas to PyG graph objects with structural augmentation."""
    cache_path = os.path.join(config["model_dir"], "graphs_cache.pt")
    if os.path.exists(cache_path):
        print(f"[Graph] Loading pre-built graphs from cache ({cache_path})...")
        return torch.load(cache_path)

    print(f"[Graph] Converting {len(df)} formulas to graphs... (This may take a while)")
    t0 = time.time()

    graphs: list = []
    skipped = 0
    rng = np.random.default_rng(config.get("seed", 42))
    neighbor_padding = config.get("neighbor_padding", 0.45)
    n_aug = max(0, int(config.get("structure_augmentations", 0)))
    jitter_min, jitter_max = config.get("jitter_range", (0.0, 0.0))
    density_sigma = max(0.0, float(config.get("density_scale_sigma", 0.0)))

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx > 0 and idx % 1000 == 0:
            elapsed_m = (time.time() - t0) / 60
            print(f"  ... processed {idx}/{len(df)} formulas ({elapsed_m:.1f} minutes elapsed)")

        density_val = row.get("density_g_cm3")
        density = float(density_val) if pd.notna(density_val) else None
        
        ehull_val = row.get("e_above_hull")
        target_ehull = float(ehull_val) if pd.notna(ehull_val) else None

        base_graph = formula_to_graph(
            formula=row["formula"],
            target_energy=float(row["formation_energy_eV_per_atom"]),
            target_bandgap=float(row["band_gap_eV"]),
            target_ehull=target_ehull,
            density=density,
            neighbor_padding=neighbor_padding,
        )
        if base_graph is not None:
            graphs.append(base_graph)
        else:
            skipped += 1
            continue

        for _ in range(n_aug):
            jitter = 0.0
            if jitter_max > jitter_min:
                jitter = float(rng.uniform(jitter_min, jitter_max))
            density_aug = density
            if density_aug is not None and density_sigma > 0:
                scale = rng.normal(1.0, density_sigma)
                density_aug = max(0.1, density_aug * scale)
            aug_graph = formula_to_graph(
                formula=row["formula"],
                target_energy=float(row["formation_energy_eV_per_atom"]),
                target_bandgap=float(row["band_gap_eV"]),
                target_ehull=target_ehull,
                density=density_aug,
                jitter=jitter,
                neighbor_padding=neighbor_padding,
                rng=rng,
            )
            if aug_graph is not None:
                graphs.append(aug_graph)

    elapsed = time.time() - t0
    print(
        f"[Graph] Built {len(graphs)} graphs (+{n_aug} aug/compound, {skipped} skipped) in {elapsed:.1f}s"
    )
    
    # Save to cache so next time it runs instantly
    print(f"[Graph] Saving {len(graphs)} graphs to cache ({cache_path})...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(graphs, cache_path)
    
    return graphs


def split_dataset(
    graphs: list,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """Split graphs into train/val/test sets."""
    n = len(graphs)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_set = [graphs[i] for i in train_idx]
    val_set = [graphs[i] for i in val_idx]
    test_set = [graphs[i] for i in test_idx]

    print(f"[Split] Train: {len(train_set)} | Val: {len(val_set)} | "
          f"Test: {len(test_set)}")
    return train_set, val_set, test_set


def train_one_epoch(
    model: MaterialGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    target_key: str,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        target = getattr(batch, target_key).view(-1, 1)

        optimizer.zero_grad()
        pred = model(batch)
        loss = nn.functional.mse_loss(pred, target)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: MaterialGNN,
    loader: DataLoader,
    target_key: str,
    device: torch.device,
) -> dict:
    """Evaluate model, return MAE, RMSE, R² metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        batch = batch.to(device)
        target = getattr(batch, target_key).view(-1)
        pred = model(batch).view(-1)

        all_preds.append(pred.cpu())
        all_targets.append(target.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    # R² score
    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_model(
    model: MaterialGNN,
    train_set: list,
    val_set: list,
    test_set: list,
    target_key: str,
    target_name: str,
    save_path: str,
    config: dict,
    device: torch.device,
) -> dict:
    """
    Full training loop with early stopping and checkpoint/resume support.

    Saves a checkpoint every 10 epochs so training can be resumed if
    interrupted (e.g. laptop sleep, crash, etc.).

    Returns:
        Dict with best validation metrics and test metrics
    """
    checkpoint_path = save_path.replace(".pt", "_checkpoint.pt")

    print(f"\n{'='*60}")
    print(f"Training: {target_name}")
    print(f"{'='*60}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Target key: {target_key}")

    model = model.to(device)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"],
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=config["batch_size"])

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["max_epochs"],
        eta_min=config["min_lr"],
    )

    best_val_mae = float("inf")
    patience_counter = 0
    best_epoch = 0
    start_epoch = 1
    history = {"train_loss": [], "val_mae": [], "val_r2": []}

    # ── Resume from checkpoint if available ──
    if os.path.exists(checkpoint_path):
        print(f"\n  ⏩ Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mae = ckpt["best_val_mae"]
        best_epoch = ckpt["best_epoch"]
        patience_counter = ckpt["patience_counter"]
        history = ckpt.get("history", history)
        print(f"  ⏩ Resuming from epoch {start_epoch} "
              f"(best MAE={best_val_mae:.4f} at epoch {best_epoch})\n")

    for epoch in range(start_epoch, config["max_epochs"] + 1):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                      target_key, device)
        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, target_key, device)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_metrics["mae"])
        history["val_r2"].append(val_metrics["r2"])

        elapsed = time.time() - t0

        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == start_epoch:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{config['max_epochs']}  "
                  f"loss={train_loss:.4f}  "
                  f"val_MAE={val_metrics['mae']:.4f}  "
                  f"val_R²={val_metrics['r2']:.4f}  "
                  f"lr={lr:.2e}  "
                  f"({elapsed:.1f}s)")

        # Early stopping check
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "node_feat_dim": model.node_feat_dim,
                    "edge_feat_dim": model.edge_feat_dim,
                    "hidden_dim": model.hidden_dim,
                },
                "best_val_mae": best_val_mae,
                "epoch": epoch,
            }, save_path)
        else:
            patience_counter += 1

        # ── Save checkpoint every 10 epochs ──
        if epoch % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "history": history,
            }, checkpoint_path)

        if patience_counter >= config["patience"]:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(best: epoch {best_epoch}, MAE={best_val_mae:.4f})")
            break

    # Clean up checkpoint after training completes
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Cleaned up checkpoint file")

    # Load best model and evaluate on test set
    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, target_key, device)

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  {target_name:34s}  ║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  Best Epoch:  {best_epoch:4d}                    ║")
    print(f"  ║  Val  MAE:    {best_val_mae:.4f}                ║")
    print(f"  ║  Test MAE:    {test_metrics['mae']:.4f}                ║")
    print(f"  ║  Test RMSE:   {test_metrics['rmse']:.4f}                ║")
    print(f"  ║  Test R²:     {test_metrics['r2']:.4f}                ║")
    print(f"  ╚══════════════════════════════════════╝")

    return {
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }


def main():
    """Main training entrypoint."""
    print("\n" + "=" * 60)
    print(" GNN Training Pipeline — Material Property Prediction")
    print("=" * 60)

    set_seed(CONFIG["seed"])
    device = torch.device("cpu")  # CPU-only for this setup
    print(f"\nDevice: {device}")

    # 1. Load dataset
    df = load_dataset(CONFIG["dataset_path"])

    # 2. Convert to graphs
    graphs = prepare_graphs(df, CONFIG)
    if len(graphs) < 100:
        print(f"[ERROR] Too few valid graphs ({len(graphs)}). Aborting.")
        return

    # 3. Split
    train_set, val_set, test_set = split_dataset(
        graphs, CONFIG["train_ratio"], CONFIG["val_ratio"], CONFIG["seed"],
    )

    # 4. Train formation energy model
    energy_model = MaterialGNN(
        hidden_dim=CONFIG["hidden_dim"],
        n_conv_layers=CONFIG["n_conv_layers"],
        dropout=CONFIG["dropout"],
    )
    energy_save = os.path.join(CONFIG["model_dir"], "gnn_energy_model.pt")
    energy_results = train_model(
        model=energy_model,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        target_key="y_energy",
        target_name="Formation Energy (eV/atom)",
        save_path=energy_save,
        config=CONFIG,
        device=device,
    )

    # 5. Train band gap model
    bandgap_model = MaterialGNN(
        hidden_dim=CONFIG["hidden_dim"],
        n_conv_layers=CONFIG["n_conv_layers"],
        dropout=CONFIG["dropout"],
    )
    bandgap_save = os.path.join(CONFIG["model_dir"], "gnn_bandgap_model.pt")
    bandgap_results = train_model(
        model=bandgap_model,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        target_key="y_bandgap",
        target_name="Band Gap (eV)",
        save_path=bandgap_save,
        config=CONFIG,
        device=device,
    )

    # 6. Train Ehull (Stability) model
    ehull_model = MaterialGNN(
        hidden_dim=CONFIG["hidden_dim"],
        n_conv_layers=CONFIG["n_conv_layers"],
        dropout=CONFIG["dropout"],
    )
    ehull_save = os.path.join(CONFIG["model_dir"], "gnn_ehull_model.pt")
    
    # Filter datasets for graphs that have the y_ehull target
    train_ehull = [g for g in train_set if hasattr(g, 'y_ehull')]
    val_ehull = [g for g in val_set if hasattr(g, 'y_ehull')]
    test_ehull = [g for g in test_set if hasattr(g, 'y_ehull')]
    
    ehull_results = None
    if len(train_ehull) > 100:
        print(f"\n[Ehull Training] Valid graphs: Train {len(train_ehull)} | Val {len(val_ehull)} | Test {len(test_ehull)}")
        ehull_results = train_model(
            model=ehull_model,
            train_set=train_ehull,
            val_set=val_ehull,
            test_set=test_ehull,
            target_key="y_ehull",
            target_name="Energy Above Hull (eV/atom)",
            save_path=ehull_save,
            config=CONFIG,
            device=device,
        )
    else:
        print(f"\n[Ehull Training] Skipped: Not enough graphs with y_ehull target ({len(train_ehull)}).")

    # Summary
    print(f"\n{'='*60}")
    print(f" Training Complete!")
    print(f"{'='*60}")
    print(f"  Energy model saved to:  {energy_save}")
    print(f"  Bandgap model saved to: {bandgap_save}")
    if ehull_results:
        print(f"  Ehull model saved to:   {ehull_save}")

    print(f"\n  Energy  — Test MAE: {energy_results['test_mae']:.4f} eV/atom, "
          f"R²: {energy_results['test_r2']:.4f}")
    print(f"  Bandgap — Test MAE: {bandgap_results['test_mae']:.4f} eV, "
          f"R²: {bandgap_results['test_r2']:.4f}")
    if ehull_results:
        print(f"  Ehull   — Test MAE: {ehull_results['test_mae']:.4f} eV/atom, "
              f"R²: {ehull_results['test_r2']:.4f}")

    print(f"\nModels saved:")
    saved_files = [energy_save, bandgap_save]
    if ehull_results:
        saved_files.append(ehull_save)
        
    for f in saved_files:
        if os.path.exists(f):
            sz = os.path.getsize(f) / 1024
            print(f"  {os.path.basename(f)}: {sz:.0f} KB")


if __name__ == "__main__":
    main()
