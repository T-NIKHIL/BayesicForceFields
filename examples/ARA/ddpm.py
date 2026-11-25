"""Supervised VAE and independent, comparable W&B grid sweeps per data fold."""
from __future__ import annotations

import argparse
import json
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import re
import copy
import json5

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dataset_builder import TensorDataset, build_dataset, analyze_dataset, extract_qoi
import utils
from plot_umap import _plot_umap

ROOT = Path(__file__).resolve().parent

# Name for the project
PROJECT = 'peptide_supervised_vae'
RDF_TARGETS = ['C', 'CT1', 'H', 'HB1', 'NH1', 'O'] # trial_1 missing 'C'
HB_TARGETS = ['PROA(NH1) to SOL(OW4)', 'SOL(OW4) to PROA(NH1)', 'SOL(OW4) to PROA(O)']
RDF_DATASET_PARAMS = {
    'name': 'rdf',
    'reference_path': str(ROOT / "02-reference/rdf"),
    'qoi_path': str(ROOT / "04-qoi/rdf"),
    'samples_yaml_path': str(ROOT / "03-sample/samples.yaml"),
    'targets': RDF_TARGETS,
    'nuisance': [0.002] * len(RDF_TARGETS), # trial_1 used 0.002
}
HB_DATASET_PARAMS = {
    'name': 'hb',
    'reference_path': str(ROOT / "02-reference/hb"),
    'qoi_path': str(ROOT / "04-qoi/hb"),
    'samples_yaml_path': str(ROOT / "03-sample/samples.yaml"),
    'targets': HB_TARGETS,
    'nuisance': [0.002] * len(HB_TARGETS), # trial_1 used 0.002
}
MODULE_PARAMS = {
    "hidden_dim": 256,
    "hidden_layers": 3,
    "time_embedding_dim": 64, # A sinsusoidal embedding of 64 uses 32 sine/cosine pairs
    "condition_dim": 1,       # Summed RDF/HB GNLL
}
# Noise schedule
DIFFUSION_PARAMS = {
    "timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
}
TRAIN_PARAMS = {
    'seed': 42,
    'num_epochs': 200,
    'lr': 0.001,
    'batch_size': 128,
    'accelerator': 'cpu',
    'weight_decay': 0.0,
    'val_frac':0.1,
    'test_frac': 0.1,
    'gradient_clip':1.0,
}
OUTPUT_ROOT = ROOT / "ddpm_runs"
DEVICE = 'cpu'
SCRIPT_MODE = 'create_dataset' # Suppported extract_qoi, create_dataset, train_DDPM, eval_DDPM, plot_UMAP

UMAP_PARAMS = {
    "n_components": 3,
    "n_neighbors": 10,
    "min_dist": 0.3,
    "metric": "euclidean",
    "random_state": 42,
    "n_jobs": 1,
    "spread": 1.0,
}
UMAP_OUTPUT = ROOT / "umap"

class SinusoidalTimeEmbedding(nn.Module):
    """Convert each integer diffusion timestep into a feature vector."""

    def __init__(self, embedding_dim):
        super().__init__()

        if embedding_dim < 4 or embedding_dim % 2:
            raise ValueError("Time embedding dimension must be even and >= 4")

        half_dim = embedding_dim // 2
        pair_indices = torch.arange(half_dim, dtype=torch.float32)
        frequencies = 1.0 / (10000.0 ** (pair_indices / half_dim))

        # Fixed frequencies: move with the model, but are not optimized.
        self.register_buffer("frequencies", frequencies)

    def forward(self, timesteps):
        # timesteps: (batch_size,)
        angles = timesteps.float()[:, None] * self.frequencies[None, :]

        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)

class ConditionalDenoiser(nn.Module):
    """Predict parameter noise conditioned on timestep and summed GNLL."""

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        hidden_layers=3,
        time_embedding_dim=32,
        condition_dim=1,
    ):
        super().__init__()

        if min(input_dim, hidden_dim, hidden_layers, condition_dim) < 1:
            raise ValueError("Model dimensions and layer count must be positive")

        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "hidden_layers": hidden_layers,
            "time_embedding_dim": time_embedding_dim,
            "condition_dim": condition_dim,
        }

        self.time_embedding = SinusoidalTimeEmbedding(
            time_embedding_dim
        )

        width = input_dim + time_embedding_dim + condition_dim
        layers = []

        for _ in range(hidden_layers):
            layers.extend([
                nn.Linear(width, hidden_dim),
                nn.SiLU(),
            ])
            width = hidden_dim

        # Noise can have any sign or magnitude: no output activation.
        layers.append(nn.Linear(width, input_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, noisy_params, timesteps, condition):
        batch_size = noisy_params.shape[0]

        if noisy_params.shape != (
            batch_size, self.config["input_dim"]
        ):
            raise ValueError("Unexpected noisy parameter shape")
        if timesteps.shape != (batch_size,):
            raise ValueError("Expected one timestep per sample")
        if condition.shape != (
            batch_size, self.config["condition_dim"]
        ):
            raise ValueError("Unexpected condition shape")

        time_features = self.time_embedding(timesteps).to(
            dtype=noisy_params.dtype
        )

        features = torch.cat(
            [noisy_params, time_features, condition],
            dim=-1,
        )
        return self.network(features)

class GaussianDiffusion(nn.Module):
    """Apply the forward DDPM process to standardized parameter vectors."""

    def __init__(self, timesteps, beta_start, beta_end):
        super().__init__()

        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        if not 0 < beta_start <= beta_end < 1:
            raise ValueError("Require 0 < beta_start <= beta_end < 1")

        self.timesteps = timesteps

        betas = torch.linspace(
            beta_start, beta_end, timesteps, dtype=torch.float32
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, clean_params, timesteps, noise):
        """Return a noisy version of each row at its assigned timestep."""
        # (batch_size, 1) broadcasts across the five parameter columns.
        alpha_bar = self.alpha_bars[timesteps].unsqueeze(-1)

        return (
            alpha_bar.sqrt() * clean_params
            + (1.0 - alpha_bar).sqrt() * noise
        )

    def p_sample(self, model, noisy_params, step, condition, generator=None):
        """Take one reverse DDPM step for every row in the batch."""
        timesteps = torch.full(
            (noisy_params.shape[0],),
            step,
            device=noisy_params.device,
            dtype=torch.long,
        )
        predicted_noise = model(noisy_params, timesteps, condition)

        beta = self.betas[step]
        alpha = self.alphas[step]
        alpha_bar = self.alpha_bars[step]

        mean = (
            noisy_params
            - beta / (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha.sqrt()

        # Step 0 returns the final estimate without another noise draw.
        if step == 0:
            return mean

        previous_alpha_bar = self.alpha_bars[step - 1]
        variance = beta * (1.0 - previous_alpha_bar) / (1.0 - alpha_bar)

        noise = torch.randn(
            noisy_params.shape,
            device=noisy_params.device,
            dtype=noisy_params.dtype,
            generator=generator,
        )
        return mean + variance.sqrt() * noise

    @torch.no_grad()
    def sample(self, model, condition, generator=None):
        """Generate one standardized parameter vector per condition row."""
        noisy_params = torch.randn(
            (condition.shape[0], model.config["input_dim"]),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )

        for step in range(self.timesteps - 1, -1, -1):
            noisy_params = self.p_sample(
                model, noisy_params, step, condition, generator
            )

        return noisy_params

def epoch_pass(model,
               diffusion,
               loader,
               optimizer=None,
               train_rng=None,
               gradient_clip=1.0,
               eval_seed=43,
               log_every=None
):
    """Run one epoch and return the mean noise-prediction MSE."""
    device = next(model.parameters()).device
    training = optimizer is not None
    model.train(training)

    rng = train_rng
    if not training:
        # Reset on each evaluation so the same samples get the same noise.
        rng = torch.Generator(device=device).manual_seed(eval_seed)

    total_loss = 0.0
    sample_count = 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch_idx, batch in enumerate(loader, start=1):

            clean_params = batch["params"].to(device)
            condition = batch["total_gnll"].to(device)
            batch_size = len(clean_params)

            if training:
                optimizer.zero_grad(set_to_none=True)

            timesteps = torch.randint(
                diffusion.timesteps,
                (batch_size,),
                device=device,
                generator=rng)
            noise = torch.randn(
                clean_params.shape,
                dtype=clean_params.dtype,
                device=device,
                generator=rng,
            )

            # One epoch comprises of :
                # (B, D) samples thorugh the q_sample function. B is the batch size and D the param dimensions.
                # Output is noised params at different timesteps. The denoiser learns how far a sample is from normal noise.
            noisy_params = diffusion.q_sample(clean_params, timesteps, noise)
            predicted_noise = model(noisy_params, timesteps, condition)
            # We compute the loss wrt noise sampled from a Normal distribution.
            loss = F.mse_loss(predicted_noise, noise)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite DDPM noise loss")

            if training:
                loss.backward()
                if gradient_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * batch_size
            sample_count += batch_size

            if log_every and (
                batch_idx % log_every == 0 or batch_idx == len(loader)
            ):
                phase = "train" if training else "eval"
                print(
                    f"{phase} | batch {batch_idx}/{len(loader)} | "
                    f"batch MSE {batch_loss:.6f} | "
                    f"running MSE {total_loss / sample_count:.6f}",
                    flush=True,
                )

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty loader")

    return total_loss / sample_count

def train_val_loop(model, diffusion, loaders, metadata, train_params, run_dir):
    if train_params["num_epochs"] < 1:
        raise ValueError("num_epochs must be positive")

    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "best.pt"

    device = next(model.parameters()).device
    seed = train_params["seed"]
    train_rng = torch.Generator(device=device).manual_seed(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
    )

    plateau_params = train_params.get("lr_plateau")
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **plateau_params)
        if plateau_params is not None else None
    )
    best_val = float("inf")

    with (run_dir / 'epochs.jsonl').open('w', encoding='utf-8') as history:
        for epoch in range(1, train_params['num_epochs'] + 1):
            train_loss = epoch_pass(model,
                                    diffusion,
                                    loaders["train"],
                                    optimizer=optimizer,
                                    train_rng=train_rng,
                                    gradient_clip=train_params["gradient_clip"],
                                    log_every=10
                                   )
            val_loss = epoch_pass(
                model, diffusion, loaders["val"], eval_seed=seed + 1
            )

            metrics = {
                "epoch": epoch,
                "train_noise_mse": train_loss,
                "val_noise_mse": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.write(json.dumps(metrics) + "\n")
            history.flush()

            print(
                f"Epoch {epoch}/{train_params['num_epochs']} | "
                f"train MSE {train_loss:.6f} | val MSE {val_loss:.6f}",
                flush=True,
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_config": model.config,
                        "model_state_dict": model.state_dict(),
                        "diffusion_config": dict(DIFFUSION_PARAMS),
                        "metadata": metadata,
                        "train_params": dict(train_params),
                        **metrics,
                    },
                    checkpoint_path,
                )

            if scheduler is not None:
                scheduler.step(val_loss)
                next_lr = optimizer.param_groups[0]["lr"]
                if next_lr < metrics["lr"]:
                    print(f"Epoch {epoch} | next learning rate {next_lr:.3g}", flush=True)

    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"])

    if not train_params.get("evaluate_test", True):
        print(
            f"Best epoch {best['epoch']} | val MSE {best['val_noise_mse']:.6f}",
            flush=True,
        )
        return checkpoint_path

    test_loss = epoch_pass(
        model, diffusion, loaders["test"], eval_seed=seed + 2
    )
    print(
        f"Best epoch {best['epoch']} | "
        f"val MSE {best['val_noise_mse']:.6f} | test MSE {test_loss:.6f}",
        flush=True,
    )

    return checkpoint_path

def main(run_dir, batch_size=None):
    train_params = dict(TRAIN_PARAMS)
    if batch_size is not None:
        train_params["batch_size"] = batch_size

    # rdf_dataset_params, hb_dataset_params, module_params, train_params = copy.deepcopy(RDF_DATASET_PARAMS), copy.deepcopy(HB_DATASET_PARAMS), copy.deepcopy(MODULE_PARAMS), copy.deepcopy(TRAIN_PARAMS)
    # # Create one dataset by adding the gaussian negative log likelihoods (See likelihoods.py)
    # # tensor_dataset = build_dataset([rdf_dataset_params, hb_dataset_params])
    combined_tensor_dictionary = torch.load(ROOT / "datasets/combined_tensor_dataset_dictionary.pt")
    combined_tensor_dataset = TensorDataset('rdf_hb_dataset', combined_tensor_dictionary)

    params = combined_tensor_dictionary['params']
    total_gnll = combined_tensor_dictionary['total_gnll']

    if (
            params.ndim != 2
            or total_gnll.shape != (len(params), 1)
            or not torch.isfinite(params).all()
            or not torch.isfinite(total_gnll).all()
        ):
            raise ValueError("Expected finite params (N, D) and total_gnll (N, 1)")

    seed = train_params["seed"]
    val_frac = train_params["val_frac"]
    test_frac = train_params["test_frac"]
    batch_size = train_params["batch_size"]

    if not (
        0 < val_frac < 1
        and 0 < test_frac < 1
        and val_frac + test_frac < 1
    ):
        raise ValueError("Invalid validation/test fractions")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    indices = np.arange(len(params))

    development_indices, test_indices = train_test_split(
        indices,
        test_size=test_frac,
        random_state=seed,
    )
    train_indices, val_indices = train_test_split(
        development_indices,
        # val_frac refers to the complete dataset.
        test_size=val_frac / (1.0 - test_frac),
        random_state=seed,
    )

    def fit_normalization(values):
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)
        # Constant columns remain finite after normalization.
        std = torch.where(std == 0, torch.ones_like(std), std)
        return mean, std

    params_mean, params_std = fit_normalization(params[train_indices])
    total_gnll_mean, total_gnll_std = fit_normalization(
        total_gnll[train_indices]
    )

    normalized_params = (params - params_mean) / params_std
    normalized_total_gnll = (total_gnll - total_gnll_mean) / total_gnll_std

    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    loaders = {}
    for name, split_indices in splits.items():
        split_dataset = TensorDataset(
            name = f"{combined_tensor_dataset.name}--{name}",
            tensor_dictionary={
                "params":normalized_params[split_indices],
                "total_gnll":normalized_total_gnll[split_indices],
            },
            sample_ids=[
                combined_tensor_dataset.sample_ids[int(index)]
                for index in split_indices
            ],
        )
        loaders[name] = DataLoader(
            split_dataset,
            batch_size=batch_size if name == "train" else len(split_dataset),
            shuffle=(name == "train"),
            generator=torch.Generator().manual_seed(seed),
        )

    metadata = {
        "input_dim": params.shape[1],
        "x_mean": params_mean,
        "x_std": params_std,
        "total_gnll_mean": total_gnll_mean,
        "total_gnll_std": total_gnll_std,
        "train_condition_min": total_gnll[train_indices].min().item(),
        "train_condition_max": total_gnll[train_indices].max().item(),
        "sample_ids": list(combined_tensor_dataset.sample_ids),
        "splits": {
            name: split_indices.tolist()
            for name, split_indices in splits.items()
        },
    }

    torch.manual_seed(seed)
    device = torch.device(train_params["accelerator"])

    model = ConditionalDenoiser(
        input_dim=metadata["input_dim"], **MODULE_PARAMS
    ).to(device)
    diffusion = GaussianDiffusion(**DIFFUSION_PARAMS).to(device)

    checkpoint_path = train_val_loop(
        model, diffusion, loaders, metadata, train_params, run_dir
    )
    print(f"Saved best checkpoint to {checkpoint_path}", flush=True)

def plot_mse_loss(run_dir):
    import matplotlib.pyplot as plt

    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    settings = checkpoint["train_params"]
    timesteps = checkpoint["diffusion_config"]["timesteps"]
    embedding_dim = checkpoint["model_config"]["time_embedding_dim"]
    epoch_pattern = re.compile(
        r"Epoch (\d+)/\d+ \| train MSE (\S+) \| val MSE (\S+)"
    )

    epochs, train_mse, val_mse = [], [], []
    with (run_dir / "run.log").open(encoding="utf-8") as log_file:
        for line in log_file:
            match = epoch_pattern.fullmatch(line.strip())
            if match:
                epoch, train_loss, val_loss = match.groups()
                epochs.append(int(epoch))
                train_mse.append(float(train_loss))
                val_mse.append(float(val_loss))

    if not epochs:
        raise ValueError(f"No epoch summaries found in {run_dir / 'run.log'}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    train_line, = ax.plot(
        epochs, train_mse, label=f"Training (final: {train_mse[-1]:.6f})"
    )
    val_line, = ax.plot(
        epochs, val_mse, label=f"Validation (final: {val_mse[-1]:.6f})"
    )

    ax.scatter(epochs[-1], train_mse[-1], color=train_line.get_color(), zorder=3)
    ax.scatter(epochs[-1], val_mse[-1], color=val_line.get_color(), zorder=3)
    ax.set(xlabel="Epoch", ylabel="Noise-prediction MSE")
    ax.set_title(
        "DDPM training and validation loss\n"
        f"Batch size: {settings['batch_size']}  |  LR: {settings['lr']:.0E}  |  "
        f"{timesteps:,} timesteps  |  Time embedding: {embedding_dim}D"
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "mse_loss.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":

    if SCRIPT_MODE == 'extract_qoi':
        # Run only once in start to extract data and create file structure
        qoi_files = [Path('qoi-rdf.pt'), Path('qoi-hb.pt')]
        for qoi_file in qoi_files:
            with qoi_file.open(encoding="utf-8") as f:
                qoi_json = json5.load(f)
            extract_qoi(qoi_json, '.')

    elif SCRIPT_MODE == 'create_dataset':
        rdf_dataset_params, hb_dataset_params, module_params, train_params = copy.deepcopy(RDF_DATASET_PARAMS), copy.deepcopy(HB_DATASET_PARAMS), copy.deepcopy(MODULE_PARAMS), copy.deepcopy(TRAIN_PARAMS)
        # Create one dataset by adding the gaussian negative log likelihoods (See likelihoods.py)
        # tensor_dataset = build_dataset([rdf_dataset_params, hb_dataset_params])
        tensor_dataset = build_dataset([rdf_dataset_params, hb_dataset_params],
                                        fname='combined_tensor_dataset_breakdown',
                                        root=ROOT)

    elif SCRIPT_MODE == 'analyze_dataset':
        # analyze_dataset(ROOT / "datasets" / "combined_tensor_dataset_breakdown_dictionary.pt")

        import numpy as np
        import torch
        from bff.bayes.effective_observations import estimate_curve_n_eff

        state = torch.load(
            f"{ROOT}/05-fit/models/rdf.lgp", map_location="cpu", weights_only=True
        )
        curves = np.asarray(state["reference_values"]).reshape(state["n_curves"], -1)

        for tolerance in (0.005, 0.01, 0.02, 0.05):
            per_curve = [
                estimate_curve_n_eff(curve, tolerance=tolerance)
                for curve in curves
            ]
            print(
                f"Tolerance {tolerance:g} | "
                f"Per curve: {np.round(per_curve, 2)} | "
                f"Total: {sum(per_curve):.2f}"
            )

    elif SCRIPT_MODE == 'plot_UMAP':
        combined_tensor_dictionary = torch.load(ROOT / "datasets/combined_tensor_dataset_dictionary.pt")
        combined_tensor_dataset = TensorDataset('rdf_hb_dataset', combined_tensor_dictionary)
        UMAP_OUTPUT.mkdir(parents=True, exist_ok=True)
        coords = _plot_umap(vectors=combined_tensor_dictionary["params"].numpy(),
                            gnll=combined_tensor_dictionary["total_gnll"].squeeze(1).numpy(),
                            umap_params=UMAP_PARAMS,
                            output_path=UMAP_OUTPUT /  'training_data_umap.html',
                            color_label="Total GNLL",
                            show=False,
                        )

    elif SCRIPT_MODE == 'train_DDPM':
        parser = argparse.ArgumentParser(description="Train the DDPM or plot an existing run")
        modes = parser.add_mutually_exclusive_group()
        modes.add_argument("--batch-size", type=int, default=TRAIN_PARAMS["batch_size"])
        modes.add_argument("--plot-run", type=Path)
        args = parser.parse_args()

        run_dir = args.plot_run
        if run_dir is None:
            if args.batch_size < 1:
                parser.error("--batch-size must be positive")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_dir = OUTPUT_ROOT / f"{stamp}-bs{args.batch_size}"
            run_dir.mkdir(parents=True, exist_ok=False)
            # Save the module and train params as a .json in the run directory
            utils.write_json(run_dir / 'settings.json',
                            {'rdf_targets': RDF_TARGETS,
                             'rdf_dataset_params': RDF_DATASET_PARAMS,
                             'hb_targets': HB_TARGETS,
                             'hb_dataset_params': HB_DATASET_PARAMS,
                             'module_params': MODULE_PARAMS,
                             'diffusion_params': DIFFUSION_PARAMS,
                             'train_params': TRAIN_PARAMS})
            with (run_dir / "run.log").open(
                "w", encoding="utf-8", buffering=1
            ) as log_file, redirect_stdout(log_file):
                main(run_dir, batch_size=args.batch_size)

        plot_mse_loss(run_dir)
        print(f"Saved loss plot to {run_dir / 'mse_loss.png'}")

    elif SCRIPT_MODE == 'eval_DDPM':
        final_model_state_dict = torch.load('zabka/ddpm_runs/final/best.pt')['model_state_dict']
        final_model_config = torch.load('zabka/ddpm_runs/final/best.pt')['model_config']
        final_model = ConditionalDenoiser(input_dim=final_model_config['input_dim'],
                                          hidden_dim=final_model_config['hidden_dim'],
                                          hidden_layers=final_model_config['hidden_layers'],
                                          time_embedding_dim=final_model_config['time_embedding_dim'],
                                          condition_dim=final_model_config['condition_dim'])
        final_model.load_state_dict(final_model_state_dict)

        # print(torch.load('zabka/ddpm_runs/final/best.pt')['metadata'])

        gaussian_diffusion_config = torch.load('zabka/ddpm_runs/final/best.pt')['diffusion_config']
        gaussian_diffusion = GaussianDiffusion(timesteps=gaussian_diffusion_config['timesteps'],
                                                beta_start=gaussian_diffusion_config['beta_start'],
                                                beta_end=gaussian_diffusion_config['beta_end'])

        # gaussian_diffusion.p_sample()

        # print(final_model)
