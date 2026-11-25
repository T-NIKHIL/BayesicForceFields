import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn import functional as F
import yaml
import numpy as np
import json5

from utils import resolve_path

TARGETS = ['C1_C2_C3_O31', 'C21_O21_C2_C3', 'C31_O31_C3_C2', 'O11_C1_C2_O21']
DATASET_PARAMS = {
    'name': 'lipid_dihedral_angles', 'reference_path': '02-reference',
    'qoi_path': '04-qoi/qoi', 'samples_yaml_path': '03-sample/samples.yaml',
    'test_frac': 0.1, 'k_folds': 5, 'targets': TARGETS, 'nuisance': [0.002] * 4,
}

class TensorDataset(Dataset):
    """Dictionary batches; retain the original template's dataset interface."""

    def __init__(self, name, tensor_dictionary, desc_preprocessors=None, sample_ids=None):
        self.name = name
        self.tensor_dictionary = tensor_dictionary
        self.desc_preprocessors = desc_preprocessors or {}
        self.sample_ids = sample_ids or [str(i) for i in range(len(next(iter(tensor_dictionary.values()))))]
        sizes = {value.shape[0] for value in tensor_dictionary.values()}
        if len(sizes) != 1 or any(value.ndim != 2 for value in tensor_dictionary.values()):
            raise ValueError('Dataset tensors must be two-dimensional with matching sample counts')
        if len(self.sample_ids) != len(self):
            raise ValueError('Dataset sample IDs must match tensor rows')
        if any(not torch.isfinite(value).all() for value in tensor_dictionary.values()):
            raise ValueError('Dataset contains non-finite values')

    def __len__(self):
        return len(self.tensor_dictionary['params'])

    def __getitem__(self, idx):
        return {name: value[idx] for name, value in self.tensor_dictionary.items()}

def build_dataset(params, fname, root):
    """Match sampled parameters to KDEs by ID and construct scalar GNLL targets."""

    # In case if a list of param configs is passed
    if isinstance(params, (list, tuple)):
        tensor_datasets = [build_dataset(config, fname, root) for config in params]
        first_dataset = tensor_datasets[0]

        # Ensure RDF and HB describe the same samples and parameters.
        for dataset in tensor_datasets[1:]:
            if (
                dataset.sample_ids != first_dataset.sample_ids
                or not torch.equal(
                    dataset.tensor_dictionary["params"],
                    first_dataset.tensor_dictionary["params"],
                )
            ):
                raise ValueError("QOI datasets have different samples or parameters")

        # Each existing target already contains one scalar GNLL per sample.
        scores = {
            f"{tensor_dataset.name}/{label}": values
            for tensor_dataset in tensor_datasets
            for label, values in tensor_dataset.tensor_dictionary.items()
            if label != "params"
        }

        tensor_dictionary = scores.copy()
        tensor_dictionary['params'] = first_dataset.tensor_dictionary["params"]
        tensor_dictionary['total_gnll'] = torch.stack(list(scores.values()), dim=0).sum(dim=0)

        combined_tensor_dataset = TensorDataset(
            name="+".join(tensor_dataset.name for tensor_dataset in tensor_datasets),
            tensor_dictionary=tensor_dictionary,
            sample_ids=first_dataset.sample_ids,
        )

        # Replace the intermediate single-QOI cache with the combined data.
        torch.save(
            combined_tensor_dataset.tensor_dictionary,
            root / "datasets" / f"{fname}_dictionary.pt",
        )
        return combined_tensor_dataset

    targets = params['targets']
    if len(targets) != len(params['nuisance']) or any(v <= 0 for v in params['nuisance']):
        raise ValueError('nuisance must contain one positive variance per target')
    samples_path = resolve_path(params['samples_yaml_path'])
    with samples_path.open(encoding='utf-8') as stream:
        samples = yaml.safe_load(stream)['samples']
    references = {}
    for target in targets:
        path = resolve_path(params['reference_path']) / f'{target}.dat'
        values = np.loadtxt(path)
        if values.ndim == 2:
            if values.shape[1] != 2:
                    raise ValueError(f"Expected two columns at {path}")
            density = values[:, 1]
        elif values.ndim == 1 or values.ndim == 0:
            density = values
        else:
            raise ValueError(f'{target}: Unexpected shape {values.shape} at {path}')
        references[target] = torch.as_tensor(density, dtype=torch.float32)
    rows, ids = [], []
    scores = {target: [] for target in targets}
    directories = sorted(resolve_path(params['qoi_path']).glob('sample-000-*'))
    for directory in directories:
        if not directory.is_dir():
            continue
        sample_id = directory.name.rsplit('-', 1)[-1]
        if sample_id not in samples:
            raise ValueError(f'{directory}: sample ID {sample_id!r} missing from {samples_path}')
        rows.append(samples[sample_id]['params'])
        ids.append(sample_id)
        for target, variance in zip(targets, params['nuisance']):
            path = directory / f'{target}.dat'
            values = np.loadtxt(path)
            if values.ndim == 2:
                if values.shape[1] != 2:
                        raise ValueError(f"Expected two columns at {path}")
                density = values[:, 1]
            elif values.ndim == 1 or values.ndim == 0:
                density = values
            else:
                raise ValueError(f'{sample_id}/{target}: Unexpected shape {values.shape} at {path}')
            density = torch.as_tensor(density, dtype=torch.float32)
            # Computing the gaussian negative log likelihood between the AIMD reference and MD Observable.
            if density.shape != references[target].shape:
                raise ValueError(
                    f"{sample_id}/{target}: sample shape {tuple(density.shape)} "
                    f"!= reference shape {tuple(references[target].shape)}"
                )
            score = F.gaussian_nll_loss(density, references[target], torch.full_like(density, variance))
            scores[target].append(score)
    if not rows:
        raise ValueError(f'No sample directories found in {resolve_path(params["qoi_path"])}')
    tensors = {'params': torch.tensor(rows, dtype=torch.float32)}
    tensors.update({target: torch.stack(values).reshape(-1, 1) for target, values in scores.items()})
    tensor_dataset = TensorDataset(params['name'], tensors, sample_ids=ids)
    if Path(root / 'datasets').exists():
        torch.save(tensor_dataset.tensor_dictionary, root / "datasets" / f"{fname}_dictionary.pt")
    else:
        Path(root / 'datasets').mkdir(parents=True, exist_ok=True)
        torch.save(tensor_dataset.tensor_dictionary, root / "datasets" / f"{fname}_dictionary.pt")
    return tensor_dataset

def analyze_dataset(dataset_path, output_dir=None):
    """Summarize stored GNLL contributions and identify outlier samples."""
    import pandas as pd

    dataset_path = Path(dataset_path).resolve()
    project_root = dataset_path.parent.parent
    output_dir = (
        Path(output_dir) if output_dir is not None
        else dataset_path.parent / "gnll_analysis"
    )

    data = torch.load(dataset_path, map_location="cpu", weights_only=True)
    components = [
        key for key in data if key.startswith(("rdf/", "hb/"))
    ]
    if not components:
        raise ValueError("Expected a dataset with individual RDF and HB GNLLs.")

    torch.testing.assert_close(
        torch.stack([data[key] for key in components]).sum(dim=0),
        data["total_gnll"],
    )

    # The cache stores tensors only. Recover IDs in the builder's sort order.
    sample_dirs = {
        group: sorted(
            path.name
            for path in (project_root / "04-qoi" / group).glob("sample-000-*")
            if path.is_dir()
        )
        for group in ("rdf", "hb")
    }
    if not sample_dirs["rdf"] or sample_dirs["rdf"] != sample_dirs["hb"]:
        raise ValueError("RDF and HB sample directories must have matching IDs.")

    sample_names = sample_dirs["rdf"]
    with (project_root / "03-sample" / "samples.yaml").open(encoding="utf-8") as f:
        samples = yaml.safe_load(f)["samples"]

    expected_params = torch.tensor(
        [samples[name.rsplit("-", 1)[-1]]["params"] for name in sample_names],
        dtype=data["params"].dtype,
    )
    if not torch.equal(expected_params, data["params"]):
        raise ValueError("Cached parameter rows do not match the sample IDs.")

    params = data["params"].numpy()
    report = pd.DataFrame(
        params,
        index=pd.Index(sample_names, name="sample_id"),
        columns=[f"parameter_{i + 1}" for i in range(params.shape[1])],
    )

    score_names = [*components, "total_gnll"]
    for key in score_names:
        report[key] = data[key].numpy().reshape(-1)

    scores = report[score_names]
    summary = scores.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T.rename(columns={"50%": "median"})
    summary.index.name = "gnll_term"
    summary["min_sample"] = scores.idxmin()
    summary["max_sample"] = scores.idxmax()

    # Flag both tails separately for each component and for the total.
    q1, q3 = scores.quantile(0.25), scores.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    low = scores.lt(lower_fence)
    high = scores.gt(upper_fence)
    summary["lower_fence"] = lower_fence
    summary["upper_fence"] = upper_fence
    summary["n_low_outliers"] = low.sum()
    summary["n_high_outliers"] = high.sum()

    outliers = report.loc[(low | high).any(axis=1)].copy()
    outliers["flagged_terms"] = [
        "; ".join(
            f"{key}: {'low' if low.at[sample, key] else 'high'}"
            for key in score_names
            if low.at[sample, key] or high.at[sample, key]
        )
        for sample in outliers.index
    ]
    outliers = outliers.sort_values("total_gnll", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "gnll_summary.csv")
    report.to_csv(output_dir / "gnll_per_sample.csv")
    outliers.to_csv(output_dir / "gnll_outliers.csv")

    print(summary.round(4).to_string())
    print(f"\nFlagged {len(outliers)} of {len(report)} samples. Reports: {output_dir}")
    return summary, report, outliers

def validate_filename(name):
    if (
        not isinstance(name, str)
        or not name.strip()
        or name in {".", ".."}
        or any(char in name for char in ("/", "\\", "\0"))
    ):
        raise ValueError(f"Invalid directory or label name: {name!r}")
    return name

def extract_qoi(a, root):
    name = validate_filename(a['name'])
    labels = [validate_filename(label) for label in a["labels"]]
    n = int(a["values_per_label"])

    if not labels or n < 1 or len(set(labels)) != len(labels):
        raise ValueError("Require unique labels and positive values_per_label.")

    outputs = np.asarray(a["outputs"], dtype=float)
    reference = np.asarray(a["outputs_ref"], dtype=float)
    expected_width = len(labels) * n

    if outputs.ndim != 2 or outputs.shape[1] != expected_width:
        raise ValueError(
            f"{name}: expected outputs shape (n_samples, {expected_width}), "
            f"got {outputs.shape}"
        )
    if reference.shape != (expected_width,):
        raise ValueError(
            f"{name}: expected outputs_ref shape ({expected_width},), "
            f"got {reference.shape}"
        )
    if not np.isfinite(outputs).all() or not np.isfinite(reference).all():
        raise ValueError(f"{name}: data contain non-finite values.")

    outputs = outputs.reshape(len(outputs), len(labels), n)
    reference = reference.reshape(len(labels), n)

    # RDF files have radius and value columns.
    # Other QOIs use one value column without inventing coordinates.
    coordinates = None
    if name == "rdf":
        settings = a["settings"]
        if int(settings["n_bins"]) != n:
            raise ValueError("RDF n_bins does not match values_per_label.")

        r_min, r_max = map(float, settings["r_range"])
        if not np.isfinite([r_min, r_max]).all() or r_min >= r_max:
            raise ValueError("Invalid RDF radius range.")

        edges = np.linspace(r_min, r_max, n + 1)
        coordinates = (edges[:-1] + edges[1:]) / 2

    root = Path(root)
    sample_root = root / "04-qoi" / name
    reference_root = root / "02-reference" / name
    sample_root.mkdir(parents=True, exist_ok=True)
    reference_root.mkdir(parents=True, exist_ok=True)

    def write_values(path, values, label):
        if coordinates is None:
            data = values
            header = f"{name}: {label}"
        else:
            data = np.column_stack((coordinates, values))
            header = f"Radius\t{name}: {label}"

        np.savetxt(path, data, fmt="%.18e", header=header)

    # Write FFMD targets : rdf, H-bond
    for sample_index, sample_values in enumerate(outputs):
        sample_dir = sample_root / f"sample-000-{sample_index:04d}"
        sample_dir.mkdir(exist_ok=True)

        for label, values in zip(labels, sample_values):
            write_values(sample_dir / f"{label}.dat", values, label)

    # Write reference target
    for label, values in zip(labels, reference):
        write_values(reference_root / f"{label}.dat", values, label)

    print(
        f"{name}: extracted {len(outputs)} samples "
        f"and {len(labels)} reference files."
    )

if __name__ == '__main__':
    dataset_params = copy.deepcopy(DATASET_PARAMS)
    dataset = build_dataset(dataset_params, load=True)
