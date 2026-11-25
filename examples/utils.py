"""Supervised VAE and independent, comparable W&B grid sweeps per data fold."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    ROOT = Path(__file__).resolve().parent
except:
    ROOT = Path("/Users/nikhilthota/Desktop/lab/pavel_iocb/projects/code_for_lab_collaborations/adam")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path

def config_key(config):
    return json.dumps(config, sort_keys=True)

def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def activation(name):
    factories = {None: nn.Identity, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
    if name not in factories:
        raise ValueError(f'Unsupported activation: {name!r}')
    return factories[name]()

def make_network(params):
    width = params.get('hidden_dim')
    hidden = params.get('hidden_layers', 0)
    dimensions = [params['input_dim']] + [width] * hidden + [params['output_dim']]
    layers = []
    for i, (start, end) in enumerate(zip(dimensions, dimensions[1:])):
        layer = nn.Linear(start, end)
        final = i == len(dimensions) - 2
        act = params.get('output_activation' if final else 'hidden_activation')
        gain = nn.init.calculate_gain(act) if act in ('relu', 'tanh') else 1.0
        kernel = params.get('layer_kernel_init', 'xavier_normal')
        if kernel not in ('xavier_normal', 'xavier_uniform'):
            raise ValueError(f'Unsupported layer_kernel_init: {kernel}')
        initializer = nn.init.xavier_normal_ if kernel == 'xavier_normal' else nn.init.xavier_uniform_
        initializer(layer.weight, gain=gain)
        if params.get('layer_bias_init', 'zeros') != 'zeros':
            raise ValueError('layer_bias_init must be zeros')
        nn.init.zeros_(layer.bias)
        layers.extend([layer, activation(act)])
    return nn.Sequential(*layers)

def build_parameters(module_params):
    """Build W&B parameter definitions for fixed settings and sweeps."""
    parameters = {}

    for name, module in module_params['modules'].items():
        optimize = module.get('parameter_optimization', False)

        for key, value in module.items():
            if key == 'parameter_optimization':
                continue

            parameter_name = f'{name}-{key}'

            if isinstance(value, dict) and 'values' in value:
                choices = value['values']

                if not choices:
                    raise ValueError(
                        f'{parameter_name}: values must not be empty'
                    )

                if optimize:
                    parameters[parameter_name] = copy.deepcopy(value)
                elif len(choices) == 1:
                    parameters[parameter_name] = {
                        'value': copy.deepcopy(choices[0])
                    }
                else:
                    raise ValueError(
                        f'{parameter_name}: multiple values require '
                        'parameter_optimization=True; otherwise provide '
                        'a fixed value'
                    )

            elif isinstance(value, dict) and 'value' in value:
                parameters[parameter_name] = {
                    'value': copy.deepcopy(value['value'])
                }

            else:
                parameters[parameter_name] = {
                    'value': copy.deepcopy(value)
                }

    return parameters

def kl_weight(epoch, warmup_epochs, target_beta):
    if warmup_epochs <= 0:
        return target_beta
    return target_beta * min(1.0, epoch / warmup_epochs)

def fit_normalization(dataset, indices):
    x = dataset.tensor_dictionary['params'][indices]
    mean, std = x.mean(0), x.std(0, unbiased=False)
    return mean, torch.where(std == 0, torch.ones_like(std), std)

def resolve_modules(module_params, config, input_dim):
    # Copy the module dictionary with the values to optimize.
    modules = copy.deepcopy(module_params['modules'])
    for name, module in modules.items():
        for key, value in module.items():
            if key == 'parameter_optimization':
                continue
            parameter_name = f'{name}-{key}'
            if parameter_name in config:
                module[key] = copy.deepcopy(config[parameter_name])
            elif isinstance(value, dict) and 'value' in value:
                module[key] = copy.deepcopy(value['value'])
            elif isinstance(value, dict) and 'values' in value:
                module[key] = config[parameter_name]
    encoder = modules['encoder']
    latent_dim = encoder['output_dim']
    encoder['input_dim'] = input_dim
    for name, module in modules.items():
        if name == 'encoder':
            continue
        for key, value in module.items():
            if value == 'mirror':
                module[key] = latent_dim if key == 'input_dim' else encoder[key]
    modules['decoder']['output_dim'] = input_dim
    modules['predictor']['output_dim'] = 1
    return modules

def aggregate_results(records, num_folds, expected_configs, monitor_loss_for_best='prediction_loss'):
    # Records contain : fold, config, run_id, best_epoch, losses, checkpoint, status
    # First filter for completed records ...
    completed = [record for record in records if record['status'] == 'completed']
    winners = []
    for fold in range(num_folds):
        # Next arange candidates by fold -> fold_winners
        candidates = [record for record in completed if record['fold'] == fold]
        if candidates:
            winners.append(min(candidates, key=lambda record: record['losses'][monitor_loss_for_best]))

    # Now sorting by config
    ranked, incomplete = [], []
    for config in expected_configs:
        # Extracts all the folds with matching config (in this should return 5 matches)
        matches = [record for record in completed if config_key(record['config']) == config_key(config)]
        # A repeated callback must not overweight a fold.
        by_fold = {record['fold']: record for record in matches}
        # In case config was not complete all five folds of data.
        if set(by_fold) != set(range(num_folds)):
            incomplete.append({'config': config, 'completed_folds': sorted(by_fold)})
            continue
        matches = [by_fold[i] for i in range(num_folds)]
        # Compute the statistics across the folds
        statistics = {
            key: {'mean': float(np.mean([r['losses'][key] for r in matches])),
                  'std': float(np.std([r['losses'][key] for r in matches], ddof=1))}
            for key in matches[0]['losses']
        }
        # Add it to the list of configs ...
        ranked.append({'config': config, 'statistics': statistics, 'fold_results': matches})
    # Finally sort the list based on the chosen metric ...
    ranked.sort(key=lambda result: result['statistics'][monitor_loss_for_best]['mean'])
    return {'fold_winners': winners, 'ranking': ranked, 'incomplete': incomplete,
            'failures': [r for r in records if r['status'] != 'completed']}

@torch.no_grad()
def sample_mcmc(
    model,
    log_probability,
    initial_positions,
    n_steps=5000,
    warmup=1000,
    step_size=0.1,
    seed=42,
):
    """
    initial_positions: (n_walkers, latent_dim)

    Returns post-warmup samples, log probabilities,
    and acceptance rates over all steps.
    """
    if not 0 <= warmup < n_steps:
        raise ValueError("Require 0 <= warmup < n_steps.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")

    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype

    model.eval()
    z = initial_positions.detach().clone().to(device=device, dtype=dtype)

    if z.ndim != 2 or min(z.shape) == 0:
        raise ValueError("initial_positions must have shape (n_walkers, latent_dim).")

    rng = torch.Generator(device=device).manual_seed(seed)

    current_logp = torch.tensor(log_probability(z),
                                device=device,
                                dtype=dtype)
    if not torch.isfinite(current_logp).all():
        raise ValueError("Initial positions must have finite log probabilities.")

    n_walkers = z.shape[0]
    accepted_count = torch.zeros(n_walkers, device=device)
    samples = []
    log_probs = []

    for step in range(n_steps):
        # Symmetric Gaussian random-walk proposal.
        noise = torch.randn(
            z.shape, generator=rng, device=device, dtype=dtype
        )
        proposed_z = z + step_size * noise

        proposed_logp = torch.tensor(log_probability(proposed_z),
                                     device=device,
                                     dtype=dtype)

        # Metropolis acceptance, evaluated independently per walker.
        log_u = torch.rand(
            n_walkers, generator=rng, device=device, dtype=dtype
        ).log()

        accepted = (
            torch.isfinite(proposed_logp)
            & (log_u < proposed_logp - current_logp)
        )

        z = torch.where(accepted[:, None], proposed_z, z)
        current_logp = torch.where(accepted, proposed_logp, current_logp)
        accepted_count += accepted

        # Retain repeated positions when proposals are rejected.
        if step >= warmup:
            samples.append(z.clone())
            log_probs.append(current_logp.clone())

    return {
        "samples": torch.stack(samples),
        "log_probability": torch.stack(log_probs),
        "acceptance_rate": accepted_count / n_steps,
    }
