"""Supervised VAE and independent, comparable W&B grid sweeps per data fold."""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path
from functools import partial
import math
from datetime import datetime, timezone
import json5

import numpy as np
import torch
import wandb
import yaml
from sklearn.model_selection import KFold, train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from dataset_builder import build_dataset, extract_qoi, TensorDataset
import utils

try:
    ROOT = Path(__file__).resolve().parent
except:
    ROOT = Path("/Users/nikhilthota/Desktop/lab/pavel_iocb/projects/code_for_lab_collaborations/adam")
# Name for the project
PROJECT = 'peptide_supervised_vae'
RDF_TARGETS = ['CT1', 'H', 'HB1', 'NH1', 'O']
HB_TARGETS = ['PROA(NH1) to SOL(OW4)', 'SOL(OW4) to PROA(NH1)', 'SOL(OW4) to PROA(O)']
RDF_DATASET_PARAMS = {
    'name': 'rdf',
    'reference_path': f'{ROOT}/02-reference/rdf',
    'qoi_path': f'{ROOT}/04-qoi/rdf',
    'samples_yaml_path': f'{ROOT}/03-sample/samples.yaml',
    'test_frac': 0.1,
    'k_folds': 5,
    'targets': RDF_TARGETS,
    'nuisance': [0.002] * 5,
}
HB_DATASET_PARAMS = {
    'name': 'hb',
    'reference_path': f'{ROOT}/02-reference/hb',
    'qoi_path': f'{ROOT}/04-qoi/hb',
    'samples_yaml_path': f'{ROOT}/03-sample/samples.yaml',
    'test_frac': 0.1,
    'k_folds': 5,
    'targets': HB_TARGETS,
    'nuisance': [0.002] * 3,
}
MODULE_PARAMS = {
    'name': 'supVAE',
    'modules': {
        'encoder': {
            'parameter_optimization': True,
            'output_dim': {'values': [2, 3, 4, 5]},
            'hidden_dim': {'values': [25, 50, 75]},
            'hidden_layers': {'values': [2, 3]},
            'hidden_activation': 'relu',
            'output_activation': 'tanh',
            'layer_kernel_init': 'xavier_normal',
            'layer_bias_init': 'zeros',
        },
        'predictor': {
            'input_dim': 'mirror',
            'output_dim': 1,
            'hidden_dim': 'mirror',
            'hidden_layers': 'mirror',
            'hidden_activation': 'relu',
            'output_activation': None, # Output should always be negative
            'layer_kernel_init': 'xavier_normal',
            'layer_bias_init': 'zeros',
        },
        'decoder': {
            'input_dim': 'mirror',
            'hidden_dim': 'mirror',
            'hidden_layers': 'mirror',
            'hidden_activation': 'relu',
            'output_activation': 'sigmoid', # Testing out 'sigmoid' since partial charges lie in [-1, 1]
            'layer_kernel_init': 'xavier_normal',
            'layer_bias_init': 'zeros',
        },
    },
}
TRAIN_PARAMS = {
    'seed': 42, 'num_epochs': 100,
    'lr': 0.001,
    'l2_coeff': 0.001, 'l1_coeff':0,
    'beta': 0.01, 'kl_warmup_epochs': 20,
    'accelerator': 'cpu',
    'batch_size': 10,
}

class VAE(nn.Module):
    """Template encoder/decoder with Gaussian latents and supervised scalar heads."""
    def __init__(self, modules, targets):
        super().__init__()
        self.model_config = copy.deepcopy(modules)
        self.targets = list(targets)
        self.encoder = utils.make_network(modules['encoder'])
        latent_dim = modules['encoder']['output_dim']
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.logvar = nn.Linear(latent_dim, latent_dim)
        for head in (self.mu, self.logvar):
            nn.init.xavier_normal_(head.weight)
            nn.init.zeros_(head.bias)
        self.decoder = utils.make_network(modules['decoder'])
        self.predictors = nn.ModuleList([utils.make_network(modules['predictor']) for _ in targets])

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        features = self.encoder(x)
        mu, logvar = self.mu(features), self.logvar(features).clamp(-20, 20)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return {
            'mu': mu, 'logvar': logvar, 'z': z, 'x_pred': self.decode(z),
            'predictions': {target: head(z) for target, head in zip(self.targets, self.predictors)},
        }

def epoch_pass(model, loader, mean, std, beta, device, l1_coeff, optimizer=None):
    model.train(optimizer is not None)
    totals, count = {}, 0

    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            # Normalize parameters
            x = (batch['params'] - mean.to(device)) / std.to(device)
            if optimizer is not None:
                optimizer.zero_grad()

            # Get model outputs
            outputs = model(x)

            # user: Add your losses to compute and track here ...
            losses = {f'{name}_pred_loss': F.l1_loss(outputs['predictions'][name], batch[name]) for name in model.targets}
            losses['prediction_loss'] = sum(losses.values())
            losses['reconstruction_loss'] = F.l1_loss(outputs['x_pred'], x)
            if l1_coeff > 0:
                losses['l1_loss'] = sum(
                    parameter.abs().sum()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
            losses['kl_loss'] = -0.5 * (
                1 + outputs['logvar'] - outputs['mu'].square() - outputs['logvar'].exp()
            ).sum(1).mean()
            if l1_coeff > 0:
                losses['total_loss'] = losses['prediction_loss'] + losses['reconstruction_loss'] + beta * losses['kl_loss'] + l1_coeff * losses['l1_loss']
            else:
                losses['total_loss'] = losses['prediction_loss'] + losses['reconstruction_loss'] + beta * losses['kl_loss']


            if any(not torch.isfinite(value) for value in losses.values()):
                raise ValueError('Non-finite VAE loss')

            if optimizer is not None:
                losses['total_loss'].backward()
                optimizer.step()

            count += len(x)

            for key, value in losses.items():
                if key != 'l1_loss':
                    totals[key] = totals.get(key, 0.0) + value.item() * len(x)

    metrics = {key: value / count for key, value in totals.items()}
    if l1_coeff > 0:
        with torch.no_grad():
            metrics['l1_loss'] = sum(
                    parameter.abs().sum()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ).item()
    return metrics

def train_val_loop(model, run, run_dir, train_params, dataset, train_indices, val_indices, monitor_loss_for_best='prediction_loss'):
    mean, std = utils.fit_normalization(dataset, train_indices)
    device = torch.device(train_params['accelerator'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=train_params['lr'],
                                weight_decay=train_params['l2_coeff'])
    train_loader = DataLoader(
        Subset(dataset, list(train_indices)), batch_size=train_params['batch_size'] or len(train_indices),
        shuffle=True, generator=torch.Generator().manual_seed(train_params['seed']),
    )
    val_loader = DataLoader(
        Subset(dataset, list(val_indices)), batch_size=train_params['batch_size'] or len(val_indices),
        shuffle=False, generator=torch.Generator().manual_seed(train_params['seed']),
    )
    best, best_epoch, best_metrics = float('inf'), 0, None
    run.define_metric('epoch')
    run.define_metric('train/*', step_metric='epoch')
    run.define_metric('val/*', step_metric='epoch')

    with (run_dir / 'epochs.jsonl').open('w', encoding='utf-8') as history:
        for epoch in range(1, train_params['num_epochs'] + 1):
            beta = utils.kl_weight(epoch, train_params['kl_warmup_epochs'], train_params['beta'])
            train = epoch_pass(model, train_loader, mean, std, beta, device, train_params['l1_coeff'], optimizer)
            val = epoch_pass(model, val_loader, mean, std, beta, device, train_params['l1_coeff'])
            metrics = {'epoch': epoch,
                        **{f'train/{k}': v for k, v in train.items()},
                        **{f'val/{k}': v for k, v in val.items()}}
            run.log(metrics)
            history.write(json.dumps(metrics) + '\n')
            if val[monitor_loss_for_best] < best:
                best, best_epoch, best_metrics = val[monitor_loss_for_best], epoch, val
                torch.save({
                    'state_dict': model.state_dict(), # Stores mode weights
                    'modules': model.model_config, # Stores model architcture
                    'targets': model.targets,
                    'mean': mean, 'std': std, # The mean and std of data needed for destand.
                    'epoch': epoch,
                    'train_params': train_params, # Training set parameters
                }, run_dir / 'best.pt')
    run.summary.update({'best_val_prediction_loss': best, 'best_epoch': best_epoch})
    return {'best_epoch': best_epoch, 'losses': best_metrics, 'checkpoint': str(run_dir / 'best.pt')}

def run_wandb_agent(dataset, indices, module_params, train_params, output_dir, fold, project, config=None, disabled=False):
    if disabled:
        # The run config is provided from run_experiment/user/
        if config is None:
            raise ValueError("Provide config when logging is disabled.")
        run = wandb.init(
            project=project,
            config=config,
            mode='disabled' if disabled else 'online',
            group=output_dir.name,
            job_type='fold-training',
            dir=str(output_dir),
        )
    else:
        # The run config is provided by wand.sweep
        run = wandb.init(
            project=project,
            mode='online',
            group=output_dir.name,
            job_type='fold-training',
            dir=str(output_dir),
        )

    selected = dict(run.config)
    digest = hashlib.sha256(utils.config_key(selected).encode()).hexdigest()[:12]
    run_dir = output_dir / f'fold-{fold}' / f'{digest}-{run.id}'
    record = {'fold': fold, 'config': selected, 'run_id': run.id}
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        utils.seed_everything(train_params['seed'] + fold)
        # This provides the dictionary format needed to intialize the VAE (see MODULE_PARAMS)
        modules = utils.resolve_modules(module_params, selected, dataset.tensor_dictionary['params'].shape[1])
        model = VAE(modules, list(dataset.tensor_dictionary)[1:])
        # indices contain both train and val
        # Choose any of the following losses for monitoring and storing best model as best.pt
        # prediction_loss, reconstruction_loss, total_loss
        record.update(train_val_loop(model, run, run_dir, train_params, dataset, *indices))
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        run.summary['error'] = record['error']
        print(f'Fold {fold}, {selected}: {record["error"]}', flush=True)
    finally:
        run.finish(exit_code=0 if record.get('status') == 'completed' else 1)
    return record

def refit_and_test(dataset, development, test, winner, module_params, train_params, output_dir, project, disabled):
    settings = dict(train_params)
    settings['num_epochs'] = max(1, int(np.floor(np.median([r['best_epoch'] for r in winner['fold_results']]) + 0.5)))
    utils.seed_everything(settings['seed'])
    modules = utils.resolve_modules(module_params, winner['config'], dataset.tensor_dictionary['params'].shape[1])
    model = VAE(modules, list(dataset.tensor_dictionary)[1:]).to(settings['accelerator'])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['l2_coeff'])
    # Storing the mean and std of the development set in model state.
    mean, std = utils.fit_normalization(dataset, development)
    loader = DataLoader(
        Subset(dataset, list(development)), batch_size=settings['batch_size'] or len(development),
        shuffle=True, generator=torch.Generator().manual_seed(train_params['seed']),
    )
    with wandb.init(project=project, group=output_dir.name, job_type='final-refit', config=winner['config'],
                    dir=str(output_dir), mode='disabled' if disabled else 'online') as run:
        # Train the model on the development set.
        for epoch in range(1, settings['num_epochs'] + 1):
            beta = utils.kl_weight(epoch, train_params['kl_warmup_epochs'], train_params['beta'])
            metrics = epoch_pass(model, loader, mean, std, beta, settings['accelerator'],
                                settings['l1_coeff'], optimizer=optimizer)
            run.log({'epoch': epoch, **{f'train/{k}': v for k, v in metrics.items()}})
        # Load the test indices into dataloader
        test_dataloader = DataLoader(
                            Subset(dataset, list(test)), batch_size=settings['batch_size'] or len(test),
                            shuffle=False, generator=torch.Generator().manual_seed(train_params['seed']),
                        )
        # Test the model on the development set.
        losses = epoch_pass(model, test_dataloader, mean, std, settings['beta'], settings['accelerator'],settings['l1_coeff'])
        run.summary.update({f'test/{key}': value for key, value in losses.items()})
    torch.save({'state_dict': model.state_dict(), 'modules': modules, 'config': winner['config'],
                'targets': model.targets, 'mean': mean, 'std': std, 'epoch': settings['num_epochs'],
                'train_params': settings}, output_dir / 'final_model.pt')
    return {'epochs': settings['num_epochs'], 'losses': losses, 'checkpoint': str(output_dir / 'final_model.pt')}

def run_experiment(dataset, dataset_params, module_params, train_params, output_dir, project, disabled=False):
    if train_params['num_epochs'] < 1 or dataset_params['k_folds'] < 2:
        raise ValueError('num_epochs must be positive and k_folds must be at least two')
    if train_params['beta'] < 0 or train_params['lr'] <= 0 or train_params['l2_coeff'] < 0 or train_params['l1_coeff'] < 0:
        raise ValueError('Require beta >= 0, lr > 0, l2_coeff >= 0 and l1_coeff >= 0')
    if train_params['batch_size'] is not None and train_params['batch_size'] < 1:
        raise ValueError('batch_size must be positive or null')
    output_dir.mkdir(parents=True, exist_ok=False)
    # Create the development (train + val) and test datasets
    development, test = train_test_split(
        np.arange(len(dataset)), test_size=dataset_params['test_frac'], random_state=train_params['seed'],
        )
    splitter = KFold(n_splits=dataset_params['k_folds'], shuffle=True, random_state=train_params['seed'])
    folds = [(development[train], development[val]) for train, val in splitter.split(development)]
    # Store the dataset, train and module params as a .json file
    utils.write_json(output_dir / 'settings.json', {'dataset': dataset_params, 'model': module_params, 'training': train_params})
    utils.write_json(output_dir / 'splits.json', {
        'sample_ids': dataset.sample_ids, 'development': development.tolist(), 'test': test.tolist(),
        'folds': [{'train': tr.tolist(), 'val': va.tolist()} for tr, va in folds],
    })

    # Creates a parameters dictionary in the format wandb.sweep requires.
    # Ex: {'encoder-output_dim' : {'value': 10}, 'encoder-hidden_dim': {'values': [25, 75]}}
    model_parameters = utils.build_parameters(module_params)

    # ------------------------------------------------------------
    # WHEN wandb.sweep IS NOT CALLED !
    # Creates a list of keys in parameters dictionary
    # Ex: ['encoder-output_dim', 'encoder-hidden_dim']
    names = list(model_parameters)
    choices = [
        spec['values'] if 'values' in spec else [spec['value']]
        for spec in model_parameters.values()
    ]
    # Creates a list of individual dictionaries based on values provided.
    # Ex: [{'encoder-output_dim':10, 'encoder-hidden_dim':25},
    #      {'encoder-output_dim':10, 'encoder-hidden_dim':75}]
    expected = [dict(zip(names, values))
                for values in itertools.product(*choices)]
    # ------------------------------------------------------------

    records = []
    for fold, indices in enumerate(folds):
        def callback():
            record = run_wandb_agent(dataset, indices, module_params, train_params,
                                     output_dir, fold, project)
            records.append(record)
            utils.write_json(output_dir / 'runs.json', records)
        print(f'Fold {fold + 1}/{len(folds)}: {len(expected)} configurations', flush=True)
        # Disabled Logging
        if disabled:
            for config in expected:
                records.append(run_wandb_agent(dataset, indices, module_params, train_params,
                                               output_dir, fold, project,
                                               config=config, disabled=True))
                utils.write_json(output_dir / 'runs.json', records)
        # Logging enabled through wand.sweep
        else:
            sweep_id = wandb.sweep({
                'name': f'{module_params["name"]}-fold-{fold}', 'method': 'grid',
                'metric': {'name': 'best_val_prediction_loss', 'goal': 'minimize'},
                'parameters': model_parameters,
            }, project=project)
            (output_dir / f'fold-{fold}-sweep.txt').write_text(sweep_id + '\n', encoding='utf-8')
            # Run hyperparam search on each fold of data.
            wandb.agent(sweep_id, function=callback, count=len(expected), project=project)

    # Report contains :
    # fold_winners : arranging configs according to 'monitor_loss_for_best' for each data fold
    # ranking : Taking average across folds for each loss, for each config and ranking based on 'monitor_loss_for_best'
    # incomplete : Runs which did not finish
    # failures : Runs which did not succeed
    # best_config
    # test : test set losses on best config
    # experiment_type : single_configuration / hyperparameter_search
    report = utils.aggregate_results(records, len(folds), expected)
    is_search = len(expected) > 1
    report['experiment_type'] = 'hyperparameter_search' if is_search else 'single_configuration'
    utils.write_json(output_dir / 'report.json', report)
    with (output_dir / 'ranking.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        keys = list(report['ranking'][0]['statistics']) if report['ranking'] else []
        writer.writerow(['config'] + [f'{key}_{stat}' for key in keys for stat in ('mean', 'std')])
        for result in report['ranking']:
            writer.writerow([utils.config_key(result['config'])] + [result['statistics'][key][stat] for key in keys for stat in ('mean', 'std')])
    if not report['ranking']:
        raise RuntimeError(f'No configuration completed all folds; inspect {output_dir / "results.json"}')

    winner = report['ranking'][0]
    report['best_config'] = winner['config']
    # Single run with best config ...
    report['test'] = refit_and_test(dataset, development, test, winner,
                                    module_params, train_params, output_dir, project, disabled)
    utils.write_json(output_dir / 'report.json', report)
    # Logging to WnB if enabled ...
    if not disabled:
        with wandb.init(project=project,
                        group=output_dir.name,
                        job_type='search-summary' if is_search else 'training_summary',
                        dir=str(output_dir),
                        mode='disabled' if disabled else 'online'
        ) as run:
            if is_search:
                columns = ['config', 'mean_prediction_mae', 'std_prediction_mae']
                table = [
                            [
                                utils.config_key(r['config']),
                                r['statistics']['prediction_loss']['mean'],
                                r['statistics']['prediction_loss']['std']
                            ]
                            for r in report['ranking']
                        ]
                run.log({'ranking': wandb.Table(
                            columns=columns,
                            data=table)})

            run.summary.update({
                'best_config': winner['config'],
                'best_cv': winner['statistics'],
                'test': report['test']['losses']
            })
    print('Fold winners:')
    for result in report['fold_winners']:
        print(f'  Fold {result["fold"]}: {result["config"]}; prediction MAE sum={result["losses"]["prediction_loss"]:.6g}')
    stats = winner['statistics']['prediction_loss']
    print(f'Overall best: {winner["config"]}\nCV prediction MAE sum: {stats["mean"]:.6g} +/- {stats["std"]:.6g}')
    print(f'Held-out test prediction MAE sum: {report["test"]["losses"]["prediction_loss"]:.6g}\nResults: {output_dir}')
    return report

if __name__ == '__main__':
    # Run only once in start to extract data and create file structure
    # qoi_files = [Path('qoi-rdf.pt'), Path('qoi-hb.pt')]
    # for qoi_file in qoi_files:
    #     with qoi_file.open(encoding="utf-8") as f:
    #         qoi_json = json5.load(f)
    #     extract_qoi(qoi_json, '.')

    rdf_dataset_params, hb_dataset_params, module_params, train_params = copy.deepcopy(RDF_DATASET_PARAMS), copy.deepcopy(HB_DATASET_PARAMS), copy.deepcopy(MODULE_PARAMS), copy.deepcopy(TRAIN_PARAMS)
    # Create one dataset by adding the gaussian negative log likelihoods (See likelihoods.py)
    # rdf_dataset = build_dataset(rdf_dataset_params, load=False)
    # hb_dataset = build_dataset(hb_dataset_params, load=False)
    tensor_dataset = build_dataset([rdf_dataset_params, hb_dataset_params])

    # return torch.load('datasets/tensor_dataset_dictionary.pt',
    #                     weights_only=False,
    #                     map_location='cpu')

    # Uncommment to run hyperparameter optimization
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output_dir = utils.resolve_path(ROOT / 'ae_param_search' / stamp)
    run_experiment(tensor_dataset, rdf_dataset_params, module_params, train_params,
                    output_dir, PROJECT, disabled=False)
