import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import ConfusionMatrixDisplay

from speech_dnn.data_loader import KFoldImageDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


n_classes = 10


def evaluate_model(
		n_epochs, n_folds, batch_size, model_params, out_path, network_cls,
		experiment_idx = 0, total_experiments = 0, n_repeats = 5, seed = 1
) -> tuple[float, float, np.ndarray]:
	if not total_experiments:
		total_experiments = n_repeats * n_folds

	accuracy_scores = []
	confusion_matrix = np.zeros((n_classes, n_classes))
	
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True
	pl.seed_everything(seed, workers = True)
	
	seeds_summary = dict((param_key, []) for param_key in model_params.keys())
	seeds_summary['fold'] = []
	seeds_summary['batch_size'] = []
	seeds_summary['accuracy'] = []
	for fold in range(n_folds * n_repeats):
		model = network_cls(n_classes = n_classes, **model_params)
		experiment_idx += 1
		print(f'Run {experiment_idx}/{total_experiments}')
		
		data_module = KFoldImageDataModule(fold, batch_size, n_folds)
		logger = TensorBoardLogger(
			"lightning_logs",
			name = f'eff_bs_{batch_size}_{"_".join(f"{key}_{value}" for key, value in model_params.items())}_fold_{fold}'
		)
		
		trainer = pl.Trainer(accelerator = 'gpu', devices = 1, max_epochs = n_epochs, logger = logger, precision=16,
			callbacks=[EarlyStopping(monitor="val_loss", patience=5)]
		)
		trainer.fit(model, datamodule = data_module)
		print("Done!")

		print('Testing')
		test_result = trainer.validate(model, datamodule = data_module)

		accuracy = test_result[0]['val_accuracy']
		accuracy_scores.append(accuracy)
		confusion_matrix += model.get_test_confusion_matrix()

		for param_key in model_params.keys():
			seeds_summary[param_key].append(model_params[param_key])

		seeds_summary['batch_size'].append(batch_size)
		seeds_summary['accuracy'].append(accuracy)
		seeds_summary['fold'].append(fold)

	summary_df = pd.DataFrame(data = seeds_summary)
	summary_df.to_csv(
		f'{out_path}/bs_{batch_size}_{"_".join(f"{key}_{value}" for key, value in model_params.items())}.csv',
		index = False
	)

	accuracy_mean = float(np.mean(accuracy_scores))
	accuracy_std = float(np.std(accuracy_scores))
	confusion_matrix /= (n_repeats * n_folds)

	return accuracy_mean, accuracy_std, confusion_matrix


def save_confusion_matrix(confusion_matrix: np.ndarray, class_labels: list[str], path: str):
	out_path = Path(path)
	out_path.parent.mkdir(parents = True, exist_ok = True)

	fig, ax = plt.subplots(figsize = (6, 6))
	cm_disp = ConfusionMatrixDisplay(confusion_matrix.round().astype('int'), display_labels = class_labels)
	cm_disp.plot(xticks_rotation = 'vertical', ax = ax, colorbar = False, values_format = 'd')
	fig.tight_layout()
	plt.savefig(out_path)


def grid_search(
		n_epochs: int, out_path: str, param_grid: dict[str, list], seed: int, fold_count: int,
		repeat_count: int, network_cls
):
	Path(out_path).mkdir(parents = True, exist_ok = True)

	summary = dict((param_key, []) for param_key in param_grid.keys())
	summary['cm_path'] = []
	summary['accuracy_mean'] = []
	summary['accuracy_std'] = []

	batch_sizes = param_grid['batch_size']
	model_params = param_grid.copy()
	del model_params['batch_size']
	folds = list(range(fold_count))

	param_keys = []
	param_vals = []
	for key in model_params.keys():
		param_keys.append(key)
		param_vals.append(model_params[key])

	param_sets = []
	for val_set in itertools.product(*param_vals):
		param_set = { }
		for key, val in zip(param_keys, val_set):
			param_set[key] = val
		param_sets.append(param_set)

	total_experiments = len(batch_sizes) * len(param_sets) * repeat_count * len(folds)
	experiment_idx = 0
	classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
	for batch_size in batch_sizes:
		for i, param_set in enumerate(param_sets):
			print(f'Experiment: bs = {batch_size}, {param_set}')

			acc_mean, acc_std, confusion_matrix = evaluate_model(
				n_epochs = n_epochs, n_folds = fold_count, n_repeats = repeat_count,
				seed = seed, batch_size = batch_size, model_params = param_set, out_path = out_path,
				experiment_idx = experiment_idx, total_experiments = total_experiments,
				network_cls = network_cls
			)
			experiment_idx += fold_count * repeat_count

			cm_save_path = f'{out_path}/bs{batch_size}/cm-{i}.png'
			summary['batch_size'].append(batch_size)
			summary['accuracy_mean'].append(acc_mean)
			summary['accuracy_std'].append(acc_std)
			summary['cm_path'].append(cm_save_path)

			for param_key in param_set.keys():
				summary[param_key].append(param_set[param_key])

			save_confusion_matrix(
				confusion_matrix, class_labels = classes, path = cm_save_path
			)

	summary_df = pd.DataFrame(data = summary)
	summary_df.to_csv(f'{out_path}/summary.csv', index = False)
