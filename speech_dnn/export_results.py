from typing import Optional

import numpy as np
import pandas as pd

from tbparse import SummaryReader


def export_results(log_dir: str) -> pd.DataFrame:
	result_reader = SummaryReader(log_dir, extra_columns = { 'dir_name' })
	results_df = result_reader.scalars
	metric_dfs = []

	for metric in results_df['tag'].unique():
		metric_data = results_df[results_df['tag'] == metric]
		last_step_idx = metric_data.groupby('dir_name')['step'].idxmax()
		agg_metric_data = metric_data.loc[last_step_idx].rename(columns = {'value': metric}).set_index('dir_name')
		metric_dfs.append(agg_metric_data[metric].sort_index())

	return pd.concat(metric_dfs, axis = 1).reset_index()


def extract_run_config(dir_name: str) -> str:
	fold_str_idx = dir_name.index('_fold_')
	return dir_name[:fold_str_idx].replace('[', '').replace(']', '').replace('\'', '').replace(', ', '+')


def extract_param(dir_name: str, param_key: str) -> Optional[str]:
	if param_key not in dir_name:
		return None

	param_key_idx = dir_name.index(param_key)
	val_start_idx = param_key_idx + len(param_key) + 1
	val_end_idx = dir_name.index('_', val_start_idx)

	return dir_name[val_start_idx:val_end_idx]


def aggregate_results(result_df: pd.DataFrame) -> pd.DataFrame:
	result_df['run_config'] = result_df['dir_name'].apply(extract_run_config)
	metrics = [col for col in result_df.columns if col not in ['dir_name', 'run_config']]
	agg = { 'dir_name': 'count' }

	for metric in metrics:
		agg[metric] = ['mean', 'std']

	agg_df = result_df.groupby('run_config').agg(agg).reset_index()
	agg_df['augmentation_type'] = agg_df['run_config'].apply(lambda cfg: extract_param(cfg, 'augmentation_type'))
	agg_df['augmentation'] = agg_df['run_config'].apply(lambda cfg: extract_param(cfg, 'augmentation'))
	agg_df['batch_size'] = agg_df['run_config'].apply(lambda cfg: extract_param(cfg, 'bs'))
	agg_df['learn_rate'] = agg_df['run_config'].apply(lambda cfg: extract_param(cfg, 'learn_rate'))
	agg_df['l2'] = agg_df['run_config'].apply(lambda cfg: extract_param(cfg, 'l2'))
	agg_df['dropout'] = agg_df['run_config'].apply(lambda cfg: extract_param(cfg, 'dropout'))

	agg_df['augmentation'] = np.where(
		agg_df['augmentation_type'].isna(), agg_df['augmentation'], agg_df['augmentation_type']
	)

	return agg_df.drop(columns = 'augmentation_type')
