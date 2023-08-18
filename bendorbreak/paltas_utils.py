import importlib
from pathlib import Path
import os
import sys
import warnings

import numpy as np
from scipy import stats
import tensorflow as tf
import paltas
import paltas.Analysis
import pandas as pd

import bendorbreak as bb
export, __all__ = bb.exporter()


def _make_param_dict(separator='/'):
	sep = separator
	mdef = f'main_deflector{sep}parameters{sep}'
	return dict((
		(mdef + 'theta_E', 'theta_E'),
		(f'subhalo{sep}parameters{sep}sigma_sub', 'sigma_sub'),
		('subhalo{sep}parameters{sep}shmf_plaw_index', 'shmf_plaw_index'),
		('los{sep}parameters{sep}delta_los', 'delta_los'),
		(mdef + 'center_x', 'center_x'),
		(mdef + 'center_y', 'center_y'),
		(mdef + 'gamma', 'gamma'),
		(mdef + 'gamma1', 'gamma1'),
		(mdef + 'gamma2', 'gamma2'),
		(mdef + 'e1', 'e1'),
		(mdef + 'e2', 'e2')))

# Mappings from short to long parameter names and back
SHORT_NAMES = _make_param_dict(separator='_')
PALTAS_NAMES = {v: k for k, v in SHORT_NAMES.items()}
# Long parameter names with slashes for separators (for building config files)
PALTAS_SLASH_NAMES = {v: k for k, v in _make_param_dict(separator='/').items()}

# Parameter order used in the March 2022 paper
MARCH_2022_PARAMETERS = (
	'main_deflector_parameters_theta_E',
	'main_deflector_parameters_gamma1',
	'main_deflector_parameters_gamma2',
	'main_deflector_parameters_gamma',
	'main_deflector_parameters_e1',
	'main_deflector_parameters_e2',
	'main_deflector_parameters_center_x',
	'main_deflector_parameters_center_y',
	'subhalo_parameters_sigma_sub')

DEFAULT_PARAMETERS = tuple([
	PALTAS_NAMES[p]
	for p in ('theta_E', 'sigma_sub', 'gamma')])
DEFAULT_TRAINING_CONFIG = (
	Path(paltas.__path__[0]) / 'Configs' / 'paper_2203_00690' / 'config_train.py')
DEFAULT_NORMS_CSV = Path(__file__).parent / "march_2022_paper_norms.csv"


__all__ += [
	'SHORT_NAMES', 'PALTAS_NAMES', 'PALTAS_SLASH_NAMES',
	'MARCH_2022_PARAMETERS', 'DEFAULT_PARAMETERS', 'DEFAULT_TRAINING_CONFIG',
	'DEFAULT_NORMS_CSV']


@export
def march2022_paper_model():
	model_url = 'https://zenodo.org/record/6326743/files/xresnet34_full_final.h5?download=1'
	model_path = Path('models/xresnet34_full_final.h5')
	if not model_path.exists():
		print("Downloading network from zenodo. Can take minutes.")
		bb.download_and_save(model_url, model_path)
	return tf.keras.models.load_model(
		Path(model_path),
		custom_objects=dict(loss=None))


@export
def load_paltas_config(config_path):
	"""Return imported config module from config_path"""
	config_dir, config_file = os.path.split(os.path.abspath(config_path))
	sys.path.insert(0, config_dir)
	config_name, _ = os.path.splitext(config_file)
	config_module = importlib.import_module(config_name)
	sys.path = sys.path[1:]
	return config_module


@export
def parse_paltas_network_output(result, learning_params, loss_type='full'):
	"""Return image_mean, image_prec from a raw model.predict() result.

	Here image_mean and image_prec are (n_images, n_params) and
	(n_images, n_params, n_params) numpy arrays, respectively.

	Arguments:
	 - result: output of model.predict()
	 - learning_params: list of parameter names
	 - loss_type: loss function the network was trained with;
		'full' or 'diagonal'.
	"""
	if loss_type.startswith('diag'):
		loss = paltas.Analysis.loss_functions.DiagonalCovarianceLoss(
			len(learning_params), flip_pairs=None, weight_terms=None)
		image_mean, log_var_pred = [x for x in loss.convert_output(result)]
		# Convert to precision matrices. Too lazy to vectorize
		log_var_pred = log_var_pred.clip(-10, 10)
		image_prec = tf.stack([tf.diag(tf.exp(-x)) for x in log_var_pred])
	else:
		loss = paltas.Analysis.loss_functions.FullCovarianceLoss(
			len(learning_params), flip_pairs=None, weight_terms=None)
		image_mean, image_prec, _ = [x for x in loss.convert_output(result)]
	return image_mean, image_prec


@export
def paltas_mean_cov(config_path, params):
	"""Return (mean, cov) arrays of distribution of params
	as defined by paltas config at config_path

	Arguments:
	 - config_path: path to paltas config file
	 - params: sequence of parameter names to extract

	Returns tuple with:
	 - mean: (n_params) array with mean of each parameter
	 - cov: (n_params, n_params) array with covariance matrix
	"""
	if not str(config_path).endswith('.py'):
		# Maybe the user gave a folder name
		# If it has only one python file, fine, that must be the config
		py_files = list(Path(config_path).glob('*.py'))
		if len(py_files) == 1:
			config_path = py_files[0]
		else:
			raise ValueError(f"{config_path} has multiple python files")

	config_dict = load_paltas_config(config_path).config_dict
	flat_dict = _flatten_dict(config_dict)

	mean_std = dict()
	for pname in params:
		value = flat_dict[pname]
		if value is None:
			# We have to get the value from the cross object dict
			for x_pname, value in config_dict['cross_object']['parameters'].items():
				x_params = [x.replace(':', '_parameters_') for x in x_pname.split(',')]
				if pname not in x_params:
					continue
				if isinstance(value, paltas.Sampling.distributions.Duplicate):
					value = value.dist
				elif isinstance(value, paltas.Sampling.distributions.DuplicateScatter):
					# This is OK only if the parameter is the _first_ one listed,
					# the second one gets extra scatter
					if pname != x_params[0]:
						raise ValueError("Second element of DuplicateScatter gets extra scatter, don't know dist")
					value = value.dist
				break
			else:
				raise ValueError(f"{pname} is None in config and not in cross_object")
		# Convert value to (mean, std)
		if isinstance(value, (int, float)):
			# Value was kept constant
			mean, std = value, 0
		else:
			# Let's hope it is a scipy stats distribution, so we can
			# back out the mean and std through sneaky ways
			self = value.__self__
			dist = self.dist
			if not isinstance(dist, (
					stats._continuous_distns.norm_gen)):
				warnings.warn(
					f"Approximating {dist.name} for {pname} with a normal distribution",
					UserWarning)
			mean, std = self.mean(), self.std()

		mean_std[pname] = (mean, std)

	# Produce mean vector / cov matrix in the right order
	mu, std = np.array([mean_std[pname] for pname in params]) .T
	cov = np.diag(std**2)
	return mu, cov


@export
def parse_norms_csv(path=DEFAULT_NORMS_CSV):
	"""Return parameter (names, means, scales) from a paltas norms csv.

	names will be a tuple of strings, means and scales both float arrays.
	"""
	df = pd.read_csv(path)
	param_names = tuple(df['parameter'].tolist())
	param_means, param_scales = [
		np.asarray(df[col].values, dtype=float)
		for col in ['mean', 'std']]
	return param_names, param_means, param_scales


def _flatten_dict(d, parent_key='', sep='_'):
    # Stolen from https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
