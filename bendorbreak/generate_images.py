from pathlib import Path
import os
import random
import re
import shutil
import string
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paltas.Configs.config_handler import ConfigHandler
from tqdm import tqdm

import bendorbreak as bb
export, __all__ = bb.exporter()


DEFAULT_DATASET = Path('./paltas_datasets/unnamed')
__all__ += ['DEFAULT_DATASET']


@export
def generate_dataset(
    config_path=bb.DEFAULT_TRAINING_CONFIG,
    save_folder=DEFAULT_DATASET,
    n: int = 1,
    save_png_too: bool = False,
    tf_record: bool = False,
    quiet: bool = False,
):
    """Generate simulated strong lensing images

    Args:
        config_path: Path to paltas configuration file
        save_folder: Folder to save images to.
        n: Size of dataset to generate (default 1)
        save_png_too: if True, also save a PNG for each image for debugging
        tf_record: if True, generate the tfrecord for the dataset
    """
    save_folder = str(save_folder)
    # TODO: everything below is the same as paltas, except that it is a
    # function, not a cmd script. Make PR if paltas is still maintained.

    # Make the directory if not already there
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not quiet:
        print("Save folder path: {:s}".format(save_folder))

    # Copy out config dict
    shutil.copy(os.path.abspath(config_path), save_folder)

    # Gather metadata in a list, will be written to dataframe later
    metadata_list = []
    metadata_path = os.path.join(save_folder, "metadata.csv")

    # Initialize our config handler
    config_handler = ConfigHandler(config_path)

    # Generate our images
    if not quiet:
        pbar = tqdm(total=n)
    successes = 0
    tries = 0
    while successes < n:
        # We always try
        tries += 1

        # Attempt to draw our image
        image, metadata = config_handler.draw_image(new_sample=True)

        # Failed attempt if there is no image output
        if image is None:
            continue

        # Save the image and the metadata
        filename = os.path.join(save_folder, "image_%07d" % successes)
        np.save(filename, image)
        if save_png_too:
            plt.imsave(
                filename + ".png",
                np.log10(image.clip(0, None)),
                cmap=plt.cm.magma,
            )

        metadata_list.append(metadata)

        # Write out the metadata every 20 images, and on the final write
        if len(metadata_list) > 20 or successes == n - 1:
            df = pd.DataFrame(metadata_list)
            # Sort the keys lexographically to ensure consistent writes
            df = df.reindex(sorted(df.columns), axis=1)
            first_write = successes <= len(metadata_list)
            df.to_csv(
                metadata_path,
                index=None,
                mode="w" if first_write else "a",
                header=first_write,
            )
            metadata_list = []

        successes += 1
        if not quiet:
            pbar.update()

    # Make sure the list has been cleared out.
    assert not metadata_list

    if not quiet:
        pbar.close()
        print("Dataset generation complete. Acceptance rate: %.3f" % (n / tries))

    # Generate tf record if requested. Save all the parameters and use default
    # filename data.tfrecord
    if tf_record:
        # Delayed import, triggers tensorflow import
        from paltas.Analysis import dataset_generation

        # The path to save the TFRecord to.
        tf_record_path = os.path.join(save_folder,'data.tfrecord')
        # Generate the list of learning parameters. Only save learning
        # parameters with associated float values.
        learning_params = []
        for key in metadata:
            if (isinstance(metadata[key],float) or
                isinstance(metadata[key],int)):
                learning_params.append(key)
        # Generate the TFRecord
        dataset_generation.generate_tf_record(save_folder,learning_params,
            metadata_path,tf_record_path)



config_header = """\
from copy import deepcopy
import sys
sys.path.append('{CONFIG_PATH}')
from {CONFIG_NAME} import *

config_dict = deepcopy(config_dict)
"""

@export
def generate_with_new_config(
    new_config: dict = None,
    n_images: int = 5,
    base_config=bb.DEFAULT_TRAINING_CONFIG,
    output_folder=DEFAULT_DATASET.parent,
    overwrite=False,
    config_folder: str = None,
    quiet=False,
):
    """Generates a paltas dataset with a changed config and returns its name.

    Results are saved in the current directory, in the following folders:
        - Config: paltas_datasets/DATASET_NAME.py
        - Images: paltas_datasets/DATASET_NAME  (/image_000000.npy, etc)

    Arguments:
        new_config: dict with config options to update.
            Use / to indicate taking a key. For example:
                {'subhalo/parameters/sigma_sub': 0.,
                 'main_deflector/parameters/theta_E': 1.}
            makes lenses with sigma_sub=0 and theta_E=1.
            You can also use use shortened param names:
                dict(sigma_sub=0, theta_E=1)

        n_images: number of images to generate
        base_config: path to paltas config py file for base settings.
        output_folder: path to place the DATASET_NAME folder in.
            Defaults to ./paltas_datasets
        config_folder: path to place the DATASET_NAME.py config file in.
            If not provided, config is deleted after generation. A copy
            will still be found in the image folder.
        overwrite: if True, overwrite existing dataset folder

    """
    if new_config is None:
        new_config = dict()
    # Translate short parameter names to long ones with slashes.
    new_config = {bb.PALTAS_SLASH_NAMES.get(k, k): v
                  for k, v in new_config.items()}

    # Construct dataset_name
    dataset_name = base_config.stem
    # Remove "config", and make sure we start with a character
    # (so the dataset is a valid python module name)
    if dataset_name.startswith("config_"):
        dataset_name = dataset_name[len("config_") :]
    if dataset_name[0].isdigit():
        dataset_name = "_" + dataset_name
    # Add safe versions of the key/value pairs
    for param_code, param_value in new_config.items():
        dataset_name += "_" + "_".join(
            [re.sub("[^a-zA-Z0-9]", "_", x) for x in (param_code, str(param_value))]
        )
    # Ensure dataset is <100 characters, and add a random suffix for uniqueness
    dataset_name = (
        dataset_name[:92] + "_" + "".join(random.choices(string.ascii_lowercase, k=8))
    )

    # Create output folders
    output_folder = Path(output_folder) / dataset_name
    if output_folder.exists():
        if overwrite:
            if not quiet:
                print(f"Removing existing dataset folder {output_folder}")
            shutil.rmtree(output_folder)
        else:
            raise FileExistsError(f"Dataset already exists at {output_folder}, "
                                  "and overwrite=False")
    else:
        output_folder.mkdir(parents=True)
    if config_folder is None:
        delete_config = True
        # Put the config in the parent folder of the images
        # (paltas's generate will also place it in the image folder)
        config_folder = output_folder.parent
    else:
        delete_config = False
        config_folder = Path(config_folder)

    # Create config and write it to a .py file
    config = config_header.format(
        CONFIG_PATH=str(base_config.parent), CONFIG_NAME=base_config.name.split(".")[0]
    )
    for param_code, param_value in new_config.items():
        config += (
            "\nconfig_dict['" + param_code.replace("/", "']['") + f"'] = {param_value}"
        )
    config_fn = config_folder / f"{dataset_name}.py"
    with open(config_fn, mode="w") as f:
        f.write(config)

    # Generate images
    if not quiet:
        print(f"Generating images from {config_fn}\n\n")

    generate_dataset(
        config_path=str(config_fn),
        save_folder=str(output_folder),
        n=n_images,
        save_png_too=False,
        tf_record=False,
        quiet=quiet,
    )
    if delete_config:
        os.remove(config_fn)
    return dataset_name


@export
def generate_image(
        new_config: dict = None,
        base_config=bb.DEFAULT_TRAINING_CONFIG):
    """Returns (image array, metadata dict) from generating
    a single image using new_config on top of base_config.

    Note image is returned without normalization; the March 2022 paper
    neural net expects the images to be divided by their std.

    Arguments:
        new_config: dict with config options to update.
            Use / to indicate taking a key. For example:
                {'subhalo/parameters/sigma_sub': 0.,
                 'main_deflector/parameters/theta_E': 1.}
            makes lenses with sigma_sub=0 and theta_E=1.
            You can also use use shortened param names:
                dict(sigma_sub=0, theta_E=1)

        base_config: path to paltas config py file for base settings.
    """
    with tempfile.TemporaryDirectory() as temp_folder:
        temp_folder = Path(temp_folder)
        dsetname = generate_with_new_config(
            new_config=new_config,
            n_images=1,
            base_config=base_config,
            output_folder=temp_folder,
            quiet=True,
        )
        image = np.load(temp_folder / dsetname / f'image_{0:07d}.npy')
        metadata = pd.read_csv(
            temp_folder / dsetname / "metadata.csv").iloc[0].to_dict()

    return image, metadata
