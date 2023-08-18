#!/usr/bin/env python
import os
from pathlib import Path
import shutil

import numpy as np

import bendorbreak as bb
export, __all__ = bb.exporter()


@export
def quicktest(
    *param_value_pairs,
    n_images: int = 5,
    n_rotations: int = 32,
    config_path: str = bb.DEFAULT_TRAINING_CONFIG,
    model: str = None,
    norm_path: str = bb.DEFAULT_NORMS_CSV,
    n_mcmc_samples: int = 1000,
    cleanup_results: bool = False,
):
    """Run an end-to-end test of paltas image generation and analysis.

    This script will:
     - Create a configuration that differes by one parameter value
       from the reference config;
     - Generate and saves test set images with that config;
     - Run a neural network over that config, saving results;
     - Run Bayesian MCMC and asymptotic frequentist hierarchical inference,
       saving and printing results.

    Results are saved in the current directory, in the following folders:
     - Config: paltas_datasets/DATASET_NAME.py
     - Images: paltas_datasets/DATASET_NAME  (/image_0000000.npy etc)
     - Network outputs: quicktest_results/DATASET_NAME_network_outputs.npz
     - Inference: quicktest_results/DATASET_NAME_inference_results.npz

    Args:
        param_value_pairs: Flat list with parameter/value pairs to set.
            Omit to just run one config. When naming parameters, use / to
            indicate taking a key, e.g. subhalo/parameters/sigma_sub.
        param_value: value you wish the parameter to take
        n_images: number of images to generate
        n_rotations: average network predictions over n_rotations image
            rotations.
        config_path: path to paltas config py file for base settings.
            Uses training set config if not provided.
        model_path: path to neural network h5 file. If not provided,
            use xresnet34_full_final.h5 in current dir; download as-needed.
        norm_path: path to norms.csv file. Defaults to norms.csv from
            the paper_2203_00690 folder.
        n_mcmc_samples: number of MCMC samples to do
            (excluding 1k burn-in samples)
        cleanup_results: delete all created files after a successful run.
            Deletes paltas_datasets and quicktest_results if these are empty.
    """
    # Delayed import to speed up --help call
    import paltas.Analysis

    config_path = Path(config_path)

    if model is None:
        model = bb.march2022_paper_model()
    if not Path(norm_path).exists():
        raise FileNotFoundError(f"Norms csv file {norm_path} not found")

    # Convert flattened list of keys and values into dict
    param_value_pairs = list(param_value_pairs)
    if len(param_value_pairs) % 2 != 0:
        raise ValueError("Provide an even number of parameter/value pairs.")
    new_config = dict([
        (param_value_pairs[i], param_value_pairs[i + 1])
        for i in range(0, len(param_value_pairs), 2)
    ])

    base_folder = bb.DEFAULT_DATASET.parent
    dataset_name = bb.generate_with_new_config(
        new_config,
        base_config=config_path,
        n_images=n_images,
        # Not really needed, that's the default anyway, but OK.
        output_folder=base_folder,
    )
    dataset_folder = base_folder / dataset_name

    # Run neural network, saving results in the dataset folder
    print(f"\n\nRunning neural network on {dataset_folder}\n\n")
    bb.run_network_on(
        dataset_folder,
        norm_path=norm_path,
        model=model,
        batch_size=min(n_images, 50),
        n_rotations=n_rotations,
        save_penultimate=False,
    )

    # Copy results to results folder
    results_folder = Path("./quicktest_results")
    results_folder.mkdir(exist_ok=True)
    network_outputs_fn = results_folder / f"{dataset_name}_network_outputs.npz"
    shutil.copy(src=dataset_folder / "network_outputs.npz", dst=network_outputs_fn)

    # Run inference
    print(f"\n\nRunning final inference\n\n")
    inf = bb.GaussianInference.from_folder(dataset_folder)
    freq_summary, freq_cov = inf.frequentist_asymptotic()
    bayes_summary, chain = inf.bayesian_mcmc(n_samples=n_mcmc_samples)

    # Combine results into one dataframe
    summary = freq_summary[["param", "truth"]].copy()
    for df, code in ((freq_summary, "maxlh"), (bayes_summary, "mcmc")):
        summary[f"{code}_fit"] = df["fit"]
        summary[f"{code}_fit_unc"] = df["fit_unc"]

    # Save and print inference results
    inference_results_fn = results_folder / f"{dataset_name}_inference_results.npz"
    np.savez(
        inference_results_fn,
        summary=summary.to_records(),
        freq_cov=freq_cov,
        chain=chain,
    )
    print(f"\n\nDone!! :-)\n\n")
    print(f"RESULTS:\n")
    try:
        print(summary.to_markdown(index=False))
    except ImportError:
        print(summary)
        print("\n\nFor prettier result prints, pip install tabulate.")

    if cleanup_results:
        # Remove files we just made
        shutil.rmtree(dataset_folder)
        for fn in [inference_results_fn, network_outputs_fn]:
            os.remove(fn)
        # Cleanup base folders if they are empty
        for big_folder in [base_folder, results_folder]:
            contents = os.listdir(big_folder)
            print(big_folder, contents)
            if contents in ([], ["__pycache__"]):
                shutil.rmtree(big_folder)

    return summary


if __name__ == "__main__":
    bb.make_cli(quicktest)
