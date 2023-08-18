from pathlib import Path

import numpy as np
import pandas as pd
import paltas
import tensorflow as tf
from tqdm import tqdm

import bendorbreak as bb
export, __all__ = bb.exporter()


@export
def single_image_test(model, img, truth=None, norms_csv=bb.DEFAULT_NORMS_CSV):
    """Return pandas DataFrame with pretty results from passing a single
    image through the model.

    Arguments:
      model: keras neural network
      img: (nx, ny) numpy array, input image
      truth: dictionary with truth values
      norms_csv: path to paltas norms csv

    """
    param_names, param_means, param_scales = bb.parse_norms_csv(norms_csv)
    result = model.predict(img[None,...,None] / img.std())
    prediction = dict(zip(
        param_names,
        bb.denormalize(result, param_means, param_scales)[0]))
    report = dict(
        prediction=prediction,
        prior_mean={p: param_means[i] for i, p in enumerate(param_names)},
        prior_scale={p: param_scales[i] for i, p in enumerate(param_names)},
    )
    if truth is not None:
        report['truth'] = {p: truth[p] for p in param_names}
        report['error_sigma'] = {
            p: (prediction[p] - truth[p])/param_scales[i]
            for i, p in enumerate(param_names)}
    report = pd.DataFrame(report)
    report['parameter'] = [
        bb.SHORT_NAMES.get(p, p)
        for p in report.index.values]
    return report.set_index('parameter')


@export
def run_network_on(
        folder,
        model=None,
        norm_path=bb.DEFAULT_NORMS_CSV,
        train_config_path=bb.DEFAULT_TRAINING_CONFIG,
        batch_size=50,
        n_rotations=1,
        regenerate_tfrecord=False,
        overwrite=False,
        return_result=False,
        output_filename='network_outputs.npz',
        save_penultimate=False,
        params_as_inputs=None,
        loss_type='full'):
    """Run a neural network over a folder with image data.
    Creates output_filename with return values in that folder.

    Arguments:
     - folder: path to folder with images, metadata, etc.
     - norm_path: path to norms.csv for the network
     - model: neural network, or path to h5 of it
     - train_config_path: path to the dataset used to train
         the network.
     - batch_size: batch size.
     - n_rotations: if > 1, average point estimates (not covariances)
         over rotations of the image, uniform(0, 360, n_rotations)
     - regenerate_tfrecord: If True, regererates tfrecord file
         even if it is already present.
     - overwrite: If True, first delete existing network_outputs.npz
         if present
     - return_result: If True, also returns stuff saved to the npz
     - save_penultimate: if True, runs the model twice; the second time
         just to save the output of the penultimate layer.
     - params_as_inputs: Sequence of parameters used as inputs to fully
         connected layer.
     - loss_type: loss function the network was trained with;
        'full' or 'diagonal'.
    """
    # Check input/output folders
    folder = Path(folder)
    assert folder.exists()
    output_path = folder / output_filename
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            print(f"Already ran network on {folder}, nothing to do")
            return

    # Load and process metadata
    metadata_path = folder / 'metadata.csv'
    df = pd.read_csv(metadata_path, index_col=False)
    # Remove silly columns from metadata, saves space
    for prefix, col_name in paltas.Configs.config_handler.EXCLUDE_FROM_METADATA:
        col_name = prefix + '_' + col_name
        if col_name in df:
            del df[col_name]
    # Add dataset name and image number
    # just in case things get mixed around again
    df['dataset_name'] = folder.name
    df['image_i'] = np.arange(len(df))

    # Load normalization / parameter names and order
    norm_df = pd.read_csv(norm_path)
    all_params = norm_df.parameter.values.tolist()

    # Separate input params (eg redshift) from output parameters (eg theta_E)
    if params_as_inputs:
        params_as_inputs = [x for x in all_params if x in params_as_inputs]
        output_params = [x for x in all_params if x not in params_as_inputs]
    else:
        params_as_inputs = None
        output_params = all_params

    if model is None:
        model = bb.march2022_paper_model()
        model_name = 'march_2022_paper'
    elif isinstance(model, (str, Path)):
        model = tf.keras.models.load_model(
            Path(model),
            custom_objects=dict(loss=None))
        model_name = Path(model).name
    else:
        model_name = 'unknown'

    if save_penultimate:
        # Model with the fully-connected head removed
        # (for our model, that's just one layer)
        # TODO: if architecure changes, have to change the index here
        model_conv = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(index=-2).output)

    # Extract training and test/population mean and cov
    training_population_mean, training_population_cov = bb.paltas_mean_cov(
        train_config_path, output_params)
    population_mean, population_cov = bb.paltas_mean_cov(
        folder, output_params)

    # Ensure we have the tfrecord dataset
    tfr_path = folder / 'data.tfrecord'
    if tfr_path.exists() and regenerate_tfrecord:
        tfr_path.unlink()
    if not tfr_path.exists():
        paltas.Analysis.dataset_generation.generate_tf_record(
            npy_folder=str(folder),
            learning_params=all_params,
            metadata_path=str(metadata_path),
            tf_record_path=str(tfr_path))

    # Construct the paltas dataset generator
    def make_dataset():
        test_dataset = paltas.Analysis.dataset_generation.generate_tf_dataset(
            tf_record_path=str(tfr_path),
            learning_params=all_params,
            batch_size=batch_size,
            n_epochs=1,
            # NB: The npy images are stored on disk unnormalized!
            norm_images=True,
            input_norm_path=norm_path,
            kwargs_detector=None,  # Don't add more noise
            log_learning_params=None,
            shuffle=False)
        if params_as_inputs:
            test_dataset = paltas.Analysis.dataset_generation.generate_params_as_input_dataset(
                test_dataset, params_as_inputs, all_params)
        return test_dataset

    test_dataset = make_dataset()

    if n_rotations == 0:
        # Should do the same as n_rotations=1, retained for testing.
        image_mean, image_prec = _predict(
            model, test_dataset, output_params, loss_type=loss_type)
        image_cov = np.linalg.inv(image_prec)

        # Convert to physical units. Modifies image_xxx variables in-place
        paltas.Analysis.dataset_generation.unnormalize_outputs(
            input_norm_path=norm_path,
            learning_params=output_params,
            mean=image_mean,
            cov_mat=image_cov,
            prec_mat=image_prec,
        )

    else:
        # Compute predictions over several angles
        # (note we skip 2 pi since it's equivalent to zero)
        means, covs = [], []
        for angle in tqdm(np.linspace(0, 2 * np.pi, n_rotations + 1)[:-1],
                        desc='Running neural net over different rotations'):
            # Get predictions on rotated dataset
            _mean, _prec = _predict(
                model,
                _rotation_generator(test_dataset, output_params, angle),
                output_params,
                loss_type=loss_type)

            # Recover covariance: rotation of precision matrix not yet coded
            _cov = np.linalg.inv(_prec)

            # Convert to physical units. Modifies image_xxx variables in-place
            # NB: must do this before back-rotation!
            paltas.Analysis.dataset_generation.unnormalize_outputs(
                input_norm_path=norm_path,
                learning_params=output_params,
                mean=_mean,
                cov_mat=_cov,
            )

            # Rotate back to original frame. Modifies in-place
            paltas.Analysis.dataset_generation.rotate_params_batch(
                output_params, _mean, -angle)
            paltas.Analysis.dataset_generation.rotate_covariance_batch(
                output_params, _cov, -angle)

            means.append(_mean)
            covs.append(_cov)
        means, covs = np.array(means), np.array(covs)

        # Average predictions obtained from different rotation angles
        image_mean = np.mean(means, axis=0)
        # Paltas paper says: covariances, and image predictions for x_lens and
        # y_lens, are not averaged over rotations.
        for param in (bb.PALTAS_NAMES['center_x'], bb.PALTAS_NAMES['center_y']):
            if param in output_params:
                i = output_params.index(param)
                image_mean[:,i] = means[0,:,i]
        image_cov = covs[0]
        image_prec = np.linalg.inv(image_cov)

    # Enforce symmetry on the precision and covariance matrices.
    # Floating-point errors could otherwise spoil this and make the entire
    # inference return -inf. Fun!
    image_cov = bb.symmetrize_batch(image_cov)
    image_prec = bb.symmetrize_batch(image_prec)

    if save_penultimate:
        # Run the model again, saving the output of the penultimate layer
        # A bit wasteful to run it twice, but OK...
        conv_outputs = model_conv.predict(make_dataset())
    else:
        conv_outputs = None

    # Save everything needed for inference to a big npz
    result = dict(
        image_mean=image_mean,
        image_cov=image_cov,
        image_prec=image_prec,
        # OK, image truths are not needed for inference. But why not save them..
        image_truth=df[output_params].values,
        population_mean=population_mean,
        population_cov=population_cov,
        training_population_mean=training_population_mean,
        training_population_cov=training_population_cov,
        # Copy over parameter names/orders for interpretability
        param_names=output_params,
        # These are a bit large, but who cares: just put everything in one file...
        convout=conv_outputs,
        metadata=df.to_records(index=False),
        model_name=model_name,
    )

    np.savez_compressed(output_path, **result)
    if return_result:
        return result


def _predict(model, dataset, learning_params, loss_type='full'):
    # Finally! Actually run the network over the images
    result = bb.parse_paltas_network_output(
        model.predict(dataset),
        learning_params, loss_type)
    return [x.numpy() for x in result]


def _rotation_generator(dataset, learning_params, angle):
    for images, truths in dataset:
        if isinstance(images, (list, tuple)):
            # We got an images, params_input list/tuple
            images, input_params = images
        else:
            input_params = None
        if not isinstance(images, np.ndarray):
            images = images.numpy()
        if not isinstance(truths, np.ndarray):
            truths = truths.numpy()

        images = paltas.Analysis.dataset_generation.rotate_image_batch(
            images,
            learning_params,
            # NB truths is changed in-place!
            truths,
            angle)

        if input_params is not None:
            # Restore the input params unchanged
            images = [images, input_params]
        yield images, truths
