from functools import partial

import alibi
import numpy as np
import tensorflow as tf
# TensorFlow-Addons is being deprecated -- I searched for an tensorflow-native
# rotation function in a maintained library, but could not find any. :-(
# See https://github.com/keras-team/keras-cv/issues/402
# Could adapt the code in keras/layers/preprocessing/image_preprocessing.py
# which does random rotation.
import tensorflow_addons as tfa

import bendorbreak as bb
export, __all__ = bb.exporter()


@export
def integrated_gradients_map(
        img, model, param,
        norms_path=bb.DEFAULT_NORMS_CSV,
        n_rotations=1, batch_size=None):

    # Paltas neural nets (at least Sebastian's) expect img to be normalized
    img_norm = img / img.std()

    # Create n_rotations copies of images along a new axis 0
    # The network will do the actual rotations for us.
    img_batch = img_norm[None,...] * np.ones((n_rotations, 1, 1))
    rotation_angles = np.linspace(0, 2 * np.pi, n_rotations + 1)[:-1]

    attr_model = single_output_rotating_model(
        model, param, norms_path=norms_path)
    ig = alibi.explainers.IntegratedGradients(
        attr_model,
        internal_batch_size=n_rotations if batch_size is None else n_rotations)

    # TODO: find a way to suppress the repetitive warning about using a
    # regression model.
    # It's probably a print statement... redirect sys.stdout? really? ugh.
    expl_batch = ig.explain([
            img_batch[...,None],
            rotation_angles
        ]).data['attributions'][0]

    # Return mean saliency map
    return expl_batch.mean(axis=0)[...,0]


@export
def single_output_rotating_model(
        model, output_param, norms_path=bb.DEFAULT_NORMS_CSV):
    """Return keras model that takes a batch of images and a rotation angles,
    and outputs a single parameter. The network is applied to the rotated image,
    and the outputs are rotated back.
    """
    param_names, means, scales = bb.parse_norms_csv(norms_path)

    image_input = tf.keras.Input(shape=(None, None, 1), name='image_input')
    angle_input = tf.keras.Input(shape=(), name='angle_input')

    # Rotate the image
    #
    # TODO: paltas rotates images with scipy.ndimage.rotate, which uses
    # a third-order spline by default. Unfortunately tfa.image.rotate only
    # supports nearest-neighbor (bad) and bilinear (order=1 in scipy.ndimage).
    # Bilinear is about 5x closer to ndimage.rotate than nearest, but still not
    # perfect.
    def _rotate_img(inputs):
        img, rot_angle = inputs
        return tfa.image.rotate(img, rot_angle, interpolation='bilinear')
    rotated_image = tf.keras.layers.Lambda(_rotate_img)([image_input, angle_input])

    # Apply the neural net
    predictions = model(rotated_image)

    # Apply postprocessing (denormalization, rotating back)
    denormalize = partial(
        bb.denormalize,
        means=tf.convert_to_tensor(means, dtype=tf.float32),
        scales=tf.convert_to_tensor(scales, dtype=tf.float32))
    output_i = param_names.index(bb.PALTAS_NAMES.get(output_param, output_param))
    class PostProcessing(tf.keras.layers.Layer):
        def call(self, inputs):
            outputs, angles = inputs
            # Select the first len(param_names) outputs, which are normalized predictions of the parameters.
            outputs = outputs[:, :len(param_names)]
            # Restore physical units
            outputs = denormalize(outputs)
            # Rotate the predictions back to the original frame
            outputs = bb.rotate_output(outputs, -angles, param_names)
            # Select the desired parameter
            return outputs[:,output_i]
    predictions = PostProcessing()([predictions, angle_input])

    return tf.keras.Model(
        inputs=[image_input, angle_input], outputs=predictions)


# Seems to work: np.pi does right think on x&y and leaves others unchanged
@export
@tf.function
def rotate_output(x, rot_angle, params):
    """Rotate truths/predictions of parameters

    Arguments:
      x: (n_images, n_params) tensor of truths or predictions
      rot_angle: angle to rotate by, in _radians_
      params: list of n_params parameter names
    """
    # Convert from big tensor to dict of per-parameter tensors
    out = {
        param: x[...,i]
        for i, param in enumerate(params)}
    # Update entries for params to be rotated
    _get_rotated(x, out, 'center_x', 'center_y', params, rot_angle)
    _get_rotated(x, out, 'e1', 'e2', params, 2 * rot_angle)
    _get_rotated(x, out, 'gamma1', 'gamma2', params, 2 * rot_angle)
    # Convert back to a big tensor again
    return tf.stack([
        out[param]
        for param in params
    ], axis=-1)


def _get_rotated(x, out, xname, yname, params, angle):
    prefix = 'main_deflector_parameters_'
    xname = prefix + xname
    yname = prefix + yname
    if xname in params:
        if yname not in params:
            raise ValueError(f"Missing {yname} paired with {xname}")
        out[xname], out[yname] = \
            _rotate(x[:,params.index(xname)], x[:,params.index(yname)], angle)


def _rotate(x, y, theta):
    # Note the angle is flipped from the usual equations, since lenstronomy has
    # x and y mixed up -- x is vertical, y is horizontal.
    # Paltas uses this minus sign when it rotates the images:
    # https://github.com/swagnercarena/paltas/blob/2edd7f418a63273d5b2fcc75819e811bceb1f149/paltas/Analysis/dataset_generation.py#L440
    theta = - theta
    return (
        x * tf.cos(theta) - y * tf.sin(theta),
        x * tf.sin(theta) + y * tf.cos(theta))
