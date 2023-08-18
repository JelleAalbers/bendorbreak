import argparse
import inspect
from pathlib import Path
import requests
import shutil

import docstring_parser
import numpy as np


def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_


export, __all__ = exporter(export_self=True)


@export
def make_cli(main_f):
    """Wrap main_f in an argparse command-line interface

    Docstring and type annotations determine argument parsing and help message.

    Un-annotated arguments will be passed as strings.
    """
    # Get the signature, used for getting argument names and types
    signature = inspect.signature(main_f).parameters

    # Parse the docstring to get the argument descriptions
    doc = docstring_parser.parse(main_f.__doc__)
    descs = {x.arg_name: x.description for x in doc.params}

    # Auto-generate and and run an argparse parser
    # There are libraries that do this, but I haven't found one that parses
    # the docstring for argument descriptions.
    desc = doc.short_description
    if doc.long_description:
        desc += "\n\n" + doc.long_description
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=desc
    )
    for pname, param in signature.items():
        kwargs = dict(help=descs.get(pname, None))
        has_default = param.default != param.empty
        if has_default and param.annotation is bool:
            # Flag argument. Note true/false inversion, gotta love argparse
            if param.default is True:
                kwargs["action"] = "store_false"
            elif param.default is False:
                kwargs["action"] = "store_true"
            else:
                raise ValueError("flag args should default to true or false")
        else:
            # Regular argument
            if has_default:
                kwargs["default"] = param.default
            if param.annotation != param.empty:
                kwargs["type"] = param.annotation
            if param.kind == param.VAR_POSITIONAL:
                kwargs["nargs"] = "*"
        parser.add_argument(("--" if has_default else "") + pname, **kwargs)

    args = parser.parse_args()

    # If there is a *args, we need to call the main function differently
    (var_pos_name,) = [
        pname
        for pname, param in signature.items()
        if param.kind == param.VAR_POSITIONAL
    ][:1] or (None,)
    if var_pos_name:
        var_pos = getattr(args, var_pos_name)
        if not var_pos:
            var_pos = tuple()
        main_f(
            *var_pos,
            **{
                pname: getattr(args, pname)
                for pname in signature.keys()
                if pname != var_pos_name
            }
        )
    else:
        main_f(**vars(args))


@export
def symmetrize_batch(x):
	"""Return symmetrized version of an array of matrices"""
	return (x.transpose((0, 2, 1)) + x)/2


@export
def cov_to_std(cov):
	"""Return (std errors, correlation coefficent matrix)
	given covariance matrix cov
	"""
	std_errs = np.diag(cov) ** 0.5
	corr = cov * np.outer(1 / std_errs, 1 / std_errs)
	return std_errs, corr


# Works: tested against paltas' unnormalize
@export
def denormalize(x, means, scales):
    x = x[:,:len(means)]
    return x * scales[None,:] + means[None,:]


@export
def download_and_save(url: str, path: Path):
    """Download a file from url and save it to path

    Arguments:
        url {str} -- URL to download from
        path {pathlib.Path} -- Path to save to
    """
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
