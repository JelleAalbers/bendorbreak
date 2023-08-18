import setuptools

with open('README.md') as file:
    readme = file.read()

with open('HISTORY.md') as file:
    history = file.read()

def get_requirements(path='requirements.txt', strict=False):
    """Return a list of requirements from a requirements file.
    
    Arguments:
     - strict: If False, strip version tags
    """
    with open(path, mode='r') as f:
        requirements = f.read().splitlines()
    if not strict:
        requirements = [x.split('==')[0] for x in requirements]
    return requirements

requirements = get_requirements()
requirements_strict = get_requirements()

setuptools.setup(
    name='bendorbreak',
    version='0.0.0',
    description='Robust strong lensing inference with neural networks',
    author='Jelle Aalbers',
    url='https://github.com/JelleAalbers/bendorbreak',
    python_requires=">=3.6",
    include_package_data=True,
    setup_requires=['pytest-runner'],
    install_requires=requirements,
    tests_require=['pytest'],
    extras_require={
        'docs': ['sphinx',
                 'sphinx_rtd_theme',
                 'nbsphinx',
                 'myst-parser',
                 'jax[cpu]'],
        'strict-deps': requirements_strict,
    },
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Physics'],
    zip_safe=False)
