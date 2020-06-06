from setuptools import setup, find_packages


setup(
    name='simple-neuralnet-python',
    version='1.1.0',
    description='Simple MLP neural network',
    author='Antti Juvonen',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['numpy','matplotlib', 'attrs'],
    tests_require=['pytest'],
    setup_requires=['pytest-runner'],
    scripts=['bin/neuralnet-sigmoid-plotter'],
    entry_points = {
        'console_scripts': [
            'neuralnet-run=simple_neuralnet_python.neuralnet:main'
            ]
    }
)
