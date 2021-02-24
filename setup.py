from setuptools import find_packages, setup
setup(
    name='flexlearn',
    packages=find_packages(include=['flexlearn']),
    version='0.1.0',
    description='a way to structure your deep learning experiments, models, datasets, results.',
    author='Jan Tuitman',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)