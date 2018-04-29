#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''The setup script.'''

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'btb',
    'category_encoders',
    'fhub_core>=0.3.1',
    'fhub_transformers>=0.2.4',
    'funcy',
    'gitpython',
    'numpy',
    'pandas;python_version>="3.5"',  # hack
    'pandas<0.21;python_version<"3.5"',  # hack
    'scikit_learn',
    'sklearn_pandas',
]

setup_requirements = [
    'pytest-runner',
]

test_requirements = [
    'pytest',
    'pytest-runner',
    'coverage',
]

extras_require = {
    #'btb': ['btb']
}

dependency_links= [
    'git+https://github.com/micahjsmith/BTB.git@48db0da94a28492220ca666de80e193c50d756f6#egg=btb-0.2.0',
]

setup(
    name='dengue_prediction',
    version='0.1.0',
    description="",
    long_description=readme + '\n\n' + history,
    author="Micah Smith",
    author_email='micahjsmith@gmail.com',
    url='https://github.com/micahjsmith/dengue_prediction',
    packages=find_packages(
        include=['dengue_prediction', 'dengue_prediction.*']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='dengue_prediction',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='dengue_prediction/tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    extras_require=extras_require,
)
