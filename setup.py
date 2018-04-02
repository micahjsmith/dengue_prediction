#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'btb==0.2.0',
    'category_encoders',
    'funcy',
    'numpy',
    'pandas',
    'scikit_learn',
    'sklearn_pandas',
]

setup_requirements = [
    'pytest-runner',
    # TODO(micahjsmith): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    'pytest',
    # TODO: put package test requirements here
]

extras_require = {
    #'btb': ['btb']
}

dependency_links= [
    'git+https://github.com/HDI-Project/BTB.git@30cc8d1affea2b37771eb865b9df30dc9d0657e8#egg=btb-0.2.0',
]

setup(
    name='dengue_prediction',
    version='0.1.0',
    description="",
    long_description=readme + '\n\n' + history,
    author="Micah Smith",
    author_email='micahjsmith@gmail.com',
    url='https://github.com/micahjsmith/dengue_prediction',
    packages=find_packages(include=['dengue_prediction']),
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
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='dengue_prediction/tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
    extras_require=extras_require,
)
