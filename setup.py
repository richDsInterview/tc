# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tc',
    version='0.1.0',
    description='script to solve a technical challenge on image identification',
    long_description=readme,
    author='Richard Armstrong',
    author_email='armstrong.richard@gmail.com',
    url='https://github.com/richDsInterview/',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'venv'))
)

