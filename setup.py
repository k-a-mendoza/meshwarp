from setuptools import setup, find_packages

setup(
    name='meshwarp',
    version='1.0',
    packages=find_packages(exclude=['tests*']),
    license='none',
    description='a module for warping meshes',
    install_requires=[],
    author='Kevin A Mendoza',
    author_email='kevinmendoza@icloud.com')