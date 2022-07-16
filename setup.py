from setuptools import find_namespace_packages, setup
import os.path as osp

with open("prl/environment/steinberger/README.md", "r") as fh:
    long_description = fh.read()

with open('%s/%s' % (osp.dirname(osp.realpath(__file__)), 'requirements.txt')) as f:
    requirements = [line.strip() for line in f]

with open('%s/%s' % (osp.dirname(osp.realpath(__file__)), 'requirements_dist.txt')) as f:
    requirements_dist = [line.strip() for line in f]

setup(
    name="prl_environment",
    version="0.0.3",
    author="hellovertex",
    author_email="hellovertex@outlook.com",
    description="Poker Environment from Eric Steinberger wrapped with additional functionality.",
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hellovertex/PokerRL",
    install_requires=requirements,
    extras_require={
        'distributed': requirements_dist,
    },
    package_data={'': ['**/*.so', '**/*.dll']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    packages=find_namespace_packages(),
)
