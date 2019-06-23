import setuptools, os, codecs, re

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().strip().split('\n')
here = os.path.abspath(os.path.dirname(__file__))

def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ *= *['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="ddpg_agent",
    version=find_version("ddpg_agent", "__init__.py"),
    author="Sam Hiatt",
    author_email="samhiatt@gmail.com",
    license='MIT',
    description="Reinforcement Learning model using Deep Deterministic Policy Gradients (DDPG)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/samhiatt/ddpg_agent",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
