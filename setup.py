import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().strip().split('\n')

setuptools.setup(
    name="ddpg_agent",
    version="0.0.1",
    author="Sam Hiatt",
    author_email="samhiatt@gmail.com",
    license='MIT',
    description="Reinforcement Learning model using Deep Deterministic Policy Gradients (DDPG)",
    long_description=long_description,
    url="https://github.com/samhiatt/ddpg_agent",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
