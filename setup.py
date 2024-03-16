from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="contextualize",
    version="0.0.1",
    packages=find_packages(),
    install_requires=required,
    entry_points={"console_scripts": ["contextualize = contextualize.cli:main"]},
    author="jmpaz",
    description="LLM prompt/context preparation utility ",
    url="https://github.com/jmpaz/contextualize",
)
