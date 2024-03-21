from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="contextualize",
    version="0.0.2",
    packages=find_packages(),
    install_requires=required,
    entry_points={"console_scripts": ["contextualize = contextualize.cli:main"]},
    author="jmpaz",
    description="LLM prompt/context preparation utility ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jmpaz/contextualize",
    python_requires=">=3.6",
)
