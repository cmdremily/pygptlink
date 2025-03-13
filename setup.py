from setuptools import setup, find_packages

setup(
    name='pygptlink',
    version='0.1.0',
    description='A Python framework for easily integrating with OpenAI and LMStudio (WIP) LLMs.',
    author='Your Name',
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.11",
    include_package_data=True,
    package_data={"pytils": ["py.typed"]},
)
