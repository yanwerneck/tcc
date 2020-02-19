from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yanwerneck",
    version="0.0.1",
    author="Yan Werneck",
    author_email="yan_werneck@yahoo.com.br",
    description="Dinamic Linear Model With Normal Distribution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yanwerneck/tcc",
    install_requires=[
        "numpy==1.18.1",
        "pandas==1.0.1",
        "scipy==1.4.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires='>=3.6',
)