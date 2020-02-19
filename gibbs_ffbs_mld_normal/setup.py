import setuptools

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
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)