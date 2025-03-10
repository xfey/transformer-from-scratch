from setuptools import setup, find_packages

setup(
    name="transformer-from-scratch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
    ],
    author="xfey",
    author_email="xfey99@gmail.com",
    description="A step-by-step tutorial for implementing Transformer architectures",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/xfey/transformer-from-scratch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
) 