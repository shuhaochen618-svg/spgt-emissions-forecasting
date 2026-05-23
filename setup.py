from setuptools import setup, find_packages

setup(
    name="spgt",
    version="1.0.0",
    description="Spatiotemporal Patch Graph Transformer (SPGT) for Multi-Sectoral Daily Carbon Emissions Forecasting",
    author="Carbon Dynamics Research Team",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
