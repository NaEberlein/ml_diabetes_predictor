
from setuptools import setup, find_packages

setup(
    name="ml_diabetes_predictor",  # Project name
    version="0.1",  # Initial version
    packages=find_packages(where="src"),  # Automatically find all packages in the 'src' directory
    package_dir={"": "src"},  # Specify the base directory for your packages
    install_requires=[  # List of dependencies
        "numpy<2",  
        "pandas",  
        "matplotlib==3.10.1",  
        "seaborn",
        "scikit-learn", 
        "shap", 
        "pytest"
    ],
)
