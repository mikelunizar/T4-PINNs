from setuptools import setup, find_packages

setup(
    name='pde_solver',            # Replace with your package name
    version='1.1.0',
    author='Mikel M Iparraguirre',
    author_email='mikel.martinez@unizar.es',
    description='PDE solver based on PINNs Neural Networks in Lightning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mikelunizar/T4-PINNs.git',  # Replace with your repo URL
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[
        # Add your required dependencies here
        'torch',
        'numpy',
        'matplotlib',
        'pytorch-lightning',
        'wandb',
        'matplotlib',
        'scikit-learn',
        'imageio',
        'jupyter',
        'torchvision',
        'PIL',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
