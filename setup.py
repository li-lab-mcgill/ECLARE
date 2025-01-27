from setuptools import setup, find_packages

setup(
    name='ATAC_RNA_triplet_loss_align',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        # 'numpy',
        # 'pandas',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            # If you want to create command-line scripts, specify them here
            # 'command_name=module:function',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your package',
    url='https://github.com/yourusername/yourrepository',  # if you have a URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # specify your Python version requirement
)