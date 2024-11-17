from setuptools import setup, find_packages

setup(
    name='MCRT', 
    version='1.0.0',  
    packages=find_packages(),  
    description='Molecular Crystal Representation from Transformer', 
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',  
    author='Minggao Feng', 
    author_email='ffmmgg@liverpool.ac.uk', 
    url='https://github.com/fmggggg/MCRT',  
    install_requires=[
        
    ],
    python_requires='>=3.8',  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
