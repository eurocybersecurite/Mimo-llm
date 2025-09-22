from setuptools import setup, find_packages

setup(
    name='mimo-llm',
    version='0.1.0',
    packages=find_packages(),
    description='A language model for code and conversation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='ABDESSEMED Mohamed Redha',
    author_email='mohamed.abdessemed@eurocybersecurite.fr',
    url='https://github.com/eurocybersecurite/Mimo-llm',
    install_requires=[
        'transformers>=4.35',
        'datasets>=2.14',
        'accelerate>=0.22',
        'bitsandbytes>=0.41',
        'peft>=0.4.0',
        'torch>=2.1',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        "assets": ["*.png"],  # inclut toutes les images PNG
    },
)
