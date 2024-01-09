from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

    
setup(
    name='stabledrnn',
    version='0.0.2',
    author='Luca Herrtti',
    author_email='luca.herrtti@gmail.com',
    description='Official Repo for Stabilizing RNN Gradients through Pre-training',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lucehe/stable_drnn',
    project_urls = {
        "Bug Tracker": "https://github.com/lucehe/stable_drnn/issues"
    },
    license='MIT',
    python_requires=">=3.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)