import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mauve-text",
    version="0.4.0",
    author="Krishna Pillutla",
    author_email="pillutla@cs.washington.edu",
    description="Implementation of the MAUVE to evaluate text generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krishnap25/mauve",
    project_urls={
        "Bug Tracker": "https://github.com/krishnap25/mauve/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.18.1',
        'scikit-learn>=0.22.1',
        'faiss-cpu>=1.7.0',
        'tqdm>=4.40.0',
        'requests']
)
