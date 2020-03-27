import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="legibleml", # Replace with your own username
    version="0.0.1",
    author="sumatrae",
    author_email="sumatrae@163.com",
    description="A easy practice of machine learning algorithm ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sumatrae/legible",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)