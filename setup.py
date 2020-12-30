import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "keras-unet-collection",
    version = "0.0.2",
    author = "Yingkai (Kyle) Sha",
    author_email = "yingkaisha@gmail.com",
    description = "The tensorflow.keras implementations of U-net, U-net++, Residual U-net, Attention U-net.",
    long_description = "The tensorflow.keras implementations of U-net, U-net++, Residual U-net, Attention U-net.",
    long_description_content_type = "text/markdown",
    url = "https://github.com/yingkaisha/keras-unet-collection",
    packages = setuptools.find_packages(),
    classifiers=[ "Programming Language :: Python :: 3",
                  "License :: OSI Approved :: MIT License",
                  "Operating System :: OS Independent",],
    python_requires='>=3.6',)
