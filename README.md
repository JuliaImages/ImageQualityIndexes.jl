# ImageQualityIndexes

ImageQualityIndexes provides the basic image quality assessment methods.

## Supported indexes

* `PSNR` -- Peak signal-to-noise ratio
* `SSIM` -- Structural similarity


## Basic usage

In this package, each index corresponds to a `ImageQualityIndex` type, there are three ways to assess the image quality:

* use the general protocol, e.g., `assess(PSNR(), x, ref)`. This reads as "**assess** the image quality of **x** using method **PSNR** with information **ref**"
* each index instance is itself a function, e.g., `PSNR()(x, ref)`
* for well-known indexes, there are also convenient name for it, e.g., `psnr(x, ref)`

For detailed usage of particular index, please check the docstring (e.g., `?PSNR`)
