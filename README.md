# ImageQualityIndexes

[![][action-img]][action-url]
[![][pkgeval-img]][pkgeval-url]
[![][codecov-img]][codecov-url]

ImageQualityIndexes provides the basic image quality assessment methods. Check the reasoning behind the code design [here](https://nextjournal.com/johnnychen94/the-principles-of-imagesjl-part-i) if you're interested in.

## Supported indexes

### Full reference indexes

* `PSNR`/`assess_psnr` -- Peak signal-to-noise ratio
* `SSIM`/`assess_ssim` -- Structural similarity
* `MSSSIM`/`assess_msssim` -- Multi-scale SSIM

### No-reference indexes

* `HASLER_AND_SUSSTRUNK_M3`/`hasler_and_susstrunk_m3` -- Colorfulness

## Basic usage

The root type is `ImageQualityIndex`, each concrete index is supposed to be one of `FullReferenceIQI`, `ReducedReferenceIQI` and `NoReferenceIQI`.

There are three ways to assess the image quality:

* use the general protocol, e.g., `assess(PSNR(), x, ref)`. This reads as "**assess** the image quality of **x** using method **PSNR** with information **ref**"
* each index instance is itself a function, e.g., `PSNR()(x, ref)`
* for well-known indexes, there are also convenient name for it for benchmark purpose.

For detailed usage of particular index, please check the docstring (e.g., `?PSNR`)

## Examples

```julia
using Images, TestImages
using ImageQualityIndexes

img = testimage("cameraman") .|> float64
noisy_img = img .+ 0.1 .* randn(size(img))
assess_ssim(noisy_img, img) # 0.24112
assess_psnr(noisy_img, img) # 19.9697

kernel = ones(3, 3)./9 # mean filter
denoised_img = imfilter(noisy_img, kernel)
assess_psnr(denoised_img, img) # 28.4249
assess_ssim(denoised_img, img) # 0.6390
assess_msssim(denoised_img, img) # 0.8460

img = testimage("fabio");
colorfulness(img) # 68.5530

```

<!-- URLS -->

[pkgeval-img]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/I/ImageQualityIndexes.svg
[pkgeval-url]: https://juliaci.github.io/NanosoldierReports/pkgeval_badges/report.html
[action-img]: https://github.com/JuliaImages/ImageQualityIndexes.jl/workflows/Unit%20test/badge.svg
[action-url]: https://github.com/JuliaImages/ImageQualityIndexes.jl/actions
[codecov-img]: https://codecov.io/github/JuliaImages/ImageQualityIndexes.jl/coverage.svg?branch=master
[codecov-url]: https://codecov.io/github/JuliaImages/ImageQualityIndexes.jl?branch=master
