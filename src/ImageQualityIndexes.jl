module ImageQualityIndexes

using MappedArrays
using ImageCore, ColorVectorSpace
using ImageCore: NumberLike, Pixel, GenericImage, GenericGrayImage
using ImageDistances, ImageFiltering
using Statistics: mean, std
using Base.Iterators: repeated

include("generic.jl")
include("psnr.jl")
include("ssim.jl")
include("colorfulness.jl")
include("deprecations.jl")

export
    # generic
    assess,

    # Peak signal-to-noise ratio
    PSNR,
    assess_psnr,

    # Structral Similarity
    SSIM,
    assess_ssim,

    # Colorfulness
    HASLER_AND_SUSSTRUNK_M3,
    hasler_and_susstrunk_m3,
    colorfulness

end # module
