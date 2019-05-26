module ImageQualityIndexes

using MappedArrays, OffsetArrays
using ImageCore, ColorVectorSpace
using ImageCore: NumberLike, Pixel, GenericImage, GenericGrayImage
using ImageDistances, ImageFiltering
using Statistics: mean
using Base.Iterators: repeated

include("generic.jl")
include("psnr.jl")
include("ssim.jl")

export
    # generic
    assess,

    # Peak signal-to-noise ratio
    PSNR,
    psnr,

    # Structral Similarity
    SSIM,
    ssim

end # module
