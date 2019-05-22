module ImageQualityIndexes

using MappedArrays
using ImageCore
using ImageCore: NumberLike, Pixel, GenericImage, GenericGrayImage
using ImageDistances

include("generic.jl")
include("psnr.jl")

export
    # generic
    assess,

    # Peak signal-to-noise ratio
    PSNR,
    psnr

end # module
