module ImageQualityIndexes

using ImageCore
using ImageCore: NumberLike, Pixel, GenericImage, GenericGrayImage
using ImageDistances

include("generic.jl")
include("psnr.jl")

export
    # generic
    assess,

    # Peak signal-to-noise ratio
    PeakSignalNoiseRatio,
    PSNR,
    psnr

end # module
