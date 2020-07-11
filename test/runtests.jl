using ImageQualityIndexes
using Test, ReferenceTests, TestImages
using Suppressor
using Statistics
using OffsetArrays
using ImageTransformations

include("testutils.jl")

@testset "ImageQualityIndexes" begin

    include("psnr.jl")
    include("ssim.jl")
    include("msssim.jl")
    include("colorfulness.jl")
    include("deprecations.jl")
    
end

nothing
