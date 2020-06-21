using ImageQualityIndexes
using Test, ReferenceTests, TestImages
using Statistics

include("testutils.jl")

@testset "ImageQualityIndexes" begin

    include("psnr.jl")
    include("ssim.jl")
    include("msssim.jl")
    include("colorfulness.jl")
    include("deprecations.jl")
    
end

nothing
