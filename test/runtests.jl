using ImageQualityIndexes
using Test, ReferenceTests, TestImages
using Statistics

include("testutils.jl")

@testset "SSIM" begin

    include("psnr.jl")
    include("ssim.jl")

end

nothing
