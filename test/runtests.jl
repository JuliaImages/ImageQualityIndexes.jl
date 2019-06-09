using ImageQualityIndexes
using Test, ReferenceTests, TestImages, ImageFiltering
using Statistics

include("testutils.jl")

@testset "ImageQualityIndexes" begin

    include("psnr.jl")
    include("ssim.jl")

end

nothing
