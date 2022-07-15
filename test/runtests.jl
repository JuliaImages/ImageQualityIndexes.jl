using ImageQualityIndexes
using Test
using ImageCore

# For lazily imported packages, world age issue might occur at first call, these tests must
# be put first.
include("world_age_issue.jl")

using ReferenceTests, TestImages
using Suppressor
using Statistics
using OffsetArrays
using ImageTransformations
using ImageFiltering

include("testutils.jl")

@testset "ImageQualityIndexes" begin

    include("psnr.jl")
    include("ssim.jl")
    include("msssim.jl")
    include("colorfulness.jl")
    include("entropy.jl")

end

nothing
