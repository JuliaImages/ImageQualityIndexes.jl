include("ssim.jl")
"""
Computes the MS-SSIM between two images.

!!! info

    SSIM is defined only for gray images. RGB images are treated as 3d Gray
    images. General `Color3` images are converted to RGB images first, you could
    manually expand them using `channelview` if you don't want them converted to
    RGB first.
```

# Reference

[1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_, 13(4), 600-612.

"""
struct MSSIM <: FullReferenceIQI
    kernel::AbstractArray{<:Real}
    W::NTuple{3}
    function MSSIM(kernel, W)
        ndims(kernel) == 1 || throw(ArgumentError("only 1-d kernel is valid"))
        issymetric(kernel) || @warn "MSSIM kernel is assumed to be symmetric"
        all(W .>= 0) || throw(ArgumentError("(α, β, γ) should be non-negative, instead it's $(W)"))
        
        # TODO

    end
end


# Separate dispatches for Gray, RGB and Color3