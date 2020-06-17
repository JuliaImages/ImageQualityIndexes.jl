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
struct MSSIM{N} <: FullReferenceIQI
    kernel::AbstractArray{<:Real}
    W::NTuple{N} # number of scales inferred from length
    function MSSIM(kernel, W)
        ndims(kernel) == 1 || throw(ArgumentError("only 1-d kernel is valid"))
        issymetric(kernel) || @warn "MSSIM kernel is assumed to be symmetric"
        all(W .>= 0) || throw(ArgumentError("α, β, γ should be non-negative for all scales, instead it's $(W)"))
        new{length(W)}(kernel, W)
    end
end

# Separate dispatches for Gray, RGB and Color3

# default values
# USE SSIM_KERNEL as default kernel
const MSSIM_W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) # α, β, γ for scales 1 through 5

MSSIM(kernel=SSIM_KERNEL) = MSSIM(kernel, MSSIM_W)

(iqi::MSSIM)(x, ref) = _mssim_map(iqi, x, ref)
assess_mssim(x, ref) = MSSIM()(x, ref)

const DOWNSAMPLE_FILTER = ones(2, 2)./4 # as per author's implementaion

# SSIM does not allow for user specifying peakval and K, so we don't allow it here either
function _mssim_map(iqi::MSSIM, x::GenericGrayImage, ref::GenericGrayImage)
    if size(x) ≠ size(ref)
        err = ArgumentError("images should be the same size, instead they're $(size(x))-$(size(ref))")
        throw(err)
    end

    # TODO: check that the image can be downsampled enough number of times, and other checks

    N = length(iqi.kernel) # number of scales

    T = promote_type(float(eltype(ref)), float(eltype(x)))
    x = of_eltype(T, x)
    ref = of_eltype(T, ref)

    # downsampling window
    window = kernelfactors(Tuple(repeated(DOWNSAMPLE_FILTER, ndims(ref))))

    mean_cs = []
    for i in 1:N-1
        cs = SSIM(iqi.kernel, (zero(typeof(iqi.W[i])), iqi.W[i], iqi.W[i]))(x, ref)
        append!(mean_cs, cs)

        imfilter(x, window, "symmetric")
        imfilter(ref, window, "symmetric")

        # downsampling
        x = x[1:2:end, 1:2:end]
        ref = ref[1:2:end, 1:2:end]
    end

    # last scale
    lcs = SSIM(iqi.kernel, (iqi.W[end], iqi.W[end], iqi.W[end]))(x, ref)
    append!(mean_cs, lcs)

    return prod(mean_cs)

    # TODO: Add sum option as well from author's implementaion

end
