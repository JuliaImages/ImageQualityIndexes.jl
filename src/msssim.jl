"""
Computes the MS-SSIM between two images.

!!! info

    MSSSIM, Multi-scale SSIM is defined only for gray images. RGB images are treated as 3d Gray
    images. General `Color3` images are converted to RGB images first, you could
    manually expand them using `channelview` if you don't want them converted to
    RGB first.
```

# Reference

[1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity. _IEEE transactions on image processing_, 13(4), 600-612.

"""
struct MSSSIM{N} <: FullReferenceIQI
    kernel::AbstractArray{<:Real}
    W::NTuple{N, <:Tuple} # number of scales inferred from length
    function MSSSIM(kernel, W)
        ndims(kernel) == 1 || throw(ArgumentError("only 1-d kernel is valid"))
        issymetric(kernel) || @warn "MSSSIM kernel is assumed to be symmetric"
        all(length.(W) .== 3) || throw(ArgumentError("(α, β, γ) required for all scales, instead it's $(W)"))
        all([ all(x .>= 0) for x in W ]) || throw(ArgumentError("α, β, γ should be non-negative for all scales, instead it's $(W)"))
        new{length(W)}(kernel, W)
    end
end

# Weights for α, β, γ in [1]
const MSSSIM_W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) # α, β, γ for scales 1 through 5

MSSSIM(kernel, W::NTuple{N, <:Real}) where N = MSSSIM(kernel, map(x->(x, x, x), W)) # shorthand for αᵢ=βᵢ=γᵢ for all scales i

# USE SSIM_KERNEL as default kernel
MSSSIM(kernel=SSIM_KERNEL) = MSSSIM(kernel, MSSSIM_W)

(iqi::MSSSIM)(x, ref) = _msssim_map(iqi, x, ref)
assess_msssim(x, ref) = MSSSIM()(x, ref)

const DOWNSAMPLE_FILTER = ones(2, 2)./4 # as per author's implementaion

# SSIM does not allow for user specifying peakval and K, so we don't allow it here either
function _msssim_map(iqi::MSSSIM, x::GenericGrayImage, ref::GenericGrayImage)
    if size(x) ≠ size(ref)
        err = ArgumentError("images should be the same size, instead they're $(size(x))-$(size(ref))")
        throw(err)
    end

    # TODO: check that the image can be downsampled enough number of times, and other checks

    level = length(iqi.W) # number of scales
    T = promote_type(float(eltype(ref)), float(eltype(x)))
    x = of_eltype(T, x)
    ref = of_eltype(T, ref)

    N,M = size(x)[end-1:end]
    # check if images are smaller than kernel
    (N < 11 || M < 11) && throw(ArgumentError("imges should be greater than 11x11"))

    # check if no. of levels are >= 1
    level < 1 && throw(ArgumentError("MS-SSIM needs at least one weight"))

    # check if weights are valid - as per authors implementaion
    sum(sum.(iqi.W)) == 0 && throw(ArgumentError("MS-SSIM must have at least one weight > 0"))

    # check if kernel can be applied

    # (H,) = size(iqi.kernel)
    # min_img_width = min(M, N)/(2^(level-1));
    # max_win_width = H;
    
    # ((H*N)<4 || (H>M) || (H>N) || (min_img_width < max_win_width)) && throw(ArgumentError("kernel can not applied for given dimention of image"))

    # downsampling window
    window = kernelfactors(Tuple(repeated(DOWNSAMPLE_FILTER, ndims(ref))))

    mean_cs = []
    for i in 1:level-1
        cs = SSIM(iqi.kernel, (zero(typeof(iqi.W[i][1])), iqi.W[i][2], iqi.W[i][3]))(x, ref)
        append!(mean_cs, cs)

        imfilter(x, window, "symmetric")
        imfilter(ref, window, "symmetric")

        # downsampling
        if (ndims(x)==2)
            x = x[1:2:end, 1:2:end]
            ref = ref[1:2:end, 1:2:end]
        else
            x = x[1:end,1:2:end, 1:2:end]
            y = ref = ref[1:end, 1:2:end, 1:2:end]
        end
    end

    # last scale
    lcs = SSIM(iqi.kernel, (iqi.W[end][1], iqi.W[end][2], iqi.W[end][3]))(x, ref)
    append!(mean_cs, lcs)

    return min.(prod(mean_cs), 1)

    # TODO: Add sum option as well from author's implementaion

end


_msssim_map(iqi::MSSSIM,
          x::AbstractArray{<:AbstractRGB},
          ref::AbstractArray{<:AbstractRGB}) =
    _msssim_map(iqi, channelview(x), channelview(ref))

_msssim_map(iqi::MSSSIM,
        x::AbstractArray{<:Color3},
        ref::AbstractArray{<:Color3}) =
    _msssim_map(iqi, of_eltype(RGB, x), of_eltype(RGB, ref))
