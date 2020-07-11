"""
    MSSSIM([kernel], [W]) <: FullReferenceIQI
    assess(iqi::MSSSIM, img, ref)
    assess_msssim(img, ref)

Computes the MS-SSIM between two images.

!!! tip

    The default parameters comes from [1]. For benchmark usage, it is recommended to not
    change the parameters, because most other SSIM implementations follows the same settings.

# Examples

For benchmark usage, it is recommended to use `assess_msssim(img, ref)`. One could also create
a custom `MSSSIM` instance and then pass it to `assess` or use it as a function. For example:

```julia
# TODO
```

# Reference

[1] Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003, November). Multiscale structural similarity for image quality assessment.In *The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers*, 2003 (Vol. 2, pp. 1398-1402). Ieee.

[2] Wang, Z. IW-SSIM: Information Content Weighted Structural Similarity Index for Image Quality Assessment. Retrived July 9, 2020, from https://ece.uwaterloo.ca/~z70wang/research/iwssim/

[3] Jorge Pessoa. pytorch-msssim. Retrived July 9, 2020, from https://github.com/jorge-pessoa/pytorch-msssim

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

Base.:(==)(ia::MSSSIM, ib::MSSSIM) = ia.kernel == ib.kernel && ia.W == ib.W

# Weights for α, β, γ in [1]
const MSSSIM_W = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) # α, β, γ for scales 1 through 5

MSSSIM(kernel, W::NTuple{N, <:Real}) where N = MSSSIM(kernel, map(x->(x, x, x), W)) # shorthand for αᵢ=βᵢ=γᵢ for all scales i

# USE SSIM_KERNEL as default kernel
MSSSIM(kernel=SSIM_KERNEL) = MSSSIM(kernel, MSSSIM_W)

# SSIM does not allow for user specifying peakval and K, so we don't allow it here either
(iqi::MSSSIM)(x, ref) = _msssim(iqi, x, ref)
assess_msssim(x, ref) = MSSSIM()(x, ref)

# Implementation details
function _msssim(iqi::MSSSIM,
                 x::Union{GenericGrayImage, AbstractArray{<:AbstractRGB}},
                 ref::Union{GenericGrayImage, AbstractArray{<:AbstractRGB}};
                 peakval=1.0,
                 K=SSIM_K)
    if size(x) ≠ size(ref)
        err = ArgumentError("images should be the same size, instead they're $(size(x))-$(size(ref))")
        throw(err)
    end
    C₁, C₂ = @. (peakval * K)^2
    C₃ = C₂/2

    level = length(iqi.W) # number of scales
    T = promote_type(float(eltype(ref)), float(eltype(x)))
    x = of_eltype(T, x)
    ref = of_eltype(T, ref)

    # check if no. of levels are >= 1
    level < 1 && throw(ArgumentError("MS-SSIM needs at least one scale"))

    # check if weights are valid - as per authors implementaion
    sum(sum.(iqi.W)) == 0 && throw(ArgumentError("MS-SSIM must have at least one weight > 0"))

    mean_lcs = NTuple{3, Float64}[]
    for i in 1:level
        # Note that this is different from other implementations, including the original one,
        # that we don't compute mean(c .* s), instead, we compute mean(c) * mean(s)
        lcs = mean.(__ssim_map_general(x, ref, iqi.kernel, C₁, C₂, C₃; crop=true))
        push!(mean_lcs, lcs)

        # downsampling
        i == level && break
        x = _average_pooling(x)
        ref = _average_pooling(ref)
    end

    # weighted them up, but only l of the last level is used according to equation (7) in [1]
    mean_lcs = map(mean_lcs, iqi.W) do lcs, w
        # similar to SSIM, here we ensure that negative numbers in s are not being raised to powers
        # less than 1
        s = w[3] < 1 ? max(0, lcs[3]) : lcs[3]
        (lcs[1], lcs[2], s) .^ w
    end
    if length(mean_lcs) == 1
        return prod(first(mean_lcs))
    else
        return mapreduce(x->x[2]*x[3], *, mean_lcs[1:end-1]) * prod(mean_lcs[end])
    end
end

_msssim(iqi::MSSSIM, x::AbstractArray{<:Color3}, ref::AbstractArray{<:Color3}) =
    _msssim(iqi, of_eltype(RGB, x), of_eltype(RGB, ref))

function _average_pooling(x::GenericImage; kernel=ones(2)./2)
    # note that this is slightly slower than two-fold method `ImageTransformations.restrict`
    window = kernelfactors(Tuple(repeated(kernel, ndims(x))))
    R = ntuple(i->first(axes(x, i)) : 2 : last(axes(x, i)), ndims(x))
    return imfilter(x, window, "symmetric")[R...]
end
