"""
    MSSSIM([kernel], [W]; num_scales = length(W)) <: FullReferenceIQI
    assess(iqi::MSSSIM, img, ref)
    assess_msssim(img, ref; num_scales = 5)

Computes the multi-scale structural similarity (MS-SSIM) between two images.

!!! tip

    The default parameters comes from [1]. For benchmark usage, it is recommended to not
    change the parameters, because most other SSIM implementations follows the same settings.

# Examples

For benchmark usage, it is recommended to use `assess_msssim(img, ref)`. One could also create
a custom `MSSSIM` instance and then pass it to `assess` or use it as a function. 

```julia
iqi = MSSSIM(KernelFactors.gaussian(2.5, 17), (1.0, 1.0, 2.0)./4)
assess(iqi, img, ref)
iqi(img, ref) # both usages are equivalent
```

# Reference

[1] Wang, Z., Simoncelli, E. P., & Bovik, A. C. (2003, November). Multiscale structural similarity for image quality assessment.In *The Thrity-Seventh Asilomar Conference on Signals, Systems & Computers*, 2003 (Vol. 2, pp. 1398-1402). Ieee.

[2] Wang, Z. IW-SSIM: Information Content Weighted Structural Similarity Index for Image Quality Assessment. Retrived July 9, 2020, from https://ece.uwaterloo.ca/~z70wang/research/iwssim/

[3] Jorge Pessoa. pytorch-msssim. Retrived July 9, 2020, from https://github.com/jorge-pessoa/pytorch-msssim

"""
struct MSSSIM{A, N} <: FullReferenceIQI
    kernel::A
    W::NTuple{N, NTuple{3, Float64}}

    function MSSSIM(kernel=SSIM_KERNEL, W=MSSSIM_W; num_scales::Integer=length(W))
        ndims(kernel) == 1 || throw(ArgumentError("only 1-d kernel is valid"))
        issymetric(kernel) || @warn "MSSSIM kernel is assumed to be symmetric"
        all(length.(W) .== 3) || throw(ArgumentError("(α, β, γ) required for all scales, instead it's $(W)"))
        (num_scales ∈ 1:length(W)) || throw(ArgumentError("`num_scales` $(num_scales) should be positive integer between [1, $(length(W))]"))
        all(x-> x>=0, flatten(W)) || throw(ArgumentError("α, β, γ should be non-negative for all scales, instead it's $(W)"))
        sum(flatten(W)) == 0 && throw(ArgumentError("MS-SSIM must have at least one weight > 0"))

        kernel = centered(kernel)
        if num_scales ≠ length(W)
            W ≠ MSSSIM_W && @warn "truncate MS-SSIM weights to scale $num_scales"
        end
        W = W[1:num_scales]
        sw = reduce((w1, w2) -> w1.+w2, W[1:num_scales])
        if !all(x -> x≈1.0, sw)
            W ≠ MSSSIM_W[1:num_scales] && @warn "normalize MS-SSIM weights so that (∑α, ∑β, ∑γ) == (1.0, 1.0, 1.0)"
            W = map(w->w./sw, W)
        end
        
        new{typeof(kernel), Int(num_scales)}(kernel, W)
    end
end
# Weights for α, β, γ in [1]
const MSSSIM_W = (
    (0.0448, 0.0448, 0.0448), # α₁, β₁, γ₁
    (0.2856, 0.2856, 0.2856), # α₂, β₂, γ₂
    (0.3001, 0.3001, 0.3001), # α₃, β₃, γ₃
    (0.2363, 0.2363, 0.2363), # α₄, β₄, γ₄
    (0.1333, 0.1333, 0.1333), # α₅, β₅, γ₅
)
# shorthand for αᵢ=βᵢ=γᵢ for all scales i
function MSSSIM(kernel, W::NTuple{N, <:Real}; num_scales=length(W)) where N
    MSSSIM(kernel, map(x->(x, x, x), W); num_scales=num_scales)
end

Base.:(==)(ia::MSSSIM, ib::MSSSIM) = ia.kernel == ib.kernel && ia.W == ib.W


# SSIM does not allow for user specifying peakval and K, so we don't allow it here either
(iqi::MSSSIM)(x, ref) = _msssim(iqi, x, ref)
assess_msssim(x, ref; num_scales=5) = MSSSIM(num_scales=num_scales)(x, ref)

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
    if axes(x) ≠ axes(ref)
        x = OffsetArrays.no_offset_view(x)
        ref = OffsetArrays.no_offset_view(ref)
    end
    C₁, C₂ = @. (peakval * K)^2
    C₃ = C₂/2

    num_scales = length(iqi.W)
    T = promote_type(float(eltype(ref)), float(eltype(x)))
    x = of_eltype(T, x)
    ref = of_eltype(T, ref)

    mean_lcs = NTuple{3, Float64}[]
    for i in 1:num_scales
        # instead of the original implementation that calculates mean(c .* s) 
        # here we use a more general version mean(c) * mean(s)
        lcs = mean.(__ssim_map_general(x, ref, iqi.kernel, C₁, C₂, C₃; crop=true))
        push!(mean_lcs, lcs)

        # downsampling
        i == num_scales && break
        x = _average_pooling(x)
        ref = _average_pooling(ref)
    end

    # weighted them up, but only l of the last scale is used according to equation (7) in [1]
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
