"""
    PeakSignalNoiseRatio <: FullReferenceIQI
    psnr(x, ref [, peakval])
    assess(PSNR(), x, ref, [, peakval])

Peak signal-to-noise ratio (PSNR) is used to measure the quality of image.

PSNR (in dB) is calculated by `10log10(PeakVal^2/mse(x, ref))`, where `PeakVal`
is the maximum possible pixel value of image `ref`, and
[`mse`](@ref ImageDistances.mse) is the mean squared error between images `x`
and `ref`.

!!! info

    For general `Color3` images, PSNR is reported against each channel of that
    color space as a `Vector`. One exception is `AbstractRGB` images, where
    `mse` is calculated over three channels by convention.

!!! info

    `peakval` is optional only for gray and RGB color space. For other color
    spaces, you need to explicitly pass it as a `Vector`.
"""
struct PeakSignalNoiseRatio <: FullReferenceIQI end
const PSNR = PeakSignalNoiseRatio # alias PSNR since it's too famous

# api
(iqi::PSNR)(x, ref, peakval) = psnr(x, ref, peakval)
(iqi::PSNR)(x, ref) = iqi(x, ref, peak_value(eltype(ref)))

@doc (@doc PeakSignalNoiseRatio)
psnr(x, ref, peakval) = psnr(x, ref, peakval)
psnr(x, ref) = psnr(x, ref, peak_value(eltype(ref)))


# implementation
""" Define the default peakval for colors """
peak_value(::Type{T}) where T <: NumberLike = one(eltype(T))
peak_value(::Type{T}) where T <: AbstractRGB = one(eltype(T))
function peak_value(::Type{T}) where T <: Color3
    err_msg = "peakval for PSNR can't be inferred and should be explicitly passed for $(T) images"
    throw(ArgumentError(err_msg))
end

psnr(x::GenericGrayImage, ref::GenericGrayImage, peakval::Real) =
    20log10(peakval) - 10log10(mse(x, ref))

psnr(x::GenericImage{<:AbstractRGB}, ref::GenericImage{<:AbstractRGB},
     peakval::Real) =
    psnr(channelview(x), channelview(ref), peakval)

function psnr(x::GenericImage{<:Color3}, ref::GenericImage{CT},
              peakvals) where {CT<:Color3}
    check_peakvals(CT, peakvals)

    cx = channelview(x)
    cref = channelview(ref)
    n = length(CT)
    [psnr(view(cx, i,:,:), view(cref, i,:,:), peakvals[i]) for i in 1:n]
end

function check_peakvals(CT, peakvals)
    if length(peakvals) ≠ length(CT)
        err_msg = "peakvals for PSNR should be length-$(length(CT)) vector for $(base_colorant_type(CT)) images"
        throw(ArgumentError(err_msg))
    end
end
