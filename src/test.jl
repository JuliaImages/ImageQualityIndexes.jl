struct Point{N}
    W::NTuple{N}
end

p =Point{3}((1, 2, 3))
Point(x) = Point{length(x)}(x)
p = Point((1, 2))

# function _msssim_map(iqi::MSSSIM, x::AbstractArray{<:AbstractRGB}, ref::AbstractArray{<:AbstractRGB})
#     c_msssim = []
#     map(eachslice(channelview(x); dims=1), eachslice(channelview(ref); dims=1)) do cx, cref
#         append!(c_msssim, _msssim_map(iqi, cx, cref))
#     end
#     return prod(c_msssim)
#     # some post-processing might be needed, though
# end
