using ImageCore: GenericGrayImage
using Statistics: mean, std

struct HASLER_AND_SUSSTRUNK_M3 <: NoReferenceIQI end

# api
(iqi::HASLER_AND_SUSSTRUNK_M3)(img) = colorfulness(img, HASLER_AND_SUSSTRUNK_M3())

"""

```
M =  colorfulness(img)
```

Calculates the colorfulness of an RGB image according to the metric,
M3 from [1]. As a guide to interpretation of results, the authors
suggest:

|Not colorful        |  0|
|slightly colorful   | 15|
|moderately colorful | 33|
|averagely colorful  | 45|
|quite colorful      | 59|
|highly colorful     | 82|
|extremely colorful  |109|

[1] Hasler, D. and Süsstrunk, S.E., 2003, June. Measuring colorfulness
in natural images. In Human vision and electronic imaging VIII
(Vol. 5007, pp. 87-96). International Society for Optics and
Photonics.

"""
function colorfulness(img::AbstractArray{<:AbstractRGB}, m::HASLER_AND_SUSSTRUNK_M3)

    R = 255 .* float(red.(img))
    G = 255 .* float(green.(img))
    B = 255 .* float(blue.(img))

    rg = R .- G 
    μrg, σrg = mean(rg), std(rg)
   
    yb = 0.5 .* (R .+ G) .- B
    μyb,  σyb = mean(yb), std(yb)

    μrgyb = sqrt(μrg^2 + μyb^2)
    σrgyb = sqrt(σrg^2 + σyb^2)

    return σrgyb + 0.3 * μrgyb

end

colorfulness(img::GenericGrayImage, m::HASLER_AND_SUSSTRUNK_M3) = 0

colorfulness(img) = colorfulness(img, HASLER_AND_SUSSTRUNK_M3())


