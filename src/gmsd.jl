"""
   gmsd(org_img, pred_img, T = 170.0)

Inspired by the official implementtation

"""
function gmsd(org_img, pred_img, T = 170.0)
    gmsd_list = []
    for i in [red, green, blue] 
        grad1x, grad1y, gm1, orient1 = imedge(i.(org_img))
        grad2x, grad2y, gm2, orient2 = imedge(i.(pred_img))

        gms_numerator = 2.0 * gm1 * gm2 .+ T
        gms_denominator = gm1 ^ 2 + gm2 ^ 2 .+ T

        gm = gms_numerator ./ gms_denominator
        gmsd = std(gm)
        push!(gmsd_list, gmsd)
    end 
    return gmsd_list
end