# using ImagePhaseCongruency
# using Images

function fsim_similarity_measure(x, y, constant)
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y .+ constant
    denominator = x^2 .+ y^2 .+ constant 

	return numerator ./ denominator
end

function fsim(org_img, pred_img, T1=0.85, T2=160)
    # alpha,beta parameters used to adjust the relative importance of PC and GM features
    alpha = 1 
    beta = 1
    fsim_list = []
    for i in [red,green,blue] 
       (M1, m1, or1, ft1, EO1, T1) = phasecong3(i.(org_img);nscale=4,minwavelength=6, mult=2,sigmaonf=0.5978)
       (M2, m2, or2, ft2, EO2, T2) = phasecong3(i.(pred_img);nscale=4,minwavelength=6, mult=2,sigmaonf=0.5978)
       pco_sum = M1
       pcp_sum = M2
       grad1x, grad1y, gm1, orient1 = imedge(i.(org_img))
       grad2x, grad2y, gm2, orient2 = imedge(i.(pred_img))
       S_pc = fsim_similarity_measure(pco_sum, pcp_sum, T1)
       S_g = fsim_similarity_measure(gm1, gm2, T2)
       S_l = (S_pc^alpha) .* (S_g^beta)
       numerator = sum(S_l .* map(max,pco_sum, pcp_sum))
       denominator = sum(map(max,pco_sum, pcp_sum))
       push!(fsim_list, numerator / denominator)
    end
    return mean(fsim_list)
end
