using SnoopPrecompile

@precompile_setup begin
    imgs_list = Any[
        [rand(Gray{N0f8}, 32, 32) for _ in 1:2],
        [rand(RGB{N0f8}, 32, 32) for _ in 1:2],
        [rand(Gray{Float64}, 32, 32) for _ in 1:2],
        [rand(RGB{Float64}, 32, 32) for _ in 1:2],
        [rand(N0f8, 32, 32) for _ in 1:2],
        [rand(Float64, 32, 32) for _ in 1:2],
    ]
    @precompile_all_calls begin
        for imgs in imgs_list
            assess_psnr(imgs...)
            assess_ssim(imgs...)
        end
    end
end
