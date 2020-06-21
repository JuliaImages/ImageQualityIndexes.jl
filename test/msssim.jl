using ImageFiltering

@testset "MS-SSIM" begin
    @info "test: MS-SSIM"
    
    iqi_a = MSSSIM()
    iqi_b = MSSSIM(KernelFactors.gaussian(1.5, 11))
    iqi_c = MSSSIM(KernelFactors.gaussian(1.5, 11), (0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
    @test (iqi_a.kernel == iqi_b.kernel == iqi_c.kernel) &&
          (iqi_a.W == iqi_b.W == iqi_c.W)

    img1 = testimage("mandril_color")
    img2 = testimage("lena_color_512")
    @test assess_msssim(img1, img2) ≈ 0.0192 atol=1e-4
    @test assess(MSSSIM(), img1, img2) ≈ MSSSIM()(img1, img2)

    # Gray Image
    type_list = generate_test_types([Float32, N0f8], [Gray])
    A = rand(128,128)
    B = rand(128,128)

    for T in type_list

        a = A .|> T
        b = B .|> T

        @test assess_msssim(a, b) == assess(MSSSIM(), a, b) == MSSSIM()(a, b)
        @test assess_msssim(a, a) ≈ 1.0
    end

    # RGB Image
    @test assess_msssim(img1, img2) ≈ 0.0192 atol=1e-4
end