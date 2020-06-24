using ImageFiltering

@testset "MS-SSIM" begin
    @info "test: MS-SSIM"

    # constructors
    iqi_a = MSSSIM()
    iqi_b = MSSSIM(KernelFactors.gaussian(1.5, 11))
    iqi_c = MSSSIM(KernelFactors.gaussian(1.5, 11), (0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
    iqi_d = MSSSIM(KernelFactors.gaussian(1.5, 11), (
        (0.0448, 0.0448, 0.0448),
        (0.2856, 0.2856, 0.2856),
        (0.3001, 0.3001, 0.3001),
        (0.2363, 0.2363, 0.2363),
        (0.1333, 0.1333, 0.1333),
    ))
    @test (iqi_a.kernel == iqi_b.kernel == iqi_c.kernel == iqi_d.kernel) &&
          (iqi_a.W == iqi_b.W == iqi_c.W == iqi_d.W)

    # Gray images
    img1 = testimage("cameraman")
    img2 = testimage("lena_gray_512")
    @test assess_msssim(img1, img2) ≈ 0.0517 atol=1e-4
    @test assess(MSSSIM(), img1, img2) ≈ MSSSIM()(img1, img2)

    # RGB images
    img3 = testimage("mandril_color")
    img4 = testimage("lena_color_512")
    @test assess_msssim(img3, img4) ≈ 0.0192 atol=1e-4
    @test assess(MSSSIM(), img3, img4) ≈ MSSSIM()(img3, img4)

    # Varying αᵢ, βᵢ, γᵢ
    iqi_e = MSSSIM(KernelFactors.gaussian(1.5, 11), (
        (0.0448, 0.2856, 0.3001),
        (0.3001, 0.0448, 0.2856),
        (0.2856, 0.3001, 0.0448),
    ))
    @test iqi_e(img3, img4) ≈ 0.0852 atol=1e-4
    # non standard parameters, result may differ from other implementations

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

end
