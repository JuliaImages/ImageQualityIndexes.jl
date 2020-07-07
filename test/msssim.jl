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

    # Gray test images
    img1 = testimage("cameraman")
    img2 = testimage("lena_gray_512")
    @test assess_msssim(img1, img2) ≈ 0.0711 atol=1e-4
    @test assess(MSSSIM(), img1, img2) ≈ MSSSIM()(img1, img2)

    # RGB test images
    img3 = testimage("mandril_color")
    img4 = testimage("lena_color_512")
    @test assess_msssim(img3, img4) ≈ 0.0219 atol=1e-4
    @test assess(MSSSIM(), img3, img4) ≈ MSSSIM()(img3, img4)

    # Varying αᵢ, βᵢ, γᵢ
    iqi_e = MSSSIM(KernelFactors.gaussian(1.5, 11), (
        (0.0448, 0.2856, 0.3001),
        (0.3001, 0.0448, 0.2856),
        (0.2856, 0.3001, 0.0448),
    ))
    @test iqi_e(img3, img4) ≈ 0.0878 atol=1e-4
    # non standard parameters, result may differ from other implementations

    # Comparing with SSIM
    iqi_f = MSSSIM(KernelFactors.gaussian(1.5, 11), (1, ))
    @test iqi_f(img3, img4) ≈ SSIM()(img3, img4) atol=1e-4

    # RGB is not the same as a 3D gray image, unlike SSIM
    assess_msssim(img3, img4) ≠ assess_msssim(channelview(img3), channelview(img4))

    # Gray type tests
    type_list = generate_test_types([Float32, N0f8], [Gray])
    A = rand(128,128)
    B = rand(128,128)

    for T in type_list

        a = A .|> T
        b = B .|> T

        @test assess_msssim(a, b) == assess(MSSSIM(), a, b) == MSSSIM()(a, b)
        @test assess_msssim(a, a) ≈ 1.0
    end

    # check if numbers are treated the same as gray colorants
    a = Gray.(A)
    b = Gray.(B)
    @test assess_msssim(a, b) ≈ assess_msssim(A, B)


    # RGB type tests
    type_list = generate_test_types([Float32, N0f8], [RGB])
    A = [RGB(0.0, 0.0, 0.0) RGB(0.0, 1.0, 0.0) RGB(0.0, 1.0, 1.0)
        RGB(0.0, 0.0, 1.0) RGB(1.0, 0.0, 0.0) RGB(1.0, 1.0, 0.0)
        RGB(1.0, 1.0, 1.0) RGB(1.0, 0.0, 1.0) RGB(0.0, 0.0, 0.0)]
    B = [RGB(0.0, 0.0, 0.0) RGB(0.0, 0.0, 1.0) RGB(1.0, 1.0, 1.0)
        RGB(0.0, 1.0, 0.0) RGB(1.0, 0.0, 0.0) RGB(1.0, 0.0, 1.0)
        RGB(0.0, 1.0, 1.0) RGB(1.0, 1.0, 0.0) RGB(0.0, 0.0, 0.0)]

    for T in type_list

        a = A .|> T
        b = B .|> T

        @test assess_msssim(a, b) == assess(MSSSIM(), a, b) == MSSSIM()(a, b)
        @test assess_msssim(a, a) ≈ 1.0
    end

    # Other Color3 type tests
    type_list = generate_test_types([Float32], [Lab, HSV])
    A = [RGB(0.0, 0.0, 0.0) RGB(0.0, 1.0, 0.0) RGB(0.0, 1.0, 1.0)
        RGB(0.0, 0.0, 1.0) RGB(1.0, 0.0, 0.0) RGB(1.0, 1.0, 0.0)
        RGB(1.0, 1.0, 1.0) RGB(1.0, 0.0, 1.0) RGB(0.0, 0.0, 0.0)]
    B = [RGB(0.0, 0.0, 0.0) RGB(0.0, 0.0, 1.0) RGB(1.0, 1.0, 1.0)
        RGB(0.0, 1.0, 0.0) RGB(1.0, 0.0, 0.0) RGB(1.0, 0.0, 1.0)
        RGB(0.0, 1.0, 1.0) RGB(1.0, 1.0, 0.0) RGB(0.0, 0.0, 0.0)]
    for T in type_list
        a = A .|> T
        b = B .|> T

        @test_nowarn assess_ssim(A, b), assess_ssim(a, B)

        @test assess_ssim(a, b) == assess(SSIM(), a, b) == SSIM()(a, b)
        @test assess_ssim(A, A) ≈ 1.0

    end
    @test assess_ssim(A, B) ≈ assess_ssim(Lab.(A), B) atol=1e-4

    type_list = generate_test_types([Float32, N0f8], [RGB, BGR])
    test_cross_type(MSSSIM(), A, B, type_list)
end
