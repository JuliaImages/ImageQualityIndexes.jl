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
    iqi_e = MSSSIM(; num_scales=5)
    iqi_f = @suppress_err MSSSIM(KernelFactors.gaussian(1.5, 11), 2 .* (0.0448, 0.2856, 0.3001, 0.2363, 0.1333))
    @test iqi_a == iqi_b == iqi_c == iqi_d == iqi_e == iqi_f

    output = @capture_err begin
        MSSSIM(KernelFactors.gaussian(1.5, 11), (0.0448, 0.2856, 0.3001))
    end
    @test output == ""
    output = @capture_err begin
        MSSSIM(KernelFactors.gaussian(1.5, 11), (0.1, 0.3, 0.6); num_scales=1)
    end
    @test occursin("truncate MS-SSIM weights to scale", output)
    @test occursin("normalize MS-SSIM weights so that (∑α, ∑β, ∑γ) == (1.0, 1.0, 1.0)", output)

    # Gray test images
    img1 = testimage("fabio_gray_512")
    img2 = testimage("lena_gray_512")
    # Tensorflow, pytorch_msssim and MATLAB original implementation: 0.1072
    # We have strong reason to suspect that MSSSIM in MATLAB R2020a is incorrect
    # https://github.com/JuliaImages/ImageQualityIndexes.jl/pull/19#issuecomment-655208537
    @test assess_msssim(img1, img2) ≈ 0.11109 atol=1e-4
    @test assess(MSSSIM(), img1, img2) ≈ MSSSIM()(img1, img2)

    # RGB test images
    # MS-SSIM in MATLAB R2020a and its original implementation only support Gray images
    img3 = testimage("mandril_color")
    img4 = testimage("lena_color_512")
    @test assess_msssim(img3, img4) ≈ 0.08516 atol=1e-4 # pytorch_msssim: 0.0786
    @test assess(MSSSIM(), img3, img4) ≈ MSSSIM()(img3, img4)

    # Varying αᵢ, βᵢ, γᵢ
    iqi_e = @suppress_err MSSSIM(KernelFactors.gaussian(1.5, 11), (
        (0.0448, 0.2856, 0.3001),
        (0.3001, 0.0448, 0.2856),
        (0.2856, 0.3001, 0.0448),
    ))
    @test iqi_e(img3, img4) ≈ 0.13109 atol=1e-4
    # non standard parameters, result may differ from other implementations

    # Comparing with SSIM
    iqi_f = MSSSIM(KernelFactors.gaussian(1.5, 11), (1, ))
    @test iqi_f(img3, img4) ≈ SSIM()(img3, img4) atol=1e-3

    # images should have the same size, but their axes can differs
    @test assess_msssim(img1, OffsetArray(img2, -2, -2)) == assess_msssim(img1, img2)
    @test_throws ArgumentError assess_msssim(img1, restrict(img2))

    # RGB - Gray
    @test assess_msssim(img3, Gray.(img4)) == assess_msssim(img3, RGB.(Gray.(img4)))
    @test assess_msssim(img3, img4) ≠ assess_msssim(channelview(img3), channelview(img4))

    # Gray type tests
    type_list = generate_test_types([Float32, N0f8], [Gray])
    A = rand(128,128)
    B = rand(128,128)

    for T in type_list

        a = A .|> T
        b = B .|> T

        @test assess_msssim(a, b) == assess(MSSSIM(), a, b) == MSSSIM()(a, b)
        @test assess_msssim(a, a) ≈ 1.0 atol=1e-4
        @test assess_msssim(a, b; num_scales=1) == assess(MSSSIM(num_scales=1), a, b) == MSSSIM(num_scales=1)(a, b)
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
        @test assess_msssim(a, a) ≈ 1.0 atol=1e-4
        @test assess_msssim(a, b; num_scales=1) == assess(MSSSIM(num_scales=1), a, b) == MSSSIM(num_scales=1)(a, b)
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

        @test_nowarn assess_msssim(A, b), assess_msssim(a, B)

        @test assess_msssim(a, b) == assess(MSSSIM(), a, b) == MSSSIM()(a, b)
        @test assess_msssim(a, a) ≈ 1.0 atol=1e-4
        @test assess_msssim(a, b; num_scales=1) == assess(MSSSIM(num_scales=1), a, b) == MSSSIM(num_scales=1)(a, b)

    end
    @test assess_msssim(A, B) ≈ assess_msssim(Lab.(A), B) atol=1e-4

    type_list = generate_test_types([Float32, N0f8], [RGB, BGR])
    test_cross_type(MSSSIM(), A, B, type_list)
end
