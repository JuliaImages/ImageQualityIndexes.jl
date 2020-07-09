using ImageFiltering

@testset "SSIM" begin
    @info "test: SSIM"

    iqi_a = SSIM()
    iqi_b = SSIM(KernelFactors.gaussian(1.5, 11))
    iqi_c = SSIM(KernelFactors.gaussian(1.5, 11), (1.0, 1.0, 1.0))
    @test (iqi_a.kernel == iqi_b.kernel == iqi_c.kernel) &&
          (iqi_a.W == iqi_b.W == iqi_c.W)

    iqi = SSIM()
    sz_img_3 = (3, 3, 3)

    # numerical test
    img1 = testimage("cameraman")
    img2 = testimage("lena_gray_512")
    @test assess_ssim(img1, img2) ≈ 0.3595 atol=1e-4 # MATLAB built-in ssim result

    # this calls the general implementation
    iqi_δ1 = SSIM(KernelFactors.gaussian(1.5, 11), (1.0+1e-5, 1.0, 1.0))
    @test assess(iqi_δ1, img1, img2) ≈ assess(SSIM(), img1, img2) atol = 1e-4
    # this calls the general implementation with max.(s, 0)
    iqi_δ2 = SSIM(KernelFactors.gaussian(1.5, 11), (1.0, 1.0, 1.0-1e-5))
    @test assess(iqi_δ2, img1, img2) ≈ assess(SSIM(), img1, img2) atol = 1e-2

    # non-standard powers
    iqi_γ = SSIM(KernelFactors.gaussian(1.5, 11), (0.5, 0.5, 0.5))
    @test iqi_γ(img1, img2) ≈ 0.5261 atol=1e-4
    # non standard parameters, result may differ from other implementations

    # Gray image
    type_list = generate_test_types([Bool, Float32, N0f8], [Gray])
    A = [1.0 1.0 1.0; 1.0 1.0 1.0; 0.0 0.0 0.0]
    B = [1.0 1.0 1.0; 0.0 0.0 0.0; 1.0 1.0 1.0]
    for T in type_list
        test_ndarray(iqi, sz_img_3, T)

        a = A .|> T
        b = B .|> T

        @test assess_ssim(a, b) == assess(SSIM(), a, b) == SSIM()(a, b)
        @test assess_ssim(a, a) ≈ 1.0

        # FIXME: the result of Bool type is not strictly equal to others
        eltype(T) <: Bool && continue
        test_numeric(iqi, a, b, T)
        test_numeric(iqi, channelview(a), channelview(b), T)
    end
    test_cross_type(iqi, A, B, type_list)

    # RGB image
    img1 = testimage("mandril_color")
    img2 = testimage("lena_color_512")
    # this differs from MATLAB built-in result as our implementation don't slide the window in the channel dimension
    @test assess_ssim(img1, img2) ≈ 0.11069226443828077 atol=1e-4

    type_list = generate_test_types([Float32, N0f8], [RGB])
    A = [RGB(0.0, 0.0, 0.0) RGB(0.0, 1.0, 0.0) RGB(0.0, 1.0, 1.0)
        RGB(0.0, 0.0, 1.0) RGB(1.0, 0.0, 0.0) RGB(1.0, 1.0, 0.0)
        RGB(1.0, 1.0, 1.0) RGB(1.0, 0.0, 1.0) RGB(0.0, 0.0, 0.0)]
    B = [RGB(0.0, 0.0, 0.0) RGB(0.0, 0.0, 1.0) RGB(1.0, 1.0, 1.0)
        RGB(0.0, 1.0, 0.0) RGB(1.0, 0.0, 0.0) RGB(1.0, 0.0, 1.0)
        RGB(0.0, 1.0, 1.0) RGB(1.0, 1.0, 0.0) RGB(0.0, 0.0, 0.0)]
    for T in type_list
        test_ndarray(iqi, sz_img_3, T)

        a = A .|> T
        b = B .|> T

        @test assess_ssim(a, b) == assess(SSIM(), a, b) == SSIM()(a, b)
        @test assess_ssim(a, b) ≠ assess_ssim(channelview(a), channelview(b))
        @test assess_ssim(a, a) ≈ 1.0

        # RGB is treated as 3d gray image
        test_numeric(iqi, a, b, T)
    end
    type_list = generate_test_types([Float32, N0f8], [RGB, BGR])
    test_cross_type(iqi, A, B, type_list)

    # general Color3 images
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

        # conversion to RGB first differs from no conversion
        @test assess_ssim(a, b) ≠ assess_ssim(channelview(a), channelview(b))
    end
    @test assess_ssim(A, B) ≈ assess_ssim(Lab.(A), B) atol=1e-4
end
