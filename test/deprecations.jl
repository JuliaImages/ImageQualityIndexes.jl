@testset "deprecations" begin
    @info "depwarns are expected"
    img1 = testimage("cameraman")
    img2 = testimage("lena_gray_512")
    @test ssim(img1, img2) == assess_ssim(img1, img2)
    @test psnr(img1, img2) == assess_psnr(img1, img2)
end
