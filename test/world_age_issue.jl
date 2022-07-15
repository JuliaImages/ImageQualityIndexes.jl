@testset "world age issue" begin
    img = rand(Gray, 64, 64)
    @testset "ssim" begin
        foo() = assess_ssim(img, img)
        @test foo() == 1.0
    end
    @testset "psnr" begin
        foo() = assess_psnr(img, img)
        @test foo() == Inf
    end
    @testset "entropy" begin
        foo() = entropy(img)
        @test_nowarn foo()
    end
end
