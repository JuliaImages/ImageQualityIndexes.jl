@testset "colorfulness" begin
    @info "test: colorfulness"

    # Test against simple result calculated by hand
    
    x = [RGB(1,0,0), RGB(0,1,0), RGB(0,0,1)]
    @test trunc(colorfulness(x)) == 337

    # Black is not colorful
    
    x = [RGB(0,0,0), RGB(0,0,0), RGB(0,0,0)]
    @test trunc(colorfulness(x)) == 0

    # White is not colorful
    
    x = [RGB(0,0,0), RGB(0,0,0), RGB(0,0,0)]
    @test trunc(colorfulness(x)) == 0
    
    # Lena 256 is a reduced color image and so should be less colorful
    # than the original

    imga = testimage("lena_color_256")
    imgb = testimage("lena_color_512")
    
    @test  colorfulness(imga) < colorfulness(imgb) 

    # A grayscale image has no color
    
    cameraman = testimage("cameraman")
    @test colorfulness(cameraman) == 0

    # A color image with only grays has no color
    
    x = convert(Array{Float64}, cameraman)
    img = RGB.(x, x, x)
    @test colorfulness(img) == 0

    # Test all invocation styles
    c1 = colorfulness(imga)
    c2 = hasler_and_susstrunk_m3(imga)
    c3 = colorfulness(imga, HASLER_AND_SUSSTRUNK_M3())
    c4 = assess(HASLER_AND_SUSSTRUNK_M3(), imga)

    @test c1 == c2 == c3 == c4
end