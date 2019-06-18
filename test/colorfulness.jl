@testset "colorfulness" begin
    @info "test: colorfulness"

    # Test against simple result calculated by hand
    
    x = [RGB(1,0,0), RGB(0,1,0), RGB(0,0,1)]
    @test trunc(colorfulness(x)) == 337

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

end
