# renaming `psnr` and `ssim` following the general guideline
# > for functions that compute something or perform an operation, start with a verb
# https://github.com/JuliaImages/Images.jl/issues/767
Base.@deprecate_binding psnr assess_psnr
Base.@deprecate_binding ssim assess_ssim
