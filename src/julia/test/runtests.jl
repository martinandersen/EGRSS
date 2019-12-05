using LinearAlgebra
using Test

include("../egrss.jl")

include("test_generators.jl")
include("test_potrf.jl")
include("test_ldl.jl")
include("test_full.jl")
include("test_full_tril.jl")
include("test_gemv.jl")
include("test_symv.jl")
include("test_trmv.jl")
include("test_trsv.jl")
include("test_trtri.jl")
include("test_trnrms.jl")
include("test_bigfloat.jl")
include("test_rational.jl")
