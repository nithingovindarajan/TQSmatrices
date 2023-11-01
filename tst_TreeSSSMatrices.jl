include("TreeSSSMatrices.jl")
using .TreeSSSMatrices
using LinearAlgebra
using Test

A = ZeroMatrix{Float64}(4,2)
A[1,1]

@test 0 == 0


length(Set([1, 2,2, 3])) == length([1,2,2,3])

all([true,false,true])


a = [1,4,6,3]

b = [x>=3  for  x in a]

A = rand(Int,3,2)
B = rand(Int,3,2)

matrix_of_matrices = Matrix{Matrix{Int}}(undef, 2, 2)
matrix_of_matrices[1,1]=A
matrix_of_matrices[2,1]=B
matrix_of_matrices[1,2]=A
matrix_of_matrices[2,2]=B



hmm = convert(Matrix{AbstractMatrix{Float64}}, matrix_of_matrices)