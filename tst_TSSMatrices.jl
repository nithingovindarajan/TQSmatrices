include("TSSMatrices.jl")
using .TSSMatrices
using LinearAlgebra
using Test


######################################
# Test 1: construct a spinner matrix #
######################################
id = 1;
neighbors = [2, 3, 6];
A = Matrix(undef, 3, 3);
A[1, 1], A[1, 2], A[1, 3] = rand(3, 4), rand(3, 2), rand(3, 3);
A[2, 1], A[2, 2], A[2, 3] = rand(4, 4), rand(4, 2), rand(4, 3);
A[3, 1], A[3, 2], A[3, 3] = rand(2, 4), rand(2, 2), rand(2, 3);
B = [rand(3,2),
     rand(4,2),
     rand(2,2)];
C = [rand(3,4), rand(3,2), rand(3,3) ];
D = rand(3, 2);
node = Spinner{Float64}(id, neighbors, A, B, C, D);

@test get_A(node,2,6) == A[1,3]
@test get_B(node,2) == B[1]
@test get_C(node,3) == C[2]
@test get_D(node) == D

######################################
# Test 2: construct a spinner matrix #
######################################