# Construction of TQS matrix for the example graph G_a(4) in the paper
include("TQSMatrices.jl")
using .TQSMatrices
using LinearAlgebra
using Statistics
using Random

# set seed for experiments
Random.seed!(1234);

# graph G_a(4):
#         1 -- 2 -- 3 -- 4 

# root node
root = 4

# input dimensions
n1 = 13
n2 = 15
n3 = 17
n4 = 19
# output dimensions
m1 = 12
m2 = 14
m3 = 16
m4 = 18
# state dimensions
p12 = 3
p21 = 2
p23 = 3
p32 = 4
p34 = 3
p43 = 3


NO_TRIALS = 10
error = []

for k ∈ 1:NO_TRIALS

	# node 1
	id = 1
	neighbors = [2]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p12, p21)
	inp = [rand(p12, n1)]
	out = [rand(m1, p21)]
	D = rand(m1, n1)
	node1 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 2
	id = 2
	neighbors = [1, 3]
	trans = Matrix(undef, 2, 2)
	trans[1, 1], trans[1, 2] = ZeroMatrix(p21, p12), rand(p21, p32)
	trans[2, 1], trans[2, 2] = rand(p23, p12), ZeroMatrix(p23, p32)
	inp = [rand(p21, n2), rand(p23, n2)]
	out = [rand(m2, p12), rand(m2, p32)]
	D = rand(m2, n2)
	node2 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 3
	id = 3
	neighbors = [2, 4]
	trans = Matrix(undef, 2, 2)
	trans[1, 1], trans[1, 2] = ZeroMatrix(p32, p23), rand(p32, p43)
	trans[2, 1], trans[2, 2] = rand(p34, p23), ZeroMatrix(p34, p43)
	inp = [rand(p32, n3), rand(p34, n3)]
	out = [rand(m3, p23), rand(m3, p43)]
	D = rand(m3, n3)
	node3 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 4
	id = 4
	neighbors = [3]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p43, p34)
	inp = [rand(p43, n4)]
	out = [rand(m4, p34)]
	D = rand(m4, n4)
	node4 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# TQS matrix
	T = TQSMatrix([node1, node2, node3, node4])
	tree = construct_tree(T.adjacency_list, root)

	# Form dense matrix from TQS
	Tdense = Matrix(T, tree)

	# form graph partitioned matrix
	TG = GraphPartitionedMatrix(T, tree)

	# Form again TQS realization
	tol = 1E-12
	ρ_max = Inf
	TG_TQS, tree = TQSMatrix(TG, root, tol, ρ_max)

	# assess error
	append!(error, relative_error(TG.mat, Matrix(TG_TQS, tree)))

end

# Compute basic error statistics
error_mean, error_std = mean(error), std(error)