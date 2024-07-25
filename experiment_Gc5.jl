# Construction of TQS matrix for the example graph G_c(5) in the paper
include("TQSMatrices.jl")
using .TQSMatrices
using LinearAlgebra
using Statistics
using Random

# set seed for experiments
Random.seed!(1234);

# graph G_a(4):
#         1 -- 3 -- 5
#              |    |
#              2    4
# NOTE: 3 and 5 are virtual nodes of zero dimensions!

# root node
root = 5

# input dimensions
n1 = 13
n2 = 15
n3 = 0        # SET TO ZERO!! 
n4 = 19
n5 = 0        # SET TO ZERO!! 
# output dimensions
m1 = 12
m2 = 14
m3 = 0        # SET TO ZERO!! 
m4 = 18
m5 = 0        # SET TO ZERO!! 
# state dimensions
p13 = 3
p31 = 2
p23 = 3
p32 = 4
p35 = 2
p53 = 3
p45 = 4
p54 = 2


NO_TRIALS = 10
error = []

for k ∈ 1:NO_TRIALS

	# node 1
	id = 1
	neighbors = [3]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p13, p31)
	inp = [rand(p13, n1)]
	out = [rand(m1, p31)]
	D = rand(m1, n1)
	node1 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 2
	id = 2
	neighbors = [3]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p23, p32)
	inp = [rand(p23, n2)]
	out = [rand(m2, p32)]
	D = rand(m2, n2)
	node2 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 3
	id = 3
	neighbors = [1, 2, 5]
	trans = Matrix(undef, 3, 3)
	trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p31, p13), rand(p31, p23), rand(p31, p53)
	trans[2, 1], trans[2, 2], trans[2, 3] = rand(p32, p13), ZeroMatrix(p32, p23), rand(p32, p53)
	trans[3, 1], trans[3, 2], trans[3, 3] = rand(p35, p13), rand(p35, p23), ZeroMatrix(p35, p53)
	inp = [rand(p31, n3), rand(p32, n3), rand(p35, n3)]
	out = [rand(m3, p13), rand(m3, p23), rand(m3, p53)]
	D = rand(m3, n3)
	node3 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 4
	id = 4
	neighbors = [5]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p45, p54)
	inp = [rand(p45, n4)]
	out = [rand(m4, p54)]
	D = rand(m4, n4)
	node4 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 5
	id = 5
	neighbors = [3, 4]
	trans = Matrix(undef, 2, 2)
	trans[1, 1], trans[1, 2] = ZeroMatrix(p53, p35), rand(p53, p45)
	trans[2, 1], trans[2, 2] = rand(p54, p35), ZeroMatrix(p54, p45)
	inp = [rand(p53, n5), rand(p54, n5)]
	out = [rand(m5, p35), rand(m5, p45)]
	D = rand(m5, n5)
	node5 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# TQS matrix
	T = TQSMatrix([node1, node2, node3, node4, node5])
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