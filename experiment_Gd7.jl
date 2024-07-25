# Construction of TQS matrix for the example graph G_c(5) in the paper
include("TQSMatrices.jl")
using .TQSMatrices
using LinearAlgebra
using Statistics
using Random

# set seed for experiments
Random.seed!(1234);

# graph G_d(7):
#         1 -- 5 -- 7
#              |    |
#              2    6 -- 4
#                   |
#                   3

# root node
root = 7

# input dimensions
n1 = 13
n2 = 15
n3 = 12
n4 = 19
n5 = 10
n6 = 11
n7 = 14
# output dimensions
m1 = 12
m2 = 14
m3 = 11
m4 = 18
m5 = 13
m6 = 11
m7 = 12
# state dimensions
p15 = 3
p51 = 2
p25 = 3
p52 = 4
p36 = 2
p63 = 3
p46 = 4
p64 = 2
p57 = 4
p75 = 5
p67 = 3
p76 = 2


NO_TRIALS = 10
error = []

for k ∈ 1:NO_TRIALS

	# node 1
	id = 1
	neighbors = [5]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p15, p51)
	inp = [rand(p15, n1)]
	out = [rand(m1, p51)]
	D = rand(m1, n1)
	node1 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 2
	id = 2
	neighbors = [5]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p25, p52)
	inp = [rand(p25, n2)]
	out = [rand(m2, p52)]
	D = rand(m2, n2)
	node2 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 3
	id = 3
	neighbors = [6]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p36, p63)
	inp = [rand(p36, n3)]
	out = [rand(m3, p63)]
	D = rand(m3, n3)
	node3 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 4
	id = 4
	neighbors = [6]
	trans = Matrix(undef, 1, 1)
	trans[1, 1] = ZeroMatrix(p46, p64)
	inp = [rand(p46, n4)]
	out = [rand(m4, p64)]
	D = rand(m4, n4)
	node4 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 5
	id = 5
	neighbors = [1, 2, 7]
	trans = Matrix(undef, 3, 3)
	trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p51, p15), rand(p51, p25), rand(p51, p75)
	trans[2, 1], trans[2, 2], trans[2, 3] = rand(p52, p15), ZeroMatrix(p52, p25), rand(p52, p75)
	trans[3, 1], trans[3, 2], trans[3, 3] = rand(p57, p15), rand(p57, p25), ZeroMatrix(p57, p75)
	inp = [rand(p51, n5), rand(p52, n5), rand(p57, n5)]
	out = [rand(m5, p15), rand(m5, p25), rand(m5, p75)]
	D = rand(m5, n5)
	node5 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 6
	id = 6
	neighbors = [3, 4, 7]
	trans = Matrix(undef, 3, 3)
	trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p63, p36), rand(p63, p46), rand(p63, p76)
	trans[2, 1], trans[2, 2], trans[2, 3] = rand(p64, p36), ZeroMatrix(p64, p46), rand(p64, p76)
	trans[3, 1], trans[3, 2], trans[3, 3] = rand(p67, p36), rand(p67, p46), ZeroMatrix(p67, p76)
	inp = [rand(p63, n6), rand(p64, n6), rand(p67, n6)]
	out = [rand(m6, p36), rand(m6, p46), rand(m6, p76)]
	D = rand(m6, n6)
	node6 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# node 7
	id = 7
	neighbors = [5, 6]
	trans = Matrix(undef, 2, 2)
	trans[1, 1], trans[1, 2] = ZeroMatrix(p75, p57), rand(p75, p67)
	trans[2, 1], trans[2, 2] = rand(p76, p57), ZeroMatrix(p76, p67)
	inp = [rand(p75, n7), rand(p76, n7)]
	out = [rand(m7, p57), rand(m7, p67)]
	D = rand(m7, n7)
	node7 = Spinner{Float64}(id, neighbors, trans, inp, out, D)

	# TQS matrix
	T = TQSMatrix([node1, node2, node3, node4, node5, node6, node7])
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