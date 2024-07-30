# Construction of TQS matrix for (the inverse of) the sequence of matrices T1, T2, 
# T3, etc. in Section 3.5 of the paper
include("TQSMatrices.jl")
using .TQSMatrices
using LinearAlgebra
using Statistics
using Random
using Plots
using SparseArrays

# set seed for experiments
Random.seed!(1234);
NO_TRIALS = 10

# settings for the construction algorithm
tol = 1E-10
ρ_max = Inf


# function to construct inverse of re
_offset_nodes(adj_list, offset) = Dict(k + offset => adj_list[k] .+ offset for k ∈ keys(adj_list))
function _random_regularbinarytreematrix(k::Int)
	if k <= 0
		error("k must be strictly positive integer")
	elseif k == 1
		A = [rand();;]
		root = 1
		adj_list = Dict(root => [])
		return A, root, adj_list
	elseif k == 2
		A = [                                                         rand() 0      rand()
			0      rand() rand()
			rand() rand() rand()]
		root = 3
		adj_list = Dict(1 => [root], 2 => [root], root => [1, 2])
		return A, root, adj_list
	else
		A11, root1, adj_list1 = _random_regularbinarytreematrix(k - 1)
		A22, _, _ = _random_regularbinarytreematrix(k - 1)
		root2 = 2 * root1
		adj_list2 = _offset_nodes(adj_list1, root1)
		A12 = zeros(size(A11, 1), size(A22, 2))
		A13 = [zeros(size(A11, 1) - 1); rand()]
		A21 = zeros(size(A22, 1), size(A11, 2))
		A23 = [zeros(size(A11, 1) - 1); rand()]
		A31 = [zeros(1, size(A11, 2) - 1) rand()]
		A32 = [zeros(1, size(A11, 2) - 1) rand()]
		A33 = rand()
		# the matrix
		A = [                                                    A11 A12 A13
			A21 A22 A23
			A31 A32 A33]
		# root    
		root = 2 * root1 + 1
		# adjecency list
		adj_list = merge(adj_list1, adj_list2)
		adj_list[root] = [root1, root2]
		append!(adj_list[root1], root)
		append!(adj_list[root2], root)
		return A, root, adj_list
	end
end
function random_regularbinarytreematrix(k::Int; return_inverse = false)
	mat, root, adj_list = _random_regularbinarytreematrix(k)
	if return_inverse
		mat = inv(mat)
	end
	nodes = collect(1:root)
	m = ones(Int, root)
	n = ones(Int, root)
	return GraphPartitionedMatrix(mat, nodes, m, n, adj_list), root
end



errors = zeros(0, 3)
for k ∈ 1:9
	tmp = []
	for j ∈ 1:NO_TRIALS
		display(k)
		T, root = random_regularbinarytreematrix(k; return_inverse = true)
		T_TQS, tree = TQSMatrix(T, root, tol, ρ_max)
		# Check if all state dimensions are equal to unity.
		for spinner ∈ values(T_TQS.spinners)
			@assert all(x -> x == 1, values(spinner.p_in))
			@assert all(x -> x == 1, values(spinner.p_out))
		end
		# Check for approximation error.
		append!(tmp, relative_error(T.mat, Matrix(T_TQS, tree)))
	end
	errors = [errors;
		k mean(tmp) std(tmp)]
end
display(errors)
