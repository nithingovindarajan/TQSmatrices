# A "proof-of-concept" preliminary implementation of TQSMatrices
module TQSMatrices

###########
# exports #
###########
export ZeroMatrix,
	Spinner, GIRS_is_consistent, TQSMatrix, IndexedVector, IndexedMatrix, is_a_tree
export GIRS_has_no_bounce_back_operators, construct_tree, GraphPartitionedMatrix
export get_block, get_hankelblock, StateGraph, TransGraph

############
# packages #
############
using LinearAlgebra
using IterTools
using Base.Iterators



##############
# ZeroMatrix #
##############
struct ZeroMatrix{Scalar <: Number} <: AbstractMatrix{Scalar}
	m::Int
	n::Int
end
ZeroMatrix(m, n) = ZeroMatrix{Float64}(m, n)
Base.:size(A::ZeroMatrix) = (A.m, A.n)
function Base.:getindex(A::ZeroMatrix, i::Int, j::Int)
	if 0 < i <= A.m && 0 < j <= A.n
		return convert(eltype(A), 0)
	else
		throw(BoundsError(A, (i, j)))
	end
end
Base.:Matrix(A::ZeroMatrix) = zeros(eltype(A), A.m, A.n)

#################
# IndexedVector #
#################

IndexedVector{T <: Any} = Dict{Int, T}
IndexedVector{T}(array, labels) where {T <: Any} = Dict{Int, T}(zip(labels, array))
function IndexedVector{T}(v::IndexedVector) where {T <: Any}
	return Dict{Int, T}(key => val for (key, val) in v)
end

#############
# Utilities #
#############

function construct_range_vec(node_sizes::IndexedVector{Int}, nodes)
	ranges = IndexedVector{UnitRange}()
	off = 0
	for node in nodes
		ranges[node] = (off+1):(off+node_sizes[node])
		off = off + node_sizes[node]
	end
	return ranges
end


function construct_range_vec(node_sizes::Vector{Int})
	ranges = IndexedVector{UnitRange}()
	off = 0
	for k in eachindex(node_sizes)
		ranges[k] = (off+1):(off+node_sizes[k])
		off = off + node_sizes[k]
	end
	return ranges
end

#################
# IndexedMatrix #
#################

struct IndexedMatrix{T <: Any}
	array::Matrix{T}
	labels::Vector{Int}
	index_map::Dict{Int, Int}
	N::Int
	function IndexedMatrix{T}(array, labels) where {T <: Any}
		@assert size(array, 1) == size(array, 2)
		@assert size(array, 1) == length(labels)
		new{T}(
			convert(Matrix{T}, array),
			labels,
			Dict(label => i for (i, label) in enumerate(labels)),
			size(array, 1)::Int,
		)
	end
end
Base.:getindex(A::IndexedMatrix, i, j) = A.array[A.index_map[i], A.index_map[j]]
function Base.:setindex!(A::IndexedMatrix, val, i, j)
	A.array[A.index_map[i], A.index_map[j]] = val
end
function IndexedMatrix{T}(A::IndexedMatrix) where {T <: Any}     # to recast into different type parameter
	return IndexedMatrix{T}(A.array, A.labels)
end
function IndexedMatrix{T}(labels::Vector{Int}) where {T <: Any} # creates an undefined IndexedMatrix
	N = length(labels)
	return IndexedMatrix{T}(Array{Matrix{Float64}}(undef, N, N), labels)
end
Base.:size(A::IndexedMatrix) = (A.N, A.N)
Base.:size(A::IndexedMatrix, i) = size(A)[i]

############################
# Graph Partitioned matrix #
############################
struct GraphPartitionedMatrix{Scalar <: Number}
	mat::AbstractMatrix{Scalar}
	# graph attributes
	nodes::Vector{Int}
	K::Int
	adjacency_list::IndexedVector{Vector{Int}}
	# mapping between nodes and matrix entries
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}
	# dimensions
	m::IndexedVector{Int}
	n::IndexedVector{Int}
	M::Int
	N::Int
	function GraphPartitionedMatrix{Scalar}(
		mat,
		nodes,
		m,
		n,
		adjacency_list,
	) where {Scalar <: Number}
		@assert length(m) == length(n) == length(nodes)
		@assert sum(values(m)) == size(mat, 1)
		@assert sum(values(n)) == size(mat, 2)
		@assert length(Set(nodes)) == length(nodes)
		@assert all([haskey(adjacency_list, node) for node in nodes])
		@assert length(keys(adjacency_list)) == length(nodes)
		for neighbors in values(adjacency_list)
			for n in neighbors
				@assert haskey(adjacency_list, n)
			end
		end

		K = length(nodes)
		M = size(mat, 1)
		N = size(mat, 2)

		# m and n
		m = IndexedVector{Int}(m, nodes)
		n = IndexedVector{Int}(n, nodes)

		# construct ranges
		mrange = construct_range_vec(m, nodes)
		nrange = construct_range_vec(n, nodes)

		# construct 
		new{Scalar}(mat, nodes, K, adjacency_list, mrange, nrange, m, n, M, N)
	end
end
GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list) =
	GraphPartitionedMatrix{eltype(mat)}(mat, nodes, m, n, adjacency_list)
Base.:size(A::GraphPartitionedMatrix) = (A.M, A.N)
Base.:getindex(A::GraphPartitionedMatrix, i::Int, j::Int) = A.mat[i, j]
Base.eltype(A::GraphPartitionedMatrix) = typeof(A).parameters[1]
# extract a block
function get_block(A::GraphPartitionedMatrix, i, j)
	rows = vcat([A.mrange[k] for k in i]...)
	columns = vcat([A.nrange[k] for k in j]...)
	return A.mat[rows, columns]
end
get_block(A::GraphPartitionedMatrix, i::Union{Int, Vector{Int}}) = get_block(A, i, i)
function get_hankelblock(A::GraphPartitionedMatrix, j)
	i = filter(x -> !(x in j), A.nodes)
	return get_block(A, i, j)
end


###########
# Spinner #
###########
struct Spinner{Scalar <: Number}
	id::Int
	neighbors::Vector{Int}
	trans::IndexedMatrix{AbstractMatrix{Scalar}}
	inp::IndexedVector{AbstractMatrix{Scalar}}
	out::IndexedVector{AbstractMatrix{Scalar}}
	D::AbstractMatrix{Scalar}
	m::Int
	n::Int
	p_in::IndexedVector{Int}
	p_out::IndexedVector{Int}

	function Spinner{Scalar}(
		id::Int,
		neighbors::Vector{Int},
		trans::IndexedMatrix{AbstractMatrix{Scalar}},
		inp::IndexedVector{AbstractMatrix{Scalar}},
		out::IndexedVector{AbstractMatrix{Scalar}},
		D::AbstractMatrix{Scalar},
		m::Int,
		n::Int,
		p_in::IndexedVector{Int},
		p_out::IndexedVector{Int},
	) where {Scalar <: Number}

		# spinners cannot have its own node id as neighbor
		@assert all(x -> x != id, neighbors)
		# neighbor list is unique
		@assert length(Set(neighbors)) == length(neighbors)
		# trans, inp, and out, p_in, p_out have correct labels
		@assert trans.labels == neighbors
		@assert Set(keys(inp)) == Set(neighbors)
		@assert Set(keys(out)) == Set(neighbors)
		@assert Set(keys(p_in)) == Set(neighbors)
		@assert Set(keys(p_out)) == Set(neighbors)
		# dimensions of D coincide with m and n
		@assert size(D) == (m, n)
		# operator dimension checks
		@assert all([size(out[k]) == (m, p_in[k]) for k in keys(out)])
		@assert all([size(inp[k]) == (p_out[k], n) for k in keys(inp)])
		@assert all([
			size(trans[k, l]) == (p_out[k], p_in[l]) for k in neighbors, l in neighbors
		])

		new{Scalar}(id, neighbors, trans, inp, out, D, m, n, p_in, p_out)
	end

end
Base.:size(S::Spinner) = (S.m, S.n)
IndexedVector{Spinner}(S) = Dict(s.id => s for s in S)
Base.eltype(x::Spinner) = typeof(x).parameters[1]


# alternative constructor 1
function Spinner{Scalar}(
	id::Int,
	trans::IndexedMatrix{AbstractMatrix{Scalar}},
	inp::IndexedVector{AbstractMatrix{Scalar}},
	out::IndexedVector{AbstractMatrix{Scalar}},
	D::AbstractMatrix{Scalar},
) where {Scalar <: Number}
	return Spinner{Scalar}(
		id,
		trans.labels,
		trans,
		inp,
		out,
		D,
		size(D, 1),
		size(D, 2),
		Dict(j => size(c, 2) for (j, c) in out),
		Dict(j => size(b, 1) for (j, b) in inp),
	)
end

# alternative constructor 2
function Spinner{Scalar}(
	id::Int,
	neighbors::Vector{Int},
	trans,
	inp,
	out,
	D,
) where {Scalar <: Number}
	return Spinner{Scalar}(
		id,
		neighbors,
		IndexedMatrix{AbstractMatrix{Scalar}}(trans, neighbors),
		IndexedVector{AbstractMatrix{Scalar}}(inp, neighbors),
		IndexedVector{AbstractMatrix{Scalar}}(out, neighbors),
		convert(AbstractMatrix{Scalar}, D),
		size(D, 1),
		size(D, 2),
		Dict(j => size(c, 2) for (j, c) in zip(neighbors, out)),
		Dict(j => size(b, 1) for (j, b) in zip(neighbors, inp)),
	)
end


#########################################################################
# Check if vector of spinners is a valid (infinite) GIRS representation #
#########################################################################


function GIRS_is_consistent(nodeset)
	for node in values(nodeset)
		for neighbor in node.neighbors
			# node i is a neighbor of node j iff node j is a neighbor of node i
			@assert node.id in nodeset[neighbor].neighbors
			# dimension consistency
			@assert node.p_out[neighbor] == nodeset[neighbor].p_in[node.id]
		end
	end
	return true
end

#########################################
# Check if vector of spinners is a tree #
#########################################

function is_a_tree(adj_list)
	# for an indirect graph to be a tree:
	# 1. no cycles
	# 2. fully connected

	visited = Dict(node => false for node in keys(adj_list))
	start_node = first(keys(adj_list))

	# depth first search
	stack = [(start_node, -1)]
	while !isempty(stack)
		node, prev = pop!(stack)
		visited[node] = true

		for neighbor in adj_list[node]
			if !visited[neighbor]
				push!(stack, (neighbor, node))
			elseif neighbor != prev
				return false  # Back edge, so it's not a tree
			end
		end
	end

	# Check if all nodes are visited
	return all(values(visited))
end

###############################################
# Check if all bounce back operators are null #
###############################################

function GIRS_has_no_bounce_back_operators(nodeset)
	for node in values(nodeset)
		for i in node.neighbors
			if !isa(node.trans[i, i], ZeroMatrix)
				return false
			end
		end
	end
	return true
end

#########################
# Determine tree levels #
#########################

struct Tree
	adjacency_list::IndexedVector{Vector{Int}}
	root::Int
	tree_depth::Int
	levels::Vector{Vector{Int}}
	parent::IndexedVector{Union{Int, Nothing}}
	children::IndexedVector{Vector{Int}}
	siblings::IndexedVector{Vector{Int}}
	descendants::IndexedVector{Vector{Int}}
	descendants_complement::IndexedVector{Vector{Int}}
end

function construct_tree(adj_list, root)
	# function assumes nodeset is a tree. Run first GIRS_is_tree

	@assert haskey(adj_list, root)
	not_yet_visited = Dict(node => true for node in keys(adj_list))

	#breadth first search
	parent = Dict{Int, Union{Int, Nothing}}()
	parent[root] = nothing
	levels = [[root]]
	not_yet_visited[root] = false
	current_level = levels[1]
	k = 1
	while any(values(not_yet_visited))
		next_level = []
		for node in current_level
			for neighbor in adj_list[node]
				if not_yet_visited[neighbor]
					parent[neighbor] = node
					push!(next_level, neighbor)
					not_yet_visited[neighbor] = false
				end
			end
		end
		push!(levels, next_level)
		k += 1
		current_level = next_level
	end
	tree_depth = k - 1


	# determine children
	children = Dict{Int, Vector{Int}}(k => [] for k in keys(adj_list))
	for k in keys(parent)
		if !isnothing(parent[k])
			push!(children[parent[k]], k)
		end
	end


	# determine siblings
	siblings = Dict{Int, Vector{Int}}(k => [] for k in keys(adj_list))
	for childrenset in values(children)
		if !isempty(childrenset)
			for c in childrenset
				siblings[c] = filter(x -> x != c, childrenset)
			end
		end
	end


	# determine descendant and descendants complement
	descendants = Dict{Int, Vector{Int}}(k => [k] for k in keys(adj_list))
	for k in keys(descendants)
		function recursively_add_children!(x)
			if !isempty(children[x])
				for l in children[x]
					push!(descendants[k], l)
					recursively_add_children!(l)
				end
			end
		end
		recursively_add_children!(k)
	end
	nodes = keys(adj_list)
	descendants_complement =
		Dict{Int, Vector{Int}}(k => collect(setdiff(nodes, descendants[k])) for k in nodes)

	return Tree(
		adj_list,
		root,
		tree_depth,
		levels,
		parent,
		children,
		siblings,
		descendants,
		descendants_complement,
	)
end
construct_tree(T::GraphPartitionedMatrix, root) = construct_tree(T.adjacency_list, root)




################
# TQS matrices #
################
struct TQSMatrix{Scalar <: Number}
	# spinners
	spinners::IndexedVector{Spinner{Scalar}}
	# graph attributes
	node_ordering::Vector{Int}
	K::Int
	adjacency_list::IndexedVector{Vector{Int}}
	# dimensions
	m::IndexedVector{Int}
	n::IndexedVector{Int}
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}
	M::Int
	N::Int

	function TQSMatrix{Scalar}(spinners, node_ordering) where {Scalar <: Number}

		# correctness of input
		@assert all(s -> eltype(s) == Scalar, values(spinners))
		@assert all([spinners[k].id == k for k in keys(spinners)])
		@assert length(spinners) == length(node_ordering)
		@assert Set(keys(spinners)) == Set(node_ordering)

		# checks spinners generate a valid TQS matrix
		@assert GIRS_is_consistent(spinners)
		@assert GIRS_has_no_bounce_back_operators(spinners)
		adjacency_list = Dict(s.id => s.neighbors for s in values(spinners))
		@assert is_a_tree(adjacency_list)

		K = length(node_ordering)
		m = Dict(s.id => s.m for s in values(spinners))
		n = Dict(s.id => s.n for s in values(spinners))
		mrange = construct_range_vec(m, node_ordering)
		nrange = construct_range_vec(n, node_ordering)
		M = sum(values(m))
		N = sum(values(n))

		new{Scalar}(spinners, node_ordering, K, adjacency_list, m, n, mrange, nrange, M, N)
	end
end
function TQSMatrix(spinners, node_ordering)
	@assert !isempty(spinners)
	T = eltype(spinners[1])
	return TQSMatrix{T}(spinners, node_ordering)
end
Base.eltype(T::TQSMatrix) = typeof(T).parameters[1]
Base.:size(T::TQSMatrix) = (T.M, T.N)
construct_tree(T::TQSMatrix, root) = construct_tree(T.adjacency_list, root)

##############################
# TQS matrix vector multiply #
##############################
StateGraph{Scalar <: Number} = Dict{Int, Dict{Int, Vector{Scalar}}}
function StateGraph{Scalar}(T::TQSMatrix) where {Scalar <: Number}
	# creates a Stategraph with zero entries
	stategraph = StateGraph{Scalar}()
	for (node, neighbors) in T.adjacency_list
		stategraph[node] = Dict(
			neighbor => zeros(Scalar, T.spinners[node].p_in[neighbor]) for
			neighbor in neighbors
		)
	end
	return stategraph
end


function Base.:*(T::TQSMatrix, x::Vector, tree::Tree)

	#check input correctness
	@assert T.N == length(x)

	# pre-allocate memory
	type_b = promote_type(eltype(T), eltype(x))
	b = Vector{type_b}(undef, T.M)
	h = StateGraph{type_b}(T)

	# diagonal stage
	for node in eachindex(T.node_ordering)
		b[T.mrange[node]] = T.spinners[node].D * x[T.nrange[node]]
	end

	# Upsweep stage: from leaves to the root
	for l ∈ length(tree.levels):-1:2
		for i in tree.levels[l]
			j = tree.parent[i]
			h[j][i] += T.spinners[i].inp[j] * x[T.nrange[i]]   #input contribution
			for w in tree.children[i]                             #children state contribution
				h[j][i] += T.spinners[i].trans[j, w] * h[i][w]
			end
			b[T.mrange[j]] += T.spinners[j].out[i] * h[j][i]
		end
	end

	# Downsweep stage: from root to the leaves
	for l ∈ 2:length(tree.levels)
		for i in tree.levels[l]
			j = tree.parent[i]
			h[i][j] += T.spinners[j].inp[i] * x[T.nrange[j]]    # input contribution
			for s in tree.siblings[i]                              # sibling contribution
				h[i][j] += T.spinners[j].trans[i, s] * h[j][s]
			end
			k = tree.parent[j]
			if !isnothing(k)
				h[i][j] += T.spinners[j].trans[i, k] * h[j][k]  # grandparent contribution
			end
			b[T.mrange[i]] += T.spinners[i].out[j] * h[i][j]
		end
	end

	return b
end

function Base.:*(T::TQSMatrix, X::AbstractMatrix, tree::Tree)
	return mapslices(x -> *(T, x, tree), X; dims = 1)
end


#######################
# TQS to dense matrix #
#######################

function Base.:Matrix(T::TQSMatrix, tree::Tree)
	return *(T, Matrix{eltype(T)}(I, T.N, T.N), tree)
end

##########################
# low rank approximation #
##########################

function lowrankapprox(A; tol = 1E-15, ρ_max = Inf)
	U, S, V = svd(A)
	ρ = findfirst(x -> x < tol, S)
	if ρ > ρ_max
		ρ = ρ_max
	end
	X = U[:, 1:k] * Diagonal(S[1:k])
	Y = V[:, 1:k]'
	return X, Y, ρ
end


########################
# Hankel factorization #
########################


struct HankelFact
	X::AbstractMatrix
	Y::AbstractMatrix
	nodes_out::Vector{Int}
	nodes_in::Vector{Int}
	# mapping between nodes and matrix entries
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}
	# dimensions
	m::IndexedVector{Int}
	n::IndexedVector{Int}
	M::Int
	N::Int
	p::Int   # "size of factorization" - if X and Y are full rank, p is also the rank

	function HankelFact(X, Y, nodes_out, nodes_in, m, n)
		@assert length(nodes_out) == length(m)
		@assert length(nodes_in) == length(n)
		@assert size(X, 1) == sum(m)
		@assert size(Y, 2) == sum(n)
		@assert size(X, 2) == size(Y, 1)

		# M, N, p
		M = sum(m)
		N = sum(n)
		p = size(X, 2)

		# m and n
		m = IndexedVector{Int}(m, nodes_out)
		n = IndexedVector{Int}(n, nodes_in)

		# construct mrange
		mrange = construct_range_vec(m, nodes_out)
		nrange = construct_range_vec(n, nodes_in)

		new(X, Y, nodes_out, nodes_in, mrange, nrange, m, n, M, N, p)
	end
end



####################
# TQS construction #
####################
TransGraph{Scalar <: Number} = Dict{Int, IndexedMatrix{Scalar}}
function TransGraph{Scalar}(T::GraphPartitionedMatrix) where {Scalar <: Number}
	transgraph = Dict{Int, IndexedMatrix{Scalar}}(
		k => IndexedMatrix{Scalar}(v) for (k, v) in T.adjacency_list
	)
	return transgraph
end


#n alternative constructor for TQS matrix
function TQSMatrix{T}(trans, inp, out, D, node_ordering) where {T <: Number}
	spinners = Dict()
	for k in node_ordering
		spinners[k] = Spinner{T}(k, trans[k], inp[k], out[k], D[k])
	end
	return TQSMatrix{T}(spinners, node_ordering)
end

function determine_partition_sizes(T, nodes_out, nodes_in)
	m = Dict{Int, Int}(node => T.m[node] for node in nodes_out)
	n = Dict{Int, Int}(node => T.n[node] for node in nodes_in)
	mrange = Dict{Int, UnitRange}()
	off = 0
	for node in Dbari
		mrange[node] = (off+1):(off+m[node])
		off = off + m[node]
	end
	nrange = Dict{Int, UnitRange}()
	off = 0
	for node in nodes_in
		nrange[node] = (off+1):(off+n[node])
		off = off + n[node]
	end

	return m, n, mrange, nrange
end

function TQSMatrix(T::GraphPartitionedMatrix, root::Int, tol = 1E-15, ρ_max = Inf)

	# construct tree
	@assert root in T.nodes
	tree = construct_tree(T, root)

	# Initialize generators
	D = Dict{Int, Matrix{eltype(T)}}(
		node => get_block(T, node) for node in eachindex(T.node_ordering)
	)
	trans = TransGraph{eltype(T)}(T)
	inp = Dict{Int, Dict{Int, Matrix{eltype(T)}}}(
		k => Dict{Int, Matrix{eltype(T)}}() for k in T.nodes
	)
	out = Dict{Int, Dict{Int, Matrix{eltype(T)}}}(
		k => Dict{Int, Matrix{eltype(T)}}() for k in T.nodes
	)

	# generator skeleton hankel graph
	H = Dict(k => Dict() for k in T.nodes)

	# Upsweep stage: from leaves to the root
	for l ∈ length(tree.levels):-1:2
		for i in tree.levels[l]
			# parent and children
			j, w = tree.parent[i], tree.children[i]
			Dbari = tree.descendants_complement[i]
			inp_nodes = [vcat([tree.descendants[w_t] for w_t in w]...); i]

			#construct F
			F = T[Dbari, i]
			F = [hcat([getXblock(H[w_t][i], Dbari) for w_t in w]...) F]

			# compute low rank factorization of F
			X, Z, ρ = lowrankapprox(F; tol = tol, ρ_max = ρ_max)

			# Set inp, trans & out, and form Y in the process
			m, n, mrange, nrange = determine_partition_sizes(T, Dbari, inp_nodes)
			out[j][i] = X[mrange[j], :]
			Y = Array{eltype(T)}(undef, ρ, 0)
			for w_t in w
				trans[i][j, w_t] = Z[:, nrange[w_t]]
				Y = [Y trans[i][j, w_t] * H[w_t][i].Y]
			end
			inp[i][j] = Z[:, nrange[i]]
			Y = [Y inp[i][j]]

			# construct low rank factorization of Hankel block
			H[i][j] = HankelFact(X, Y)
		end
	end

	# Downsweep stage: from root to the leaves
	for l ∈ 2:length(tree.levels)
		for i in tree.levels[l]
			# parent and children
			j = tree.parent[i]
			k = tree.parent[j]
			v = tree.siblings[i]
			Di = tree.descendants[i]
			inp_nodes = isnothing(k) ? [] : tree.descendants_complement[j]
			inp_nodes = [inp_nodes; vcat([tree.descendants[v_t] for v_t in v]...); j]

			#construct F
			F = T[Di, j]
			F = [hcat([getXblock(H[v_t][j], Di) for v_t in v]...) F]
			if !isnothing(k)
				F = [getXblock(H[k][j], Di) F]
			end

			# compute low rank factorization of F
			X, Z, ρ = lowrankapprox(F; tol = tol, ρ_max = ρ_max)

			# Set trans,inp, out, and form Y in the process
			m, n, mrange, nrange = determine_partition_sizes(T, Di, inp_nodes)
			out[i][j] = X[mrange[i], :]
			if isnothing(k)
				Y = Array{eltype(T)}(undef, ρ, 0)
			else
				trans[j][i, k] = Z[:, nrange[k]]
				Y = trans[j][i, k] * H[k][j].Y
			end
			for v_t in v
				trans[j][i, v_t] = Z[:, nrange[v_t]]
				Y = [Y trans[i][j, w_t] * H[v_t][j].Y]
			end
			inp[i][j] = Z[:, nrange[i]]
			Y = [Y inp[i][j]]

			# construct low-rank factorization of Hankel block
			H[j][i] = HankelFact(X, Y)

		end
	end


	# construct TQS matrix (create alternative constructor!)
	T_TQS = TQSMatrix(trans, inp, out, D, T.nodes)


	return T_TQS, tree

end

#############
# TQS solve #
#############


end



