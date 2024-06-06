module TQSMatrices


###########
# exports #
###########
export ZeroMatrix, Spinner, GIRS_is_consistent, TQSMatrix, IndexedVector, GIRS_is_tree
export GIRS_has_no_bounce_back_operators, determine_tree_hierarchy, GraphPartitionedMatrix
export get_block, get_hankelblock

############
# packages #
############
using LinearAlgebra
using IterTools

##############
# ZeroMatrix #
##############
struct ZeroMatrix{Scalar <: Number} <: AbstractMatrix{Scalar}
	m::UInt
	n::UInt
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

#################################
# IndexedVector & IndexedMatrix 
#################################
IndexedVector{T <: Any} = Dict{UInt, T}
IndexedVector{T}(array, labels) where {T <: Any} = Dict{UInt, T}(zip(labels, array))
struct IndexedMatrix{T <: Any}
	array::Matrix{T}
	index_map::Dict{UInt, UInt}
	function IndexedMatrix{T}(array, labels) where {T <: Any}
		@assert size(array, 1) == size(array, 2)
		@assert size(array, 1) == length(labels)
		new{T}(convert(Matrix{T}, array), Dict(label => i for (i, label) in enumerate(labels)))
	end
end
Base.:getindex(A::IndexedMatrix, i, j) = A.array[A.index_map[i], A.index_map[j]]
IndexedMatrix(array, labels) = IndexedMatrix{eltype(array)}(array, labels)


############################
# Graph Partitioned matrix #
############################
struct GraphPartitionedMatrix{Scalar <: Number}
	mat::AbstractMatrix
	# graph attributes
	nodes::Vector{UInt}
	K::UInt
	adjacency_list::IndexedVector{Vector{UInt}}
	# mapping between nodes and matrix entries
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}
	# dimensions
	m::IndexedVector{UInt}
	n::IndexedVector{UInt}
	M::UInt
	N::UInt
	function GraphPartitionedMatrix{Scalar}(mat, nodes, m, n, adjacency_list) where {Scalar <: Number}
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
		m = IndexedVector{UInt}(m, nodes)
		n = IndexedVector{UInt}(n, nodes)

		# construct mrange
		mrange = Dict{UInt, UnitRange}()
		off = 0
		for node in nodes
			mrange[node] = (off+1):(off+m[node])
			off = off + m[node]
		end

		# construct nrange
		nrange = Dict{UInt, UnitRange}()
		off = 0
		for node in nodes
			nrange[node] = (off+1):(off+n[node])
			off = off + n[node]
		end

		# construct 
		new{Scalar}(mat, nodes, K, adjacency_list, mrange, nrange, m, n, M, N)
	end
end
GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list) = GraphPartitionedMatrix{eltype(mat)}(mat, nodes, m, n, adjacency_list)
Base.:size(A::GraphPartitionedMatrix) = (A.M, A.N)
Base.:getindex(A::GraphPartitionedMatrix, i::Int, j::Int) = A.mat[i, j]
# extract a block
function get_block(A::GraphPartitionedMatrix, i, j)
	rows = vcat([A.mrange[k] for k in i]...)
	columns = vcat([A.nrange[k] for k in j]...)
	return A.mat[rows, columns]
end
get_block(A::GraphPartitionedMatrix, i::Union{UInt, Vector{UInt}}) = get_block(A, i, i)
function get_hankelblock(A::GraphPartitionedMatrix, j)
	i = filter(x -> !(x in j), A.nodes)
	return get_block(A, i, j)
end


###########
# Spinner #
###########
struct Spinner{Scalar <: Number}
	id::UInt
	neighbors::Vector{UInt}
	trans::IndexedMatrix{AbstractMatrix{Scalar}}
	inp::IndexedVector{AbstractMatrix{Scalar}}
	out::IndexedVector{AbstractMatrix{Scalar}}
	D::AbstractMatrix{Scalar}
	m::Int
	n::Int
	p_in::IndexedVector{UInt}
	p_out::IndexedVector{UInt}

	function Spinner{Scalar}(id, neighbors, trans, inp, out, D) where {Scalar <: Number}
		# spinners cannot have its own node id as neighbor
		@assert all(x -> x != id, neighbors)
		# neighbor list is unique
		@assert length(Set(neighbors)) == length(neighbors)
		# dimensionality checks
		@assert length(neighbors) == size(trans, 1)
		@assert size(trans, 1) == size(trans, 2)
		@assert length(inp) == size(trans, 1)
		@assert length(out) == size(trans, 1)
		# operator dimension checks
		@assert all([size(el, 1) == size(D, 1) for el in out])
		@assert all([size(el, 2) == size(D, 2) for el in inp])
		for iter in eachindex(inp)
			@assert all([size(el, 1) == size(inp[iter], 1) for el in trans[iter, :]])
		end
		for iter in eachindex(out)
			@assert all([size(el, 2) == size(out[iter], 2) for el in trans[:, iter]])
		end
		# construct
		new{Scalar}(
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
end
Base.:size(S::Spinner) = (S.m, S.n)
IndexedVector{Spinner}(S) = Dict(s.id => s for s in S)
Base.eltype(x::Spinner) = typeof(x).parameters[1]



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

function GIRS_is_tree(nodeset)
	# for an indirect graph to be a tree:
	# 1. no cycles
	# 2. fully connected

	adj_list = Dict(node.id => node.neighbors for node in values(nodeset))
	visited = Dict(node.id => false for node in values(nodeset))
	start_node = first(keys(nodeset))

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

function determine_tree_hierarchy(nodeset, root)
	# function assumes nodeset is a tree. Run first GIRS_is_tree

	@assert haskey(nodeset, root)
	adj_list = Dict(node.id => node.neighbors for node in values(nodeset))
	not_yet_visited = Dict(node.id => true for node in values(nodeset))

	#breadth first search
	parent = Dict{UInt, Union{UInt, Nothing}}()
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


	# determine children now...
	children = Dict{UInt, Set{UInt}}(k => Set([]) for k in keys(nodeset))
	for k in keys(parent)
		if !isnothing(parent[k])
			push!(children[parent[k]], k)
		end
	end

	return tree_depth, levels, parent, children
end


################
# TQS matrices #
################
struct TQSMatrix{Scalar <: Number}
	# spinners
	spinners::IndexedVector{Spinner{Scalar}}
	# tree attributes
	node_ordering::Vector{UInt}
	K::UInt
	adjecency_list::IndexedVector{Vector{UInt}}
	root::UInt
	levels::Vector{Vector{UInt}}
	parent::IndexedVector{Union{UInt, Nothing}}
	children::IndexedVector{Set{UInt}}
	tree_depth::UInt
	# dimensions
	m::IndexedVector{UInt}
	n::IndexedVector{UInt}
	M::UInt
	N::UInt

	function TQSMatrix{Scalar}(spinners, node_ordering, root) where {Scalar <: Number}

		# correctness of input
		@assert all(s -> eltype(s) == Scalar, values(spinners))
		@assert all([spinners[k].id == k for k in keys(spinners)])
		@assert length(spinners) == length(node_ordering)
		@assert Set(keys(spinners)) == Set(node_ordering)
		@assert root in node_ordering

		# checks spinners generate a valid TQS matrix
		@assert GIRS_is_consistent(spinners)
		@assert GIRS_is_tree(spinners)
		@assert GIRS_has_no_bounce_back_operators(spinners)

		# construct
		K = length(node_ordering)
		adjecency_list = Dict(s.id => s.neighbors for s in values(spinners))
		tree_depth, levels, parent, children = determine_tree_hierarchy(spinners, root)
		m = Dict(s.id => s.m for s in values(spinners))
		n = Dict(s.id => s.n for s in values(spinners))
		M = sum(values(m))
		N = sum(values(n))
		new{Scalar}(spinners, node_ordering, K, adjecency_list, root, levels, parent,
			children, tree_depth, m, n, M, N)
	end
end
function TQSMatrix(spinners, node_ordering, root)
	@assert !isempty(spinners)
	T = eltype(spinners[1])
	return TQSMatrix{T}(spinners, node_ordering, root)
end
Base.:size(T::TQSMatrix) = (T.M, T.N)
function Base.:Matrix(T::TQSMatrix)
	return 0
end

##############################
# TQS matrix vector multiply #
##############################



#######################
# TQS to dense matrix #
#######################



####################
# TQS construction #
####################


#############
# TQS solve #
#############


end
