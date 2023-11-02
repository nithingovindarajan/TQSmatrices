module TreeSSSMatrices


###########
# exports #
###########
export ZeroMatrix, Spinner, get_A, get_B, get_C, get_D, get_p, TreeSSS


############
# packages #
############
using LinearAlgebra
using IterTools

##############
# ZeroMatrix #
##############
struct ZeroMatrix{Scalar<:Number} <: AbstractMatrix{Scalar}
    m::UInt
    n::UInt
end
Base.:size(A::ZeroMatrix) = (A.m, A.n)
function Base.:getindex(A::ZeroMatrix, i::Int, j::Int)
    if 0 < i <= A.m && 0 < j <= A.n
        return convert(eltype(A), 0)
    else
        throw(BoundsError(A, (i, j)))
    end
end
Base.:Matrix(A::ZeroMatrix) = zeros(eltype(A), A.m, A.n)


###########
# Spinner #
###########
struct Spinner{Scalar<:Number}
    id::UInt
    neighbors::Vector{UInt}
    index_map::Dict{UInt,UInt}
    A::Matrix{AbstractMatrix{Scalar}}
    B::Vector{AbstractMatrix{Scalar}}
    C::Vector{AbstractMatrix{Scalar}}
    D::AbstractMatrix{Scalar}
    m::Int
    n::Int
    p_in::Vector{UInt}
    p_out::Vector{UInt}

    function Spinner{Scalar}(id, neighbors, A, B, C, D) where {Scalar<:Number}
        # id is positive
        @assert id > 0
        # neighbors are positive integers
        @assert all(x -> x > 0, neighbors)
        # spinners cannot have its own node id as neighbor
        @assert all(x -> x != id, neighbors)
        # neighbor list is unique
        @assert length(Set(neighbors)) == length(neighbors)
        # dimensionality checks
        @assert length(neighbors) == size(A, 1)
        @assert size(A, 1) == size(A, 2)
        @assert length(B) == size(A, 1)
        @assert length(C) == size(A, 1)
        # operator dimension checks
        @assert all([size(el, 1) == size(D, 1) for el in C])
        @assert all([size(el, 2) == size(D, 2) for el in B])
        for iter in eachindex(B)
            @assert all([size(el, 1) == size(B[iter], 1) for el in A[iter, :]])
        end
        for iter in eachindex(C)
            @assert all([size(el, 2) == size(C[iter], 2) for el in A[:, iter]])
        end
        # construct
        new{Scalar}(
            id,
            neighbors,
            Dict(node => i for (i, node) in enumerate(neighbors)),
            convert(Matrix{AbstractMatrix{Scalar}}, A),
            convert(Vector{AbstractMatrix{Scalar}}, B),
            convert(Vector{AbstractMatrix{Scalar}}, C),
            convert(AbstractMatrix{Scalar}, D),
            size(D, 1),
            size(D, 2),
            [size(el, 2) for el in C],
            [size(el, 1) for el in B],
        )
    end
end
get_A(S::Spinner, i, j) = S.A[S.index_map[i], S.index_map[j]]
get_B(S::Spinner, i) = S.B[S.index_map[i]]
get_C(S::Spinner, j) = S.C[S.index_map[j]]
get_D(S::Spinner) = S.D
get_p_in(S::Spinner, i) = S.p_in[S.index_map[i]]
get_p_out(S::Spinner, i) = S.p_out[S.index_map[i]]
Base.:size(S::spinner) = (S.m, S.n)

#########################################################################
# Check if vector of spinners is a valid (infinite) GIRS representation #
#########################################################################


function GIRS_is_consistent(nodes::Vector{Spinner{scalar}}) where {scalar<:Number}
    no_nodes = length(nodes)
    for node in nodes
        # nodes id's n should be between 1<=n<=N
        @assert 1 <= node.id <= no_nodes
        # neighbor of id  should be between 1<=n<=N
        @assert all(x -> 1 <= x <= no_nodes, node.neighbors)
        for neighbor in node.neighbors
            # node i is a neighbor of node j iff node j is a neighbor of node i
            @assert node.id in nodes[neighbor].neighbors
            # dimension consistency
            @assert get_p_out(node, neighbor) == get_p_in(nodes[neighbor], node.id)
        end
    end

end

#########################################
# Check if vector of spinners is a tree #
#########################################

function GIRS_is_tree(nodes::Vector{Spinner{T}}) where {T<:Number}

    # create adjency list
    no_nodes = length(nodes)
    adj_list = Dict(node.id => node.neighbors for node in nodes)

    # Depth first search
    visited = falses(no_nodes)
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
    if sum(visited) == no_nodes
        return true
    else
        return false  # Some nodes are disconnected
    end
end

###############################################
# Check if all bounce back operators are null #
###############################################

function GIRS_has_no_bounce_back_operators(nodes::Vector{Spinner{T}}) where {T<:Number}
    for node in nodes
        for i in node.neighbors
            @assert isa(get_A(node, i, i), ZeroMatrix)
        end
    end
end

#######################################################
# Determine leaf order, parents and children of nodes #
#######################################################


###########
# TreeSSS #
###########
struct TreeSSS{Scalar<:Number}
    nodes::Vector{Spinner{Scalar}}
    m::Vector{UInt}
    n::Vector{UInt}
    no_nodes::UInt
    M::UInt
    N::UInt
    index_map::Dict{UInt,UInt}

    # useful tree attributes
    # tree_order::Int
    # leaf_order::Any
    # k_leafs::Vector{Int}
    # root_node::Int
    # parent::Vector{Union{Int,Nothing}}
    # children::Vector{Vector{Int}}

    function TreeSSS{Scalar}(nodes) where {Scalar<:Number}
        # checks
        @assert GIRS_is_consistent(nodes)
        @assert GIRS_is_tree(nodes)
        @assert GIRS_has_no_bounce_back_operators(nodes)

        # construct
        m = [node.m for node in nodes]
        n = [node.n for node in nodes]
        no_nodes = length(nodes)
        M = sum(m)
        N = sum(n)
        index_map = Dict(node.id => i for (i, node) in enumerate(nodes))
        new{Scalar}(nodes, m, n, no_nodes, M, N, index_map)
    end
end
get_node(T::TreeSSS, i) = T.node[T.index_map[i]]
get_m(T::TreeSSS, i) = T.m[T.index_map[i]]
get_n(T:::TreeSSS, i) = T.n[T.index_map[i]]
Base.:size(T::TreeSSS) = (T.M, T.N)


##################
# SSS to TreeSSS #
##################


##################
# HSS to TreeSSS #
##################


###########################
# TreeSSS to dense matrix #
###########################


##################################
# TreeSSS matrix vector multiply #
##################################


########################
# TreeSSS construction #
########################


#################
# TreeSSS solve #
#################


end
