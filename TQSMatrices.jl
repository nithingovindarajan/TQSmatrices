module TQSMatrices


###########
# exports #
###########
export ZeroMatrix, Spinner, GIRS_is_consistent, TQS, IndexedVector, GIRS_is_tree, GIRS_has_no_bounce_back_operators


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
ZeroMatrix(m,n) = ZeroMatrix{Float64}(m,n)
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
IndexedVector{T<:Any} = Dict{UInt,T}
IndexedVector{T}(array , labels) where {T<:Any} = Dict{UInt,T}(zip(labels, array)) 
struct IndexedMatrix{T<:Any}
    array::Matrix{T}
    index_map::Dict{UInt,UInt}
    function IndexedMatrix{T}(array , labels) where {T<:Any}
        @assert size(array,1) == size(array,2)
        @assert size(array,1) == length(labels)
        new{T}(convert(Matrix{T},array),  Dict(label => i for (i, label) in enumerate(labels)) )
    end
end
Base.:getindex(A::IndexedMatrix,i,j) = A.array[A.index_map[i],A.index_map[j]]
IndexedMatrix(array,labels) = IndexedMatrix{eltype(array)}(array,labels)
    

###########
# Spinner #
###########
struct Spinner{Scalar<:Number}
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

    function Spinner{Scalar}(id, neighbors, trans, inp, out, D) where {Scalar<:Number}
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
            IndexedMatrix{AbstractMatrix{Scalar}}(trans,neighbors),
            IndexedVector{AbstractMatrix{Scalar}}(inp, neighbors),
            IndexedVector{AbstractMatrix{Scalar}}(out, neighbors),
            convert(AbstractMatrix{Scalar}, D),
            size(D, 1),
            size(D, 2),
            Dict(j => size(c,2) for (j, c) in zip(neighbors,out)),
            Dict(j => size(b,1) for (j, b) in zip(neighbors,inp)),
        )
    end
end
Base.:size(S::Spinner) = (S.m, S.n)
IndexedVector{Spinner}(S) = Dict(s.id => s for s in S)


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
    stack = [(start_node,-1)]
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
            if !isa(node.trans[i,i], ZeroMatrix)
                return false
            end
        end
    end
    return true
end

#########################
# Determine tree levels #
#########################




# ################
# # TQS matrices #
# ################
# struct TQS{Scalar<:Number}
#     nodes::Vector{Spinner{Scalar}}
#     m::Vector{UInt}
#     n::Vector{UInt}
#     no_nodes::UInt
#     M::UInt
#     N::UInt
#     index_map::Dict{UInt,UInt}

#     # useful tree attributes
#     # tree_order::Int
#     # leaf_order::Any
#     # k_leafs::Vector{Int}
#     # root_node::Int
#     # parent::Vector{Union{Int,Nothing}}
#     # children::Vector{Vector{Int}}

#     function TQS{Scalar}(nodes) where {Scalar<:Number}
#         # checks
#         @assert GIRS_is_consistent(nodes)
#         @assert GIRS_is_tree(nodes)
#         @assert GIRS_has_no_bounce_back_operators(nodes)

#         # construct
#         m = [node.m for node in nodes]
#         n = [node.n for node in nodes]
#         no_nodes = length(nodes)
#         M = sum(m)
#         N = sum(n)
#         index_map = Dict(node.id => i for (i, node) in enumerate(nodes))
#         new{Scalar}(nodes, m, n, no_nodes, M, N, index_map)
#     end
# end
# get_node(T::TQS, i) = T.node[T.index_map[i]]
# get_m(T::TQS, i) = T.m[T.index_map[i]]
# get_n(T:::TreeSSS, i) = T.n[T.index_map[i]]
# Base.:size(T::TQS) = (T.M, T.N)


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
