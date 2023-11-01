module TreeSSSMatrices


###########
# exports #
###########
export ZeroMatrix, Spinner


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
    neighbors::Vector{Uint}
    index_map::Dict{UInt,UInt}
    A::Matrix{AbstractMatrix{Scalar}}
    B::Vector{AbstractMatrix{Scalar}}
    C::Vector{AbstractMatrix{Scalar}}
    D::AbstractMatrix{Scalar}
    m::Int
    n::Int
    p::Vector{UInt}

    function Spinner{Scalar}(id, neighbors, A, B, C, D) where {Scalar<:Number}

        # id is positive
        @assert id > 0

        # neighbors are positive integers
        @assert all(x -> x > 0, neighbors)

        # spinners cannot have its own node id as neighbor
        @assert any(x -> x == id, neighbors)

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
        new{T}(
            id,
            neighbors,
            Dict(node => i for (i, node) in enumerate(neighbors)),
            convert(Matrix{AbstractMatrix{Scalar}}, A),
            convert(Matrix{AbstractMatrix{Scalar}}, B),
            convert(Matrix{AbstractMatrix{Scalar}}, C),
            convert(Matrix{AbstractMatrix{Scalar}}, D),
            size(D, 1),
            size(D, 2),
            [size(el, 1) for el in B],
        )
    end
end
function get_A(spinner::Spinner, i, j)
    return spinner.A[spinner.index_map[i], spinner.index_map[j]]
end
function get_B(spinner::Spinner, i)
    return spinner.B[spinner.index_map[i]]
end
function get_C(spinner::Spinner, j)
    return spinner.C[spinner.index_map[j]]
end
function get_D(spinner::Spinner)
    return spinner.D
end
function get_p(spinner::Spinner,i)
    return spinner.p[spinner.index_map[i]]
end


#########################################################################
# Check if vector of Spinners is a valid (infinite) GIRS representation #
#########################################################################

# a vector of spinners of size N:
# - the nodes id should correspond with the index in the vector
# - neighbor index n should  1<=n<=N
# - spinner i is a neighbor of spinner j iff spinner j is a neighbor of spinner i
# - dimension checks


#########################################
# Check if vector of Spinners is a tree #
#########################################


###############################################
# Check if all bounce back operators are null #
###############################################


#######################################################
# Determine leaf order, parents and children of nodes #
#######################################################


###########
# TreeSSS #
###########
struct TreeSSS{Scalar<:Number}
    spinners::Vector{Spinner{Scalar}}
    m::Vector[UInt]
    n::Vector[UInt]
    no_nodes::Int

    # useful tree attributes
    tree_order::Int
    leaf_order::Any
    k_leafs::Vector{Int}
    root_node::Int
    parent::Vector{Union{Int,Nothing}}
    children::Vector{Vector{Int}}
end


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
