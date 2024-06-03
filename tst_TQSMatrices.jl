include("TQSMatrices.jl")
using .TQSMatrices
using LinearAlgebra
using Test


######################################
# Test 1: construct spinner matrices #
######################################

#         3 -- 1 -- 2 
#              |
#         4 -- 6 -- 5

# input dimensions
n1 = 6
n2 = 8
n3 = 7
n4 = 9
n5 = 5
n6 = 7
# output dimensions
m1 = 6
m2 = 8
m3 = 7
m4 = 9
m5 = 5
m6 = 7
# state dimensions
p12 = 3
p21 = 2
p13 = 3 
p31 = 2
p12 = 2
p21 = 2
p46 = 4
p64 = 5
p56 = 2
p65 = 1
p16 = 2
p61 = 1

id = 1;
neighbors = [2, 3, 6];
trans = Matrix(undef, 3, 3);
trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p12, p21), rand(p12, p31), rand(p12, p61);
trans[2, 1], trans[2, 2], trans[2, 3] = rand(p13, p21), ZeroMatrix(p13, p31), rand(p13, p61);
trans[3, 1], trans[3, 2], trans[3, 3] = rand(p16, p21), rand(p16, p31), ZeroMatrix(p16, p61);
inp = [rand(p12,m1),
       rand(p13,m1),
       rand(p16,m1)];
out = [rand(n1,p21), rand(n1,p31), rand(n1,p61) ];
D = rand(n1, m1);
node1 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node1.trans[2,6] == trans[1,3]
@test node1.inp[2] == inp[1]
@test node1.out[3] == out[2]
@test node1.D == D


id = 2;
neighbors = [1];
trans = Matrix(undef, 1, 1);
trans[1,1] = ZeroMatrix(p21,p12)
inp = [rand(p21, n2)];
out = [rand(m2,p12)];
D = rand(m2, n2);
node2 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node2.trans[1,1] == trans[1,1]
@test node2.inp[1] == inp[1]
@test node2.out[1] == out[1]
@test node2.D == D

id = 3;
neighbors = [1];
trans = Matrix(undef, 1, 1);
trans[1,1] = ZeroMatrix(p31,p13)
inp = [rand(p31, n3)];
out = [rand(m3,p13)];
D = rand(m3, n3);
node3 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node3.trans[1,1] == trans[1,1]
@test node3.inp[1] == inp[1]
@test node3.out[1] == out[1]
@test node3.D == D

id = 4;
neighbors = [6];
trans = Matrix(undef, 1, 1);
trans[1,1] = ZeroMatrix(p46,p64)
inp = [rand(p46, n4)];
out = [rand(m4, p64)];
D = rand(m4, n4);
node4 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node4.trans[6,6] == trans[1,1]
@test node4.inp[6] == inp[1]
@test node4.out[6] == out[1]
@test node4.D == D

id = 5;
neighbors = [6];
trans = Matrix(undef, 1, 1);
trans[1,1] = ZeroMatrix(p56,p65)
inp = [rand(p56, n5)];
out = [rand(m5, p65)];
D = rand(m5, n5);
node5 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node5.trans[6,6] == trans[1,1]
@test node5.inp[6] == inp[1]
@test node5.out[6] == out[1]
@test node5.D == D

id = 6;
neighbors = [1,4,5];
trans = Matrix(undef, 3, 3);
trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p61, p16), rand(p61, p46), rand(p61, p56);
trans[2, 1], trans[2, 2], trans[2, 3] = rand(p64, p16), ZeroMatrix(p64, p46), rand(p64, p56);
trans[3, 1], trans[3, 2], trans[3, 3] = rand(p65, p16), rand(p65, p46), ZeroMatrix(p65, p56);
inp = [rand(p61,m6),
       rand(p64,m6),
       rand(p65,m6)];
out = [rand(n6,p16), rand(n6,p46), rand(n6,p56)];
D = rand(m6, n6);
node6 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node6.trans[5,4] == trans[3,2]
@test node6.inp[5] == inp[3]
@test node6.out[4] == out[2]
@test node6.D == D


nodeset_acyclic = IndexedVector{Spinner}([node1, node2, node3, node4, node5, node6]) 

# A non tree graph:
#              1 -- 2 -- 5
#              |    |
#              4 -- 3

# input dimensions
n1 = 6
n2 = 8
n3 = 7
n4 = 9
n5 = 5
# output dimensions
m1 = 6
m2 = 8
m3 = 7
m4 = 9
m5 = 5
# state dimensions
p12 = 3
p21 = 2
p14 = 3 
p41 = 2
p25 = 3 
p52 = 2
p14 = 2
p41 = 2
p23 = 2
p32 = 2
p43 = 2
p34 = 2


id = 1;
neighbors = [2, 4];
trans = Matrix(undef, 2, 2);
trans[1, 1], trans[1, 2] = ZeroMatrix(p12, p21), rand(p12, p41)
trans[2, 1], trans[2, 2] = rand(p14, p21), ZeroMatrix(p14, p41)
inp = [rand(p12,m1),
       rand(p14,m1)];
out = [rand(n1,p21), rand(n1,p31)];
D = rand(n1, m1);
node1 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node1.trans[2,4] == trans[1,2]
@test node1.inp[2] == inp[1]
@test node1.out[4] == out[2]
@test node1.D == D

id = 2;
neighbors = [1, 5, 3];
trans = Matrix(undef, 3, 3);
trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p21, p12), rand(p21, p52), rand(p21, p32);
trans[2, 1], trans[2, 2], trans[2, 3] = rand(p25, p12), ZeroMatrix(p25, p52), rand(p25, p32);
trans[3, 1], trans[3, 2], trans[3, 3] = rand(p23, p12), rand(p23, p52), ZeroMatrix(p23, p32);
inp = [rand(p21,m2),
       rand(p25,m2),
       rand(p23,m2)];
out = [rand(n2,p12), rand(n2,p52), rand(n2,p32)];
D = rand(n2, m2);
node2 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node2.trans[3,1] == trans[3,1]
@test node2.inp[5] == inp[2]
@test node2.out[1] == out[1]
@test node2.D == D


id = 3;
neighbors = [2, 4];
trans = Matrix(undef, 2, 2);
trans[1, 1], trans[1, 2] = ZeroMatrix(p32, p23), rand(p32, p43)
trans[2, 1], trans[2, 2] = rand(p34, p23), ZeroMatrix(p34, p43)
inp = [rand(p32,m3),
       rand(p34,m3)];
out = [rand(n3,p23), rand(n3,p43)];
D = rand(n3, m3);
node3 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node3.trans[2,4] == trans[1,2]
@test node3.inp[4] == inp[2]
@test node3.out[2] == out[1]
@test node3.D == D


id = 4;
neighbors = [1, 3];
trans = Matrix(undef, 2, 2);
trans[1, 1], trans[1, 2] = rand(p41, p14), rand(p41, p34)
trans[2, 1], trans[2, 2] = rand(p43, p14), ZeroMatrix(p43, p34)
inp = [rand(p41,m4),
       rand(p43,m4)];
out = [rand(n4,p14), rand(n4,p34)];
D = rand(n4, m4);
node4 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node4.trans[1,3] == trans[1,2]
@test node4.inp[3] == inp[2]
@test node4.out[1] == out[1]
@test node4.D == D


id = 5;
neighbors = [2];
trans = Matrix(undef, 1, 1);
trans[1,1] = ZeroMatrix(p52,p25)
inp = [rand(p52, n5)];
out = [rand(m5,p25)];
D = rand(m5, n5);
node5 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node5.trans[2,2] == trans[1,1]
@test node5.inp[2] == inp[1]
@test node5.out[2] == out[1]
@test node5.D == D

nodeset_cyclic = IndexedVector{Spinner}([node1, node2, node3, node4, node5]) 

########################################################
# Test 2: Checking if vector of spinners is consistent #
########################################################

@assert GIRS_is_consistent(nodeset_acyclic)
@assert GIRS_is_consistent(nodeset_cyclic)

########################################################
# Test 3: Checking if vector of spinners is consistent #
########################################################

@assert GIRS_has_no_bounce_back_operators(nodeset_acyclic)
@assert !GIRS_has_no_bounce_back_operators(nodeset_cyclic)

#######################################################################
# Test 4: Checking if vector of spinners has no bounce back operators #
#######################################################################