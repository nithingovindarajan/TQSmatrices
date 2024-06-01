





A = Matrix(undef, 3, 3);
A[1, 1], A[1, 2], A[1, 3] = rand(3, 4), rand(3, 2), rand(3, 3);
A[2, 1], A[2, 2], A[2, 3] = rand(4, 4), rand(4, 2), rand(4, 3);
A[3, 1], A[3, 2], A[3, 3] = rand(2, 4), rand(2, 2), rand(2, 3);

labels = [3,2,1]

tst = IndexedTable(A,labels)
tst2 = IndexedTable{Matrix{ComplexF64}}(A,labels)
tst[1,1] == tst.array[3,3]
tst2[1,1]