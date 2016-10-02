import scipy.sparse as sparse
import numpy as np
import sys

from numba_plsa.plsa import plsa_direct

n_docs, n_terms, sparsity = 10000, 100000, 0.01
m = int(n_docs * n_terms * sparsity)
test_data_lil = sparse.lil_matrix((n_docs, n_terms))

print "Drawing"
nums = np.random.poisson(size=m, lam=0.5) + 1
x = np.random.choice(n_docs, size=m)
y = np.random.choice(n_terms, size=m)
print "Assigning"
test_data_lil[x, y] = nums

test_data_coo = test_data_lil.tocoo()

plsa_direct(test_data_coo, 5, 10)

