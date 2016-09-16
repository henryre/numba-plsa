# Numba pLSA
pLSA for sparse matrices implemented with Numba. Wicked fast.

### Performance comparisons

First, we compare to a popular Python pLSA package. Using the same initializations, the two methods gave the same solutions.

| Implementation | Corpus size | Vocabulary size | Number of topics | Number of iterations | Time (best of 3) |
|:--------------:|:-----------:|:---------------:|:--------------------:|:----------------:|
| [PLSA package](https://github.com/hitalex/PLSA) | 13 | 2126 | 5 | 30 | 45.7 s |
| numba-plsa, basic | 13 | 2126 | 5 | 30 | 0.075 s |

