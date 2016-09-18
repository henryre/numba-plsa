# Numba pLSA
pLSA for sparse matrices implemented with Numba. Wicked fast.

### Usage
TKTKTK

### Performance comparisons

The numba-plsa package provides two implementations: a basic NumPy method and a numba method. We compare the two implementations on artificial problems of different sizes. These results were obtained on a standard laptop with 4 GB of RAM available. The script `speed_test.py` can be used to recreate the figures. 

| Corpus size | Vocabulary size | Number of topics | Number of iterations | Basic method time (best of 3) | Numba method time (best of 3) |
|:-----------:|:---------------:|:----------------:|:--------------------:|:-----------------------------:|:-----------------------------:|
| 100  | 500  | 10 | 10 | **0.063 s**  | 0.094 s      |
| 250  | 1000 | 10 | 10 | **0.328 s**  | 0.438 s      | 
| 1000 | 5000 | 10 | 10 | 12.906 s     | **9.625 s**  |
| 2000 | 6000 | 10 | 10 | 31.453 s     | **23.078 s** |
| 3000 | 5000 | 10 | 10 | 40.078 s     | **28.969 s** |

We can also compare numba-plsa to a popular Python package on GitHub: [PLSA](https://github.com/hitalex/PLSA). We used the example data from the PLSA repo. The two methods resulted the same distributions when using the same initializations.

| Implementation | Corpus size | Vocabulary size | Number of topics | Number of iterations | Time (best of 3) |
|:--------------:|:-----------:|:---------------:|:----------------:|:----------------:|:----------------:|
| [PLSA package](https://github.com/hitalex/PLSA) | 13 | 2126 | 5 | 30 | 45.7 s |
| numba-plsa, basic | 13 | 2126 | 5 | 30 | 0.075 s |

