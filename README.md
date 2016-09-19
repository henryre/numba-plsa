# Numba pLSA
pLSA for sparse matrices implemented with Numba. Wicked fast.

### Installation

1. Clone the repo: ```git clone https://github.com/henryre/numba-plsa.git ```
2. Install NumPy: ```pip install numpy``` or ```conda install numpy```
3. Install numba: ```pip install numba``` or ```conda install numba```
4. Run the example: ```python example.py```
5. Cash out.

### Usage

The `plsa` method wraps the basic algorithmic functionality. Two very basic classes are included to assist with topic modeling tasks for text corpora. The `Corpus` class takes on text documents and can build a document-term matrix. The `PLSAModel` class has a `train` method which provides an interface to `plsa`.

For an example, see the example (conveniently named `example.py`).

### Performance comparisons

The numba-plsa package provides two implementations: a basic NumPy method and a numba method. We compare the two implementations on artificial problems of different sizes. These results were obtained on a standard laptop with 4 GB of RAM available. The script `speed_test.py` can be used to recreate the figures. 

| Corpus size | Vocabulary size | Number of topics | Number of iterations | Basic method time (best of 3) | Numba method time (best of 3) |
|:-----------:|:---------------:|:----------------:|:--------------------:|:-----------------------------:|:-----------------------------:|
| 100  | 500  | 10 | 10 | 0.05 s  | **0.00 s**  |
| 250  | 1000 | 10 | 10 | 0.25 s  | **0.05 s**  | 
| 1000 | 5000 | 10 | 10 | 0.23 s  | **0.03 s**  |
| 2000 | 6000 | 10 | 10 | 9.77 s  | **1.00 s** |
| 3000 | 5000 | 10 | 10 | 30.31 s | **2.95 s** |

We can also compare numba-plsa to a popular Python package on GitHub: [PLSA](https://github.com/hitalex/PLSA). We used the example data from the PLSA repo. The two methods resulted the same distributions when using the same initializations.

| Implementation | Corpus size | Vocabulary size | Number of topics | Number of iterations | Time (best of 3) |
|:--------------:|:-----------:|:---------------:|:----------------:|:----------------:|:----------------:|
| [PLSA package](https://github.com/hitalex/PLSA) | 13 | 2126 | 5 | 30 | 45.70 s |
| numba-plsa, basic | 13 | 2126 | 5 | 30 | 0.08 s |

