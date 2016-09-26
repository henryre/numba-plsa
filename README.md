# Numba pLSA
pLSA for sparse matrices implemented with Numba. Wicked fast.

### Installation

1. Clone the repo: ```git clone https://github.com/henryre/numba-plsa.git ```
2. Install NumPy: ```pip install numpy``` or ```conda install numpy```
3. Install numba: ```pip install numba``` or ```conda install numba```
4. Run the example: ```python example.py```
5. Cash out.

### Usage

The numba-plsa package provides two implementations: a basic NumPy method and a numba method. The `plsa` method wraps the basic algorithmic functionality, and the algorithm is chosen by using the `method` argument (`method='numba'` or `method='basic'`, the default). The basic method works for any NumPy document-term matrix, whereas the numba method is optimized for sparse matrices. The `plsa` method automatically converts the input document-term matrix to a COO sparse matrix.

Two very basic classes are included to assist with topic modeling tasks for text corpora. The `Corpus` class takes on text documents and can build a document-term matrix. The `PLSAModel` class has a `train` method which provides an interface to `plsa`.

For an example, see the example (conveniently named `example.py`). The numba method runs in under a second on a standard laptop with 4 GB of RAM available. The [20 newsgroups](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) data set, which contains 2,000 documents, is used for evaluation. Assuming NumPy seeds play nice cross-OS, the results should be

```
Top topic terms
================
Topic 1: boswell, yalcin, onur, wright, mbytes
Topic 2: premiums, yeast, vitamin, sinus, candida
Topic 3: ports, pci, stereo, ankara, istanbul
Topic 4: icons, atari, lsd, cyprus, apps
Topic 5: wires, neutral, circuit, wiring, wire
Topic 6: gifs, simtel, jfif, gif, jpeg
Topic 7: nhl, sleeve, gant, players, league
Topic 8: mormon, gaza, xxxx, israeli, arabs
Topic 9: chi, det, suck, cubs, pit
Topic 10: cramer, theism, odwyer, bds, clayton
```

We can assign coherent labels to most topics, such as "pharmaceuticals" for Topic 2, "middle east" for Topic 8, and "baseball" for Topic 9. Adjusting corpus construction parameters, running for more iterations, or changing the number of topics can yield even better results.

### Performance comparisons

We compare the two implementations on artificial problems of different sizes, all with document-term matrix sparsity around 95% (which is fairly dense for a text-based corpus). These results were obtained on a standard laptop with 4 GB of RAM available. The script `speed_test.py` can be used to recreate the figures. 

| Corpus size | Vocab size | Number of topics | Basic method avg. time / iteration (best of 3) | Numba method avg. time  / iteration (best of 3) |
|:-----------:|:---------------:|:----------------:|:----------------------------------------------:|:-----------------------------------------------:|
| 100  | 500  | 10 | 0.0047 s | **0.00058 s** |
| 250  | 1000 | 10 | 0.024 s  | **0.0028 s**  |
| 100  | 2500 | 10 | 0.026 s  | **0.0028 s**  |
| 1000 | 5000 | 10 | 1.16 s   | **0.042 s**   |
| 2000 | 6000 | 10 | 2.59 s   | **0.12 s**    |
| 3000 | 5000 | 10 | 3.26 s   | **0.13 s**    |

We can also compare numba-plsa to a popular Python package on GitHub: [PLSA](https://github.com/hitalex/PLSA). We used the example data from the PLSA repo. The two methods resulted the same distributions when using the same initializations.

| Implementation | Corpus size | Vocab size | Number of topics | Number of iterations | Time (best of 3) |
|:--------------:|:-----------:|:---------------:|:----------------:|:----------------:|:----------------:|
| [PLSA package](https://github.com/hitalex/PLSA) | 13 | 2126 | 5 | 30 | 44.89 s |
| numba-plsa, basic | 13 | 2126 | 5 | 30 | 0.082 s |
| **numba-plsa, numba** | 13 | 2126 | 5 | 30 | **0.006 s** |
