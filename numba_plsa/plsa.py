import numpy as np
import time
from plsa_basic import normalize_basic, plsa_basic
from plsa_numba import plsa_numba

def plsa(doc_term, n_topics, n_iter, min_count=1, method='basic'):
    # Get size
    n_docs, n_terms = doc_term.shape

    # Do min count
    keep_term = doc_term.sum(axis=0) >= min_count
    n_keep_terms = int(np.sum(keep_term))
    doc_term = doc_term[:, keep_term]
    keep_doc = doc_term.sum(axis=1) > 0
    n_keep_docs = int(np.sum(keep_doc))
    doc_term = doc_term[keep_doc, :]

    # Initialize distributions
    topic_doc = np.random.rand(n_keep_docs, n_topics)
    normalize_basic(topic_doc)

    term_topic = np.random.rand(n_topics, n_keep_terms)
    normalize_basic(term_topic)

    # Run pLSA algorithm
    print """
        Running {0} pLSA algorithm
        ============================
        Number of iterations: {1}
        Number of documents: {2} / {3} before min_count
        Number of terms: {4} / {5} before min_count
        Number of topics: {6}
        Sparsity factor: {7:.5f}
        ============================
    """.format(
      method, n_iter, n_keep_docs, n_docs,
      n_keep_terms, n_terms, n_topics,
      np.mean(doc_term != 0)
    )

    start = time.clock()
    if method == 'basic':
        p_t_d, p_w_t = plsa_basic(doc_term, topic_doc, term_topic, n_iter)
    elif method == 'numba':
        p_t_d, p_w_t = plsa_numba(doc_term, topic_doc, term_topic, n_iter)
    else:
        raise ValueError('Unrecognized method <{0}>'.format(method))
    elapsed = time.clock() - start
    print "\nAlgorithm ran in {0:.3f} seconds".format(elapsed)

    p_t_d_all = np.zeros((n_docs, n_topics))
    p_t_d_all[keep_doc, :] = p_t_d

    p_w_t_all = np.zeros((n_topics, n_terms))
    p_w_t_all[:, keep_term] = p_w_t

    return p_t_d_all, p_w_t_all
