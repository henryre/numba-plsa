import numpy as np
from plsa_basic import normalize_basic, plsa_basic
from plsa_numba import plsa_numba

def plsa(doc_term, n_topics, n_iter, min_count=1, method='basic'):
    # Get size
    n_docs, n_terms = doc_term.shape

    # Do min count
    keep_term = doc_term.sum(axis=0) >= min_count
    n_keep_terms = int(np.sum(keep_term))
    doc_term = doc_term[:, keep_term]

    # Initialize distributions
    topic_doc = np.random.rand(n_docs, n_topics)
    normalize_basic(topic_doc)

    term_topic = np.random.rand(n_topics, n_keep_terms)
    normalize_basic(term_topic)

    # Run pLSA algorithm
    print """
        Running {0} pLSA algorithm
        ============================
        Number of iterations: {1}
        Number of documents: {2}
        Number of terms: {3} / {4} before min_count
        Number of topics: {5}
        Sparsity factor: {6:.5f}
        ============================
    """.format(
      method, n_iter, n_docs, n_keep_terms, n_terms, n_topics,
      np.mean(doc_term != 0)
    )
    if method == 'basic':
        p_t, p_t_d, p_w_t = plsa_basic(doc_term, topic_doc, term_topic, n_iter)
    elif method == 'numba':
        p_t, p_t_d, p_w_t = plsa_numba(doc_term, topic_doc, term_topic, n_iter)
    else:
        raise ValueError('Unrecognized method <{0}>'.format(method))

    p_t_all = np.zeros((n_docs, n_terms, n_topics))
    p_t_all[:, keep_term, :] = p_t

    p_w_t_all = np.zeros((n_topics, n_terms))
    p_w_t_all[:, keep_term] = p_w_t

    return p_t_all, p_t_d, p_w_t_all
