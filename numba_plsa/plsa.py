import numpy as np
import time
from plsa_basic import normalize_basic, plsa_basic
from plsa_numba import plsa_numba

class PLSAModel(object):
  def __init__(self):
    self.topic_doc = None
    self.term_topic = None

  def train(self, doc_term, n_topics, n_iter, min_count=1, method='basic'):
    self.topic_doc, self.term_topic = plsa(doc_term, n_topics, n_iter, 
                                           min_count, method)

  def top_topic_terms(self, n=10, normalized=False):
    if not normalized:
      return self.term_topic.argsort()[:, -n:]
    else:
      norm = self.term_topic / (self.term_topic.sum(axis=0) + 1e-08)
      return norm.argsort()[:, -n:]

  def top_topic_docs(self, n=10):
    return self.topic_doc.argsort(axis=0)[-n:, :]


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
        Number of documents: {2} / {3} before min_count ({4})
        Number of terms: {5} / {6} before min_count ({4})
        Number of topics: {7}
        Sparsity factor: {8:.5f}
        ============================
    """.format(
      method, n_iter, n_keep_docs, n_docs, min_count, n_keep_terms, n_terms,
      n_topics, np.mean(doc_term != 0)
    )

    start = time.clock()
    if method == 'basic':
        p_t_d, p_w_t = plsa_basic(doc_term, topic_doc, term_topic, n_iter)
    elif method == 'numba':
        p_t_d, p_w_t = plsa_numba(doc_term, topic_doc, term_topic, n_iter)
    else:
        raise ValueError('Unrecognized method <{0}>'.format(method))
    elapsed = time.clock() - start
    print "\nRan {0} iterations in {1:.3f} seconds".format(n_iter, elapsed)

    p_t_d_all = np.zeros((n_docs, n_topics))
    p_t_d_all[keep_doc, :] = p_t_d

    p_w_t_all = np.zeros((n_topics, n_terms))
    p_w_t_all[:, keep_term] = p_w_t

    return p_t_d_all, p_w_t_all
