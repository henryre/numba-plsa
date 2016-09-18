import numba
import numpy as np

@numba.jit(nopython=True)
def plsa_numba(doc_term, topic_doc, term_topic, n_iter):
  n_docs, n_topics = topic_doc.shape
  n_terms = term_topic.shape[1]

  topic_full = np.zeros((n_docs, n_terms, n_topics), dtype=np.double)
  dist_table = np.zeros_like(topic_full, dtype=np.double)
  
  term_sum = np.zeros((n_topics))
  doc_sum = np.zeros((n_docs))

  for i in xrange(n_iter):
    ### Expectation ###
    for d in xrange(n_docs):
      for t in xrange(n_terms):
        p = np.zeros((n_topics))
        s = 0
        for z in xrange(n_topics):
          p[z] = topic_doc[d, z] * term_topic[z, t]
          s += p[z]
        for z in xrange(n_topics):
          topic_full[d, t, z] = p[z] / s
    ### Maximization ###
    # Compute full likelihood table
    topic_doc[:] = 0
    term_topic[:] = 0
    doc_sum[:] = 0
    term_sum[:] = 0
    for d in xrange(n_docs):
      for t in xrange(n_terms):
        for z in xrange(n_topics):
          q = doc_term[d, t] * topic_full[d, t, z]
          term_topic[z, t] += q
          term_sum[z] += q
          topic_doc[d, z] += q
          doc_sum[d] += q
    for z in xrange(n_topics):
      for t in xrange(n_terms):
        term_topic[z, t] /= term_sum[z]
    for d in xrange(n_docs):
      for z in xrange(n_topics):
        topic_doc[d, z] /= doc_sum[d]

