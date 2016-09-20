import numpy as np
import scipy.sparse as sparse
import time
from numba_plsa.plsa import plsa_basic, plsa_numba, normalize_basic

def normalize_port(vec):
  s = sum(vec)
  assert(abs(s) != 0.0)
  for i in range(len(vec)):
    assert(vec[i] >= 0)
    vec[i] = vec[i] * 1.0 / s

def plsa_port(doc_term, topic_doc, term_topic, n_iter):
  n_docs, n_terms = doc_term.shape
  n_topics = topic_doc.shape[1]
  # Port of the PLSA package algorithm
  topic_prob = np.zeros([n_docs, n_terms, n_topics], dtype=np.float)
  for iteration in range(n_iter):
    for d_index in range(n_docs):
      for w_index in range(n_terms):
        prob = topic_doc[d_index, :] * term_topic[:, w_index]
        if sum(prob) == 0.0:
          exit(0)
        else:
          normalize_port(prob)
        topic_prob[d_index][w_index] = prob
    # update P(w | z)
    for z in range(n_topics):
      for w_index in range(n_terms):
        s = 0
        for d_index in range(n_docs):
          count = doc_term[d_index][w_index]
          s = s + count * topic_prob[d_index, w_index, z]
        term_topic[z][w_index] = s
      normalize_port(term_topic[z])
    # update P(z | d)
    for d_index in range(n_docs):
      for z in range(n_topics):
        s = 0
        for w_index in range(n_terms):
          count = doc_term[d_index][w_index]
          s = s + count * topic_prob[d_index, w_index, z]
        topic_doc[d_index][z] = s
      normalize_port(topic_doc[d_index])
  return topic_doc, term_topic

def set_problem(n_docs, n_terms, n_topics, sparsity=0.95):
  topic_doc = np.random.rand(n_docs, n_topics)
  normalize_basic(topic_doc)
  term_topic = np.random.rand(n_topics, n_terms)
  normalize_basic(term_topic)

  d = np.random.rand(n_docs, n_terms)
  ct = 3 * np.round(np.abs(np.random.randn(n_docs, n_terms)))
  doc_term = (d > sparsity) * (1 + ct)
  doc_term[np.random.choice(n_docs, n_terms), np.arange(n_terms)] = 1

  return doc_term, topic_doc, term_topic

def test_numba(n, r, c, d, td, tt, n_iter):
  times = []
  for t in xrange(n):
    r_c = r.copy()
    c_c = c.copy()
    d_c = d.copy()
    td_c = td.copy()
    tt_c = tt.copy()
    start = time.time()
    plsa_numba(r_c, c_c, d_c, td_c, tt_c, n_iter)
    end = time.time()
    times.append(end - start)
  return min(times), td_c, tt_c

def test_basic(n, dt, td, tt, n_iter):
  times = []
  for t in xrange(n):
    dt_c = dt.copy()
    td_c = td.copy()
    tt_c = tt.copy()
    start = time.time()
    plsa_basic(dt_c, td_c, tt_c, n_iter)
    end = time.time()
    times.append(end - start)
  return min(times), td_c, tt_c

if __name__ == '__main__':
  problem_specs = [
                   (100, 500, 10),
                   (100, 2500, 10),
                   (250, 1000, 10),
                   (1000, 5000, 10),
                   (2000, 6000, 10),
                   (3000, 5000, 10),
                  ]
  for prob in problem_specs:
    print "n_docs={0}\tn_terms={1}\tn_topics={2}".format(*prob)
    dt, td, tt = set_problem(*prob)
    sdt = sparse.coo_matrix(dt)
    n_iter = 50 
    numba_best = test_numba(3, sdt.row, sdt.col, sdt.data, td, tt, n_iter)
    print "\tnumba time / iter (best of 3):\t{0}s".format(numba_best[0] / n_iter)
    basic_best = test_basic(3, dt, td, tt, n_iter)
    print "\tbasic time / iter (best of 3):\t{0}s".format(basic_best[0] / n_iter)
    #port_best = test_port(3, dt, td, tt, n_iter)
    #print "port time / iter (best of 3):\t{0}s".format(port_best[0] / n_iter)
    
