import numpy as np
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
  doc_term[:, np.random.permutation(n_terms)] = 1
  doc_term[np.random.permutation(n_docs), :] = 1

  return doc_term, topic_doc, term_topic

def test(f, n, doc_term, topic_doc, term_topic, n_iter):
  times = []
  for t in xrange(n):
    doc_term_c = doc_term.copy()
    topic_doc_c = topic_doc.copy()
    term_topic_c = term_topic.copy()
    start = time.clock()
    f(doc_term_c, topic_doc_c, term_topic_c, n_iter)
    end = time.clock()
    times.append(end - start)
  return min(times), topic_doc, term_topic

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
    doc_term, topic_doc, term_topic = set_problem(*prob)
    numba_best = test(plsa_numba, 3, doc_term, topic_doc, term_topic, 10)
    print "numba time (best of 3):\t{0}s".format(numba_best[0])
    basic_best = test(plsa_basic, 3, doc_term, topic_doc, term_topic, 10)
    print "basic time (best of 3):\t{0}s".format(basic_best[0])
    #port_best = test(plsa_port, 3, doc_term, topic_doc, term_topic, 10)
    #print "port time (best of 3):\t{0}s".format(port_best[0])
    if (np.allclose(numba_best[1], basic_best[1])): #and 
        #np.allclose(numba_best[1], port_best[1])):
      print "All topic_doc distributions close!"
    else:
      print "topic_doc distributions not close"
    if (np.allclose(numba_best[2], basic_best[2])): #and 
        #np.allclose(numba_best[2], port_best[2])):
      print "All term_topic distributions close!"
    else:
      print "term_topic distributions not close"
    
    
