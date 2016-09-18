import numpy as np

def normalize_basic(p):
  p /= p.sum(axis=len(p.shape)-1, keepdims=True)

def plsa_basic(doc_term, topic_doc, term_topic, n_iter):
  n_docs, n_topics = topic_doc.shape
  n_terms = term_topic.shape[1]

  for i in xrange(n_iter):
    if i % 5 == 0:
      print "Running iteration {0}".format(i)
    ### Expectation ###
    topic_full = topic_doc[:, np.newaxis, :] * term_topic.T
    normalize_basic(topic_full)
    ### Maximization ###
    # Compute full likelihood table
    dist_table = doc_term[:, :, np.newaxis] * topic_full
    # Maximize over terms conditioned on topics
    term_topic = np.sum(dist_table, axis=0).T
    normalize_basic(term_topic)
    # Maximize over topics conditioned on documents
    topic_doc = np.sum(dist_table, axis=1)
    normalize_basic(topic_doc)
 
  return topic_doc, term_topic
