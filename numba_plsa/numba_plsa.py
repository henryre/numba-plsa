import numba
import numpy as np

def normalize_basic(p):
    p /= p.sum(axis=len(p.shape)-1, keepdims=True)

def plsa(doc_term, n_topics, n_iter, method='basic'):
    # Get size
    n_docs, n_terms = doc_term.shape

    # Initialize distributions
    topic_doc = np.random.rand(n_docs, n_topics)
    normalize_basic(topic_doc)

    term_topic = np.random.rand(n_topics, n_terms)
    normalize_basic(term_topic)

    # Run pLSA algorithm
    print """
        Running {0} pLSA algorithm
        ============================
        Number of iterations: {1}
        Number of documents: {2}
        Number of terms: {3}
        Number of topics: {4}
        ============================
    """.format(method, n_iter, n_docs, n_terms, n_topics)
    if method == 'basic':
        return plsa_basic(doc_term, topic_doc, term_topic, n_iter)
    elif method == 'numba':
        return plsa_numba(doc_term, topic_doc, term_topic, n_iter)
    else:
        raise ValueError('Unrecognized method <{0}>'.format(method))

def plsa_basic(doc_term, topic_doc, term_topic, n_iter):
        n_docs, n_topics = topic_doc.shape
        n_terms = term_topic.shape[1]

        for i in xrange(n_iter):
            print "Running iteration {0}".format(i+1)
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

        return topic_full, topic_doc, term_topic

@numba.jit(nopython=True)
def plsa_numba(doc_term, topic_doc, term_topic, n_iter):
        n_docs, n_topics = topic_doc.shape
        n_terms = term_topic.shape[1]

        topic_full = np.zeros((n_docs, n_terms, n_topics), dtype=np.double)
        dist_table = np.zeros_like(topic_full, dtype=np.double)

        for i in xrange(n_iter):
            ### Expectation ###
            for d in xrange(n_docs):
                for t in xrange(n_terms):
                    p = topic_doc[d, :] * term_topic[:, t]
                    topic_full[d, t, :] = p / np.sum(p)
            ### Maximization ###
            # Compute full likelihood table
            for z in xrange(n_terms):
                dist_table[:, :, z] = doc_term * topic_full[:, :, z]
            # Maximize over terms conditioned on topics
            term_topic[:] = 0
            for d in xrange(n_docs):
                term_topic += dist_table[d, :, :].transpose()
            for z in xrange(n_topics):
                term_topic[z] /= np.sum(term_topic[z])
            # Maximize over topics conditioned on documents
            topic_doc[:] = 0
            for t in xrange(n_terms):
                topic_doc += dist_table[:, t, :]
            for d in xrange(n_docs):
                topic_doc[d] /= np.sum(topic_doc[d])

        return topic_full, topic_doc, term_topic

