import numpy as np

def normalize_basic(p):
    p /= p.sum(axis=len(p.size)-1, keepdims=True)

def plsa(doc_term, n_topics, n_iter, method='basic'):
    # Get size
    n_docs, n_terms = doc_term.size

    # Initialize distributions
    topic_doc = np.random.rand(n_docs, n_topics)
    normalize_basic(topic_doc)

    term_topic = np.random.rand(n_topics, n_terms)
    normalize_basic(term_topic)

    # Run pLSA algorithm
    if method == 'basic':
        return plsa_basic(doc_term, topic_doc, term_topic, n_iter)
    else:
        raise ValueError('Unrecognized method <{0}>'.format(method))

def plsa_basic(doc_term, topic_doc, term_topic, n_iter):
        n_docs, n_topics, n_terms = topic_doc.shape + [term_topic.shape[1]]
        print """
            Running basic pLSA algorithm
            ============================
            Number of iterations: {0}
            Number of documents: {1}
            Number of terms: {2}
            Number of topics: {3}
            ============================
        """.format(n_iter, n_docs, n_terms, n_topics)

        for i in range(n_iter):
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

        return topic_full, topic_doc, term_topic
