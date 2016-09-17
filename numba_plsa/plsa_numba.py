import numba
import numpy as np

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
            for z in xrange(n_topics):
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

        return topic_doc, term_topic
