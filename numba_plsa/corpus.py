import numpy as np
from collections import defaultdict

class CorpusBuilder(object):
  def __init__(self, stopwords=None, min_len=3, max_len=12):
    self.stops = stopwords or set()
    self.mn = min_len
    self.mx = max_len
    self.vocab_table = dict()
    self.vocab_list = list()
    self.doc_table = dict()
    self.doc_list = list()
    self.docs = list()

  def clean(self, word):
    word = ''.join(c for c in word if ord(c) < 128)
    if len(word) < self.mn or len(word) > self.mx or word in self.stops:
        return None
    return word

  def add_document(self, text, name=None):
    name = name or str(len(self.docs))
    if name in self.doc_table:
      raise Exception("Document name <{0}> already in use".format(name))
    counts = defaultdict(int)
    for word in text.split():
      clean_word = self.clean(word)
      if not clean_word:
        continue
      if clean_word not in self.vocab_table:
        self.vocab_table[clean_word] = len(self.vocab_table)
        self.vocab_list.append(clean_word)
      counts[self.vocab_table[clean_word]] += 1
    self.docs.append(counts)
    self.doc_table[name] = len(self.doc_table)
    self.doc_list.append(name)

  def get_term(self, idx):
    return self.vocab_list[idx]

  def get_doc(self, idx):
    return self.doc_list[idx]

  def get_doc_term(self):
    doc_term = np.zeros((len(self.doc_table), len(self.vocab_table)))
    for i, doc in enumerate(self.docs):
      for j in doc:
        doc_term[i, j] = doc[j]
    return doc_term
