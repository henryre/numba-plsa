import os
from get_data import data_extract, get_data, stopword_file
from numba_plsa.corpus import CorpusBuilder
from numba_plsa.plsa import plsa

def get_stopwords(fname):
  with open(fname, 'rb') as f:
    return set(
      line.split(' ', 1)[0].strip() for line in f if line[0] not in [' ', '|']
    )

def get_article_text(fname):
  text = ''
  with open(fname, 'rb') as f:
    for line in f:
      if 'Lines:' in line:
        break
    for line in f:
      text += line.replace('\n', ' ')
  return text

def print_title(txt):
  print "\n{0}\n{1}".format(txt, '=' * len(txt))

def main(stopword_file, data_extract):

  stops = get_stopwords(stopword_file)

  print_title("Building corpus")
  
  CB = CorpusBuilder(stopwords=stops, min_len=4, max_len=8)
  doc_count = 0
  for root, directories, fnames in os.walk(data_extract):
    for fname in fnames:
      n = os.path.join(root, fname)
      CB.add_document(name=n, text=get_article_text(n))
      doc_count += 1
      print 'Processed {0} documents\r'.format(doc_count),
  doc_term = CB.get_doc_term()
 
  print_title("\nRunning pLSA")
  n_topics, n_iter = 20, 25
  topic_doc, term_topic = plsa(doc_term, n_topics, n_iter, min_count=20)

if __name__ == '__main__':

  print_title("Fetching data")
  get_data()

  main(stopword_file, data_extract)
