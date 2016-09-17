import os
import tarfile
from urllib import urlretrieve

data_dir = 'data'
data_file = os.path.join(data_dir, 'mini_newsgroups.tar.gz')
data_extract = os.path.join(data_dir, 'mini_newsgroups')
stopword_file = os.path.join(data_dir, 'stop.txt')
data_uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz'
stopword_uri = 'http://snowball.tartarus.org/algorithms/english/stop.txt'

def get_data():
  if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
  if not os.path.isfile(data_file) and not os.path.isdir(data_extract):
    print "Downloading newsgroups data"
    urlretrieve(data_uri, data_file)
  if not os.path.isdir(data_extract):
    print "Extracting newsgroups data"
    tar = tarfile.open(data_file, "r:gz")
    tar.extractall(path=data_extract)
    tar.close()
  if os.path.isfile(data_file):
    os.remove(data_file)
  if not os.path.isfile(stopword_file):
    print "Downloading stopwords"
    urlretrieve(stopword_uri, stopword_file)