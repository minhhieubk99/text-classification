from pyvi import ViTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix
import string
import os
import gensim

import pickle



train_path = './DATA/Train_Full/'
test_path = './DATA/Test_Full/'


labels = os.listdir(train_path)
with open('./DATA/vietnamese-stopwords-dash.txt', encoding='utf-8') as f:
    stop_words = f.read().splitlines()


def load_data(path):
  data = []
  label = []
  for index in range(10):
    print(index)
    dir_path = path + labels[index] + '/'
    for document in os.listdir(dir_path):
      with open(dir_path+document, encoding='utf-16') as f:
        doc = f.readlines()
        doc = ' '.join(doc)      

        doc = ViTokenizer.tokenize(doc)

        doc = doc.lower() 

        doc = gensim.utils.simple_preprocess(doc)

        doc = ' '.join(doc)
        tokens = doc.split() 

        tokens = [word for word in tokens if word not in stop_words]
        tokens = ' '.join(tokens)
        data.append(tokens)
        label.append(index)
  return data, label


# x_train, y_train = load_data(train_path)

# pickle.dump(x_train, open('./DATA/x_train.pkl', 'wb'))
# pickle.dump(y_train, open('./DATA/y_train.pkl', 'wb'))

x_test, y_test = load_data(test_path)

pickle.dump(x_test, open('./DATA/x_test.pkl', 'wb'))
pickle.dump(y_test, open('./DATA/y_test.pkl', 'wb'))

