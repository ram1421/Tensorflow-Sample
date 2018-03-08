import pandas as pd
import tensorflow as tf

#TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
#TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
#TRAIN_URL="https://docs.google.com/spreadsheets/d/e/2PACX-1vR-M1JDEWwh9XBIf_ImdvzdWQTpjAhGzKK8cCPhhdnsN6nB8TEkOYitBjchc9CrjpB_AFmp0G8OpZof/pub?gid=853462602&single=true&output=csv"
#TEST_URL="https://docs.google.com/spreadsheets/d/e/2PACX-1vR-M1JDEWwh9XBIf_ImdvzdWQTpjAhGzKK8cCPhhdnsN6nB8TEkOYitBjchc9CrjpB_AFmp0G8OpZof/pub?gid=853462602&single=true&output=csv"

TRAIN_URL="E:\Learn\TensorFlow\TensorFlow_GettingStared\TensorFlow_Dataset_Test.csv"
TEST_URL="E:\Learn\TensorFlow\TensorFlow_GettingStared\TensorFlow_Dataset_Test.csv"

CSV_COLUMN_NAMES = ['age', 'married',
                   'salary', 'employment', 'status']
status = ['denied', 'approved']

def maybe_download():
   #https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
   #If an absolute path /path/to/file.txt is specified the file will be saved at that location.
   train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
   test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

   # print the train , test path
   ## train_path now holds the pathname: ~/.keras/datasets/iris_training.csv
   return train_path, test_path

def load_data(y_name='status'):
   """Returns the iris dataset as (train_features, train_label), (test_features, test_label)."""
   train_path, test_path = maybe_download()


   train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
   # what does these values have....
   train_features, train_label = train, train.pop(y_name)

   test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
   test_features, test_label = test, test.pop(y_name)

   return (train_features, train_label), (test_features, test_label)


def train_input_fn(features, labels, batch_size):
   """An input function for training"""
   # Convert the inputs to a Dataset.
   dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

   # Shuffle, repeat, and batch the examples.
   dataset = dataset.shuffle(1000).repeat().batch(batch_size)

   # Return the dataset.
   return dataset


def eval_input_fn(features, labels, batch_size):
   """An input function for evaluation or prediction"""
   features=dict(features)
   if labels is None:
       # No labels, use only features.
       inputs = features
   else:
       inputs = (features, labels)

   # Convert the inputs to a Dataset.
   dataset = tf.data.Dataset.from_tensor_slices(inputs)

   # Batch the examples
   assert batch_size is not None, "batch_size must not be None"
   dataset = dataset.batch(batch_size)

   # Return the dataset.
   return dataset