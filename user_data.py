
#pandas is being used for data manipulation and analysis
import pandas as pd
import tensorflow as tf

#The training set and test set started out as a single data set
#IMP:Adding examples to the training set usually builds a better model,adding more examples to the test enables us to better gauge the model's effectiveness
#The training set contains the examples that we'll use to train the model;
TRAIN_PATH= "E:\Learn\TensorFlow\TensorFlow_GettingStared\TensorFlow_Dataset_Train.csv"
# The test set contains the examples that we well  use to evaluate the trained model's effectiveness.
TEST_PATH= "E:\Learn\TensorFlow\TensorFlow_GettingStared\TensorFlow_Dataset_Test.csv"

#age,married,salary,employment - > Variables
#status -> Label
CSV_COLUMN_NAMES = ['age', 'married',
                   'salary', 'employment', 'status']
#type of classes ( on this case we have 2 types of status ( denied or approved))
status = ['denied', 'approved']

def download_data():
   #https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file
   #Keras utility in tensorflow is used to downloads a file from a URL if it not already in the cache.
   #As this path is from a file system, file will not be downloaded into keras cache.
   #train_path - >E:\Learn\TensorFlow\TensorFlow_GettingStared\TensorFlow_Dataset_Train.csv
   train_path = tf.keras.utils.get_file(TRAIN_PATH.split('/')[-1], TRAIN_PATH)

   #test_path ->E:\Learn\TensorFlow\TensorFlow_GettingStared\TensorFlow_Dataset_Test.csv
   test_path = tf.keras.utils.get_file(TEST_PATH.split('/')[-1], TEST_PATH)

   # print the train , test path
   return train_path, test_path

def load_data(label_name='status'):

   train_path, test_path = download_data()

   # using pandas library, read csv file and replace column values at header at position 0 in csv with CSV_COLUMN_NAMES
   train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
   #seperate features and labels data frames are created
   train_features, train_label = train, train.pop(label_name)
   #sample output  - train_features withoutlabel label data
   #S.NO    age      married   salary   employment
   #0       30          1        20000   2
   #sample output  - train_label without header and feature data
   #0   0

   # using pandas library, read csv file and replace column values at header at position 0 in csv with CSV_COLUMN_NAMES
   test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
   test_features, test_label = test, test.pop(label_name)

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