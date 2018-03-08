import argparse
import tensorflow as tf

import user_data

#this is the size
#batch size ????
batch_size=100
#train_steps ???
train_steps=1000

def main(argv):

   #args = parser.parse_args(argv[1:])

   # Fetch the data
   (train_features, train_label), (test_features, test_label) = user_data.load_data()

   # Feature columns describe how to use the input.
   #A feature column is a data structure that tells your model how to interpret the data in each feature
   #set of all possible status in which a deal might be (approved / denied)
   my_feature_columns = []
   for key in train_features.keys():
       my_feature_columns.append(tf.feature_column.numeric_column(key=key))

   # Build 2 hidden layer DNN with 10, 10 units respectively.
   classifier = tf.estimator.DNNClassifier(
       feature_columns=my_feature_columns,
       # one hidden layers of 10 nodes each.
       hidden_units=[10],
       # The n_classes parameter specifies the number of possible values that the neural network can predict.
       # Since we have 2 status , we set n_classes to 2.
       n_classes=2)

   # Train the Model.
   classifier.train(
       input_fn=lambda:user_data.train_input_fn(train_features, train_label,
                                                batch_size),
       steps=train_steps)

   # Evaluate the model.
   eval_result = classifier.evaluate(
       input_fn=lambda:user_data.eval_input_fn(test_features, test_label,
                                               batch_size))

   print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

   # Generate predictions from the model
   expected = ['denied']
   predict_x = {
       'age': [29],
       'married': [1],
       'salary': [2000],
       'employment': [2],
   }

   predictions = classifier.predict(
       input_fn=lambda:user_data.eval_input_fn(predict_x,
                                               labels=None,
                                               batch_size=batch_size))

   print(predictions)
   for pred_dict, expec in zip(predictions, expected):
       template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

       class_id = pred_dict['class_ids'][0]
       probability = pred_dict['probabilities'][class_id]

       print(template.format(user_data.status[class_id],
                             100 * probability, expec))


if __name__ == '__main__':
   tf.logging.set_verbosity(tf.logging.INFO)
   tf.app.run(main)
