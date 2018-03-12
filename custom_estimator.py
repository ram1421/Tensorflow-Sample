# This example is considering the Dealertrack user data and making a pridection if the deal will be approved or rejected.
#Note: I have NOT used any PII data from any environment. The datasets (training & testing) were created based on application_dtcan table.
# What is Tensorflow ? It is a symbolic math library, and also used for machine learning applications such as neural networks.
#Tensorflow provides API in Python, C++ , Haskell , java and thrid party packages are also available in C#.
#This below sample is written in Python using python libraries like pandas

# import tensorflow from python
import tensorflow as tf
import user_data

#this is the size
#batch size - The number of examples in a one iteration
#IMP: Smaller batch sizes usually enables to train a model faster at the EXPENSE of accuracy
#100 is deafult
batch_size=100
#train_steps : this argument tells to stop training after the specified number of iterations.
#IMP: Increasing steps INCREASES the amount of time the model will train.
#training a model longer does not guarantee a better model
#1000 is default number of steps
train_steps=1000

#Choosing the right number of steps & batch_size usually requires both experience and experimentation.

def main(args):

   # Step 1 : Fetch the data
   (train_features, train_label), (test_features, test_label) = user_data.load_data()

    # Step 2: Create feature columns to describe data
   # Feature columns describe how to use the input.
   #A feature column is a data structure that tells your model how to interpret the data in each feature
   #set of all possible status in which a deal might be (approved / denied)
   my_feature_columns = []
   for key in train_features.keys(): # Keys -> 'age', 'married', 'salary', 'employment'
       my_feature_columns.append(tf.feature_column.numeric_column(key=key))# 'age', 'married', 'salary', 'employment'

    # STEP 3: Select the type of model
    # Build 2 hidden layer Deep Neural Network with 4 nodes respectively.- as seen in the image.
    # It crucial get the correct number of hidden layers , type of classifier( recurrent , convolution NN) and the number of nodes in each hidden layer.
    # Increasing the number of hidden layers and neurons typically creates a more powerful model, which requires more data to train effectively.
   classifier = tf.estimator.DNNClassifier(
       feature_columns=my_feature_columns,
       # Two hidden layers of 4 nodes each.
       hidden_units=[4,4],
       # The n_classes parameter specifies the number of possible values that the neural network can predict.
       # Since we have 2 status , we set n_classes to 2.
       n_classes=2)

    # STEP : 4 we start the training of Model using trainig data ( features + labels)with batch_size of 100 and train_steps of 1000
    # Based on the type of the activation functions ( step or sigmoid function) weights are adjusted using back propogation.
   # This model uses forward propogation

   # number of train_steps / batch_size are number of iterations ( 1000/100 = 10 iteratons) and increasing step by 100 in each iteration.
   #INFO:tensorflow:Loss for final step: 0.0049694357.
   # Train the Model. this type of training is an example of supervised machine learning as we have dataset with labels.
   classifier.train(
       input_fn=lambda:user_data.train_input_fn(train_features, train_label,
                                                batch_size),
       steps=train_steps)


   # STEP 5 : Evaluate the model is used to determine how effectively the model makes predictions.
   # We now provide the test data ( features and label)
   eval_result = classifier.evaluate(
       input_fn=lambda:user_data.eval_input_fn(test_features,
                                               test_label,
                                               batch_size))
    #Test set accuracy: 1.000
   print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))



   # STEP :6  Generate predictions from the model
   expected = ['denied']
   predict_x = {
       'age': [29],
       'married': [1],
       'salary': [2000],
       'employment': [2],
   }

    #IMP: We are not passing in the label value, as we don't know the value
   predictions = classifier.predict(
       input_fn=lambda:user_data.eval_input_fn(predict_x,
                                               labels=None,
                                               batch_size=batch_size))

   for pred_dict, expec in zip(predictions, expected):
       template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

       class_id = pred_dict['class_ids'][0]
       probability = pred_dict['probabilities'][class_id]

       print(template.format(user_data.status[class_id],
                             100 * probability, expec))


if __name__ == '__main__':
    # logging info too
   tf.logging.set_verbosity(tf.logging.INFO)
   tf.app.run(main)
