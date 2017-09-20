'''
@author: Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@date: 26.08.2017
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@license : MIT License
'''
import tensorflow as tf
import os
import numpy as np
from six.moves import configparser
import seq_convertors
import pickle

class Simple_multy_layer_perceptron(object):

    def train_NN(self, config, train_important_information):

        ##########################
        ### DATASET
        ##########################
        
        data_dir = config.get('directories', 'exp_dir')  + '/train_features_dir'
        NN_dir = config.get('directories', 'exp_dir')  + '/NN_train_dir'

        if not os.path.isdir(NN_dir):
            os.mkdir(NN_dir)

        logdir = NN_dir + '/logdir'

        if not os.path.isdir(logdir):
            os.mkdir(logdir)

        #########################
        ### SETTINGS
        ##########################

        # Hyperparameters
        learning_rate = float(config.get('simple_NN', 'learning_rate'))

        # Architecture
        n_hidden_1 = int(config.get('simple_NN', 'n_hidden_1'))
        n_hidden_2 = int(config.get('simple_NN', 'n_hidden_2'))
        n_input = train_important_information['input_dim']
        training_epochs = int(config.get('simple_NN', 'training_epochs'))
        batch_size = int(config.get('simple_NN', 'batch_size'))
        valid_batch_number = int(config.get('simple_NN', 'valid_batches'))
        n_classes = train_important_information['num_labels']
        total_batch = train_important_information['training_batch_total']
        max_input_length = train_important_information['train_utt_max_length']
        max_target_length = train_important_information['train_label_max_length']

        ##########################
        ### GRAPH DEFINITION
        ##########################

        g = tf.Graph()
        with g.as_default():
    
            with tf.name_scope('input'):   
        
                #create the inputs placeholder
                inputs = tf.placeholder(
                tf.float32, shape=[max_input_length, batch_size, input_dim], name='features')

                #the length of all the input sequences
                input_seq_length = tf.placeholder(
                        tf.int32, shape=[batch_size],
                                                         name='input_seq_length')

                #reference labels
                targets = tf.placeholder( 
                             tf.int32, shape=[max_target_length, batch_size, 1],  
                             name='targets')

                #the length of all the output sequences
                target_seq_length = tf.placeholder(
                                   tf.int32, shape=[batch_size],
                                   name='output_seq_length')

            with tf.name_scope('inputs-processing'):

                #split the 3D input tensor in a list of batch_size*input_dim tensors
                split_inputs = tf.unstack(inputs, name='split_inputs_training_op')

                #convert the sequential data to non sequential data
                nonseq_inputs = seq_convertors.seq2nonseq(split_inputs, input_seq_length, 
                                                                           name='inputs-processing')
    
            # Model parameters
            with tf.name_scope("weights"):
    
                weights = {
                'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.1), name = "h1_value"),
                'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1), name =
                                                                                             "h2_value"),
                'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], stddev=0.1), name = 
                                                                                      "weight_out_value")
                   }

            with tf.name_scope("biases"):

                biases = {
                 'b1': tf.Variable(tf.zeros([n_hidden_1]), name = "b1_value"),
                 'b2': tf.Variable(tf.zeros([n_hidden_2]), name = "b2_value"),
                 'out': tf.Variable(tf.zeros([n_classes]), name = "bias_out_value")
                  }

            # Multilayer perceptron

            with tf.name_scope("layer-1"):

                layer_1 = tf.add(tf.matmul(nonseq_inputs, weights['h1']), biases['b1'])

                #tf.summary.histogram('soft_max_layer_1', layer_1)

                layer_1_out = tf.nn.tanh(layer_1)

                #tf.summary.histogram('activation_layer_1', layer_1)

            with tf.name_scope("layer-2"):
    
                layer_2 = tf.add(tf.matmul(layer_1_out, weights['h2']), biases['b2'])

                #tf.summary.histogram('soft_max_layer_2', layer_1)

                layer_2_out = tf.nn.tanh(layer_2)

                #tf.summary.histogram('activation_layer_2', layer_1)

            with tf.name_scope("soft-max"):
    
                logits = tf.add(tf.matmul(layer_2_out, weights['out']), biases['out'])

                #tf.summary.histogram('soft_max_layer_out', layer_1)

            with tf.name_scope("logits-processing"):

                seq_logits = seq_convertors.nonseq2seq(logits, input_seq_length, len(split_inputs), 
                                                                                name="logits-processing")

                #convert to non sequential data
                nonseq_logits = seq_convertors.seq2nonseq(seq_logits, input_seq_length, name="logits-processing")

            with tf.name_scope("targets-processing"):

                #split the 3D targets tensor in a list of batch_size*input_dim tensors
                split_targets = tf.unstack(targets)

                nonseq_targets = seq_convertors.seq2nonseq(split_targets, target_seq_length, 
                                                                               name="targets-processing")
                #make a vector out of the targets
                nonseq_targets = tf.reshape(nonseq_targets, [-1])

                #one hot encode the targets
                #pylint: disable=E1101
                end_nonseq_targets = tf.one_hot(nonseq_targets, int(nonseq_logits.get_shape()[1]))

            with tf.name_scope('cross_entropy'):

                # Loss and optimizer
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=nonseq_logits, labels=end_nonseq_targets)
                cost = tf.reduce_mean(loss, name='cost_op')

            with tf.name_scope('train'):

                #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train = optimizer.minimize(cost, name='train_op')

            with tf.name_scope('Accuracy'):

                # Prediction
                correct_prediction = tf.equal(tf.argmax(end_nonseq_targets, 1), tf.argmax(nonseq_logits, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')

            #create a summary for our cost and accuracy
            tf.summary.scalar("cost", cost)
            tf.summary.scalar("accuracy", accuracy)

            #tf.summary.histogram('histogram-cost', cost)
            #tf.summary.histogram('histogram-accuracy', accuracy)

            # merge all summaries into a single "operation" which we can execute in a session 
            summary_op = tf.summary.merge_all()

            saver = tf.train.Saver(max_to_keep=10000)


        ##########################
        ### TRAINING & EVALUATION
        ##########################

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9


        with tf.Session(graph=g, config=config) as sess:

        #with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            # create log writer object
            writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
       
            for epoch in range(training_epochs):

                avg_cost = 0.

       
                for i in range(valid_batch_number, total_batch):

                    train_batch_x = np.load(data_dir+'/batch_inputs_'+str(i)+'.npy')
                    train_batch_y = np.load(data_dir+'/batch_targets_'+str(i)+'.npy')
                    train_input_seq_length = np.load(data_dir+'/batch_input_seq_length_'+str(i)+'.npy')
                    train_target_seq_length = np.load(data_dir+'/batch_output_seq_length_'+str(i)+'.npy')
                    
                    # perform the operations we defined earlier on batch
                    _, c, summary = sess.run([train, cost, summary_op], feed_dict={inputs:  train_batch_x,
                                                                                   targets: train_batch_y,
                                                                                   input_seq_length:  train_input_seq_length,
                                                                                   target_seq_length: train_target_seq_length})
                    avg_cost += c
                    # write log
                    writer.add_summary(summary, epoch * total_batch + i)

                train_acc = 0
                for j in range(valid_batch_number, total_batch):

                    train_x = np.load(data_dir+'/batch_inputs_'+str(j)+'.npy')
                    train_y = np.load(data_dir+'/batch_targets_'+str(j)+'.npy')
                    train_x_seq_length = np.load(data_dir+'/batch_input_seq_length_'+str(j)+'.npy')
                    train_y_seq_length = np.load(data_dir+'/batch_output_seq_length_'+str(j)+'.npy')
                    
                    train_batch_acc = sess.run(accuracy, feed_dict={inputs: train_x,
                                                        targets: train_y,
                                                        input_seq_length: train_x_seq_length,
                                                        target_seq_length: train_y_seq_length})

                    train_acc += train_batch_acc

                train_acc /= (total_batch - valid_batch_number)

                valid_acc = 0
                for j in range(valid_batch_number):

                    validation_x = np.load(data_dir+'/batch_inputs_'+str(j)+'.npy')
                    validation_y = np.load(data_dir+'/batch_targets_'+str(j)+'.npy')
                    validation_x_seq_length = np.load(data_dir+'/batch_input_seq_length_'+str(j)+'.npy')
                    validation_y_seq_length = np.load(data_dir+'/batch_output_seq_length_'+str(j)+'.npy')
                    
                    validation_batch_acc = sess.run(accuracy, feed_dict={inputs: validation_x,
                                                        targets: validation_y,
                                                        input_seq_length: validation_x_seq_length,
                                                        target_seq_length: validation_y_seq_length})

                    valid_acc += validation_batch_acc

                valid_acc /= valid_batch_number

                #print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)), end="")
                #print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))
                accuracy_log_file = open(logdir+'/accuracy_log', "a")
                accuracy_log_file.write("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, avg_cost / (i + 1)))
                accuracy_log_file.write(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc)+'\n')
                accuracy_log_file.close()

                #saver.save(sess, NN_dir+'/final')
                saver.save(sess, NN_dir + '/model.ckpt', global_step=epoch+1)

         
