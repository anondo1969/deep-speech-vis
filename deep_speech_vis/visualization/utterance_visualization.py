'''
@author: Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@date: 14.09.2017
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@license : MIT License
'''

import tensorflow as tf
import numpy as np
from neural_network import seq_convertors
import analyze_relevance
import pickle

class Visualize_single_utterance(object):

    def __init__(self, config, train_important_information, test_important_information):

        utt_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        random_utterance_id = int (config.get('visualization', 'random_utterance_id'))
        with open(utt_dir+"/utt_"+str(random_utterance_id), "rb") as fp:
            utt_mat = pickle.load(fp)

        utt_mat = np.array(utt_mat)
        utt_mat = utt_mat.reshape(utt_mat.shape[1], utt_mat.shape[2])

        input_seq_length = [utt_mat.shape[0]]
        #pad the inputs
        utt_mat = np.append(utt_mat, np.zeros([max_length-utt_mat.shape[0], utt_mat.shape[1]]), 0)

        self.input_seq_length = input_seq_length
        self.utt_mat = utt_mat
        self.save_dir = config.get('directories', 'exp_dir') + '/NN_train_dir'
        self.max_length = test_important_information['test_utt_max_length']
        self.input_dim = train_important_information['input_dim']

        image_dir = config.get('directories', 'exp_dir') + '/heat_map_image_dir'

        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)

    def retrieved_data(self):

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config) as sess:

        #with tf.Session() as sess:

            saver = tf.train.import_meta_graph(self.save_dir+'/'+'model.ckpt-'+str(self.epoch_id)+'.meta')
            
            saver.restore(sess, self.save_dir+'/'+'model.ckpt-'+str(self.epoch_id))

            #variable_list = tf.global_variables()

            #print variable_list

            self.weights_h1 = sess.run("weights/h1_value:0")
            
            self.weights_h2 = sess.run("weights/h2_value:0")
            
            self.weights_out = sess.run("weights/weight_out_value:0")

            self.bias_b1 = sess.run("biases/b1_value:0")

            self.bias_b2 = sess.run("biases/b2_value:0")

            self.bias_out = sess.run("biases/bias_out_value:0")

            print 'Information retrieved from checkpoint: ' + str(self.epoch_id)


    def decode_data(self, epoch_id):

        self.epoch_id = epoch_id

        self.retrieved_data()
        
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        ##########################
        ### GRAPH DEFINITION
        ##########################

        g = tf.Graph()
        with g.as_default():

            decode_inputs = tf.placeholder(
                         tf.float32, shape=[self.max_length, self.input_dim], name='decode_inputs')

            decode_seq_length = tf.placeholder(
                                  tf.int32, shape=[1], name='decode_seq_length')

            split_inputs = tf.unstack(tf.expand_dims(decode_inputs, 1),
                                                            name="decode_split_inputs_op")


            nonseq_inputs = seq_convertors.seq2nonseq(split_inputs, decode_seq_length)

            # Multilayer perceptron
            layer_1 = tf.add(tf.matmul(nonseq_inputs, self.weights_h1), self.bias_b1)
            layer_1_out = tf.nn.tanh(layer_1)
    
            layer_2 = tf.add(tf.matmul(layer_1_out, self.weights_h2), self.bias_b2)
            layer_2_out = tf.nn.tanh(layer_2)
    
            logits = tf.add(tf.matmul(layer_2_out, self.weights_out), self.bias_out,
                                                                              name="logits_op")

        ##########################
        ###      EVALUATION
        ##########################

        with tf.Session(graph=g, config=config) as sess:

        #with tf.Session(graph=g) as sess:

            sess.run(tf.global_variables_initializer())

            inputs_value, layer_1_out_value, layer_2_out_value, logits_value = sess.run([nonseq_inputs, layer_1_out, layer_2_out, logits], feed_dict={decode_inputs: self.utt_mat, decode_seq_length: self.input_seq_length})
        
            results = {1: inputs_value, 2: layer_1_out_value, 3: layer_2_out_value, 4: logits_value}
            weights = {1: self.weights_h1, 2: self.weights_h2, 3: self.weights_out}
            biases = {1: self.bias_b1, 2: self.bias_b2, 3: self.bias_out}

            #np.save('in_e'+str(self.epoch_id)+'.npy', results[1])
            #np.save('l1_e'+str(self.epoch_id)+'.npy', results[2])
            #np.save('l2_e'+str(self.epoch_id)+'.npy', results[3])
            #np.save('lg_e'+str(self.epoch_id)+'.npy', results[4])

            print 'Completed decoding for epoch: ' + str(self.epoch_id)

            return results, weights, biases




















