'''
@author: Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@date: 26.08.2017
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@license : MIT License
'''

import tensorflow as tf
import numpy as np
import seq_convertors
import pickle
import os

class Decode(object):

    def __init__(self, config, max_length, input_dim, total_uttarences, load_dir, decode_dir):

        self.load_dir = load_dir
        self.save_dir = config.get('directories', 'exp_dir')  + '/NN_train_dir'
        self.max_length = max_length
        self.input_dim = input_dim
        self.prior = np.load(config.get('directories', 'exp_dir') + '/' + config.get('general', 'gmm_name') + '/prior.npy')
        self.total_uttarences = total_uttarences
        self.epochs = config.get('simple_NN', 'training_epochs')

        with open(load_dir + "/utt_id", "rb") as fp:
            self.utt_id_list = pickle.load(fp)

        if not os.path.isdir(decode_dir):
            os.mkdir(decode_dir)


    def retrieved_data(self):

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5


        with tf.Session(config=config) as sess:

        #with tf.Session() as sess:

            saver = tf.train.import_meta_graph(self.save_dir+'/'+'model.ckpt-'+self.epochs+'.meta')
            
            saver.restore(sess, self.save_dir+'/'+'model.ckpt-'+self.epochs)

            #variable_list = tf.global_variables()

            #print variable_list

            self.weights_h1 = sess.run("weights/h1_value:0")
            
            self.weights_h2 = sess.run("weights/h2_value:0")
            
            self.weights_out = sess.run("weights/weight_out_value:0")

            self.bias_b1 = sess.run("biases/b1_value:0")

            self.bias_b2 = sess.run("biases/b2_value:0")

            self.bias_out = sess.run("biases/bias_out_value:0")



    def decode_data(self, writer):

        self.retrieved_data()

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
            layer_1 = tf.nn.tanh(layer_1)
    
            layer_2 = tf.add(tf.matmul(layer_1, self.weights_h2), self.bias_b2)
            layer_2 = tf.nn.tanh(layer_2)
    
            logits = tf.add(tf.matmul(layer_2, self.weights_out), self.bias_out,
                                                                              name="logits_op")

            seq_logits = seq_convertors.nonseq2seq(logits, decode_seq_length, 
                                                                           len(split_inputs))
   
            decode_logits = seq_convertors.seq2nonseq(seq_logits,  decode_seq_length)

            outputs = tf.nn.softmax(decode_logits, name="final_operation")


        ##########################
        ###      EVALUATION
        ##########################

        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5


        with tf.Session(graph=g, config=config) as sess:

        #with tf.Session(graph=g) as sess:

            sess.run(tf.global_variables_initializer())

            for i in range(self.total_uttarences):

                with open(self.load_dir+"/utt_"+str(i), "rb") as fp:
                    utt_mat = pickle.load(fp)

                utt_mat = np.array(utt_mat)
                utt_mat = utt_mat.reshape(utt_mat.shape[1], utt_mat.shape[2])

                utt_id = self.utt_id_list[i]

                input_seq_length = [utt_mat.shape[0]]
                #pad the inputs
                utt_mat = np.append( utt_mat, np.zeros([self.max_length-utt_mat.shape[0], 
                                                                          utt_mat.shape[1]]), 0)

                outputs_value = sess.run('final_operation:0', feed_dict={'decode_inputs:0': utt_mat,
                                                       'decode_seq_length:0': input_seq_length})
 
                # print (outputs_value.shape)
                # print (type(outputs_value))  

  

                #get state likelihoods by dividing by the prior
                output = outputs_value/self.prior

                #floor the values to avoid problems with log
                np.where(output == 0, np.finfo(float).eps, output)

                # print (output.shape)
                # print (type(output))

                #write the pseudo-likelihoods in kaldi feature format
                writer.write_next_utt(utt_id, np.log(output))

        #close the writer
        writer.close()


