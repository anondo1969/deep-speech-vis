'''
@author: Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@date: 26.08.2017
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@license : MIT License
'''

import os
import numpy as np
import pickle

class Padded_Batch_Data_Save(object):
   

    def __init__(self, feat_dir, save_dir, total_batch_number, max_length):

        self.feat_dir = feat_dir
        self.save_dir = save_dir
        self.total_batch_number = total_batch_number
        self.max_length = max_length

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_batch_data(self):

        for step in range (self.total_batch_number):

            with open(self.feat_dir+"/utt_mat_"+str(step), "rb") as fp:
                kaldi_batch_data = pickle.load(fp)

            with open(self.feat_dir+"/target_mat_"+str(step), "rb") as fp:
                kaldi_batch_labels = pickle.load(fp)
 
            
            
            processed_batch_inputs, processed_batch_targets, processed_batch_input_seq_length, processed_batch_output_seq_length =self.process_kaldi_batch_data(kaldi_batch_data, kaldi_batch_labels)

            np.save(self.save_dir+'/batch_inputs_'+str(step)+'.npy', processed_batch_inputs)
            np.save(self.save_dir+'/batch_targets_'+str(step)+'.npy', processed_batch_targets)
            np.save(self.save_dir+'/batch_input_seq_length_'+str(step)+'.npy', processed_batch_input_seq_length)
            np.save(self.save_dir+'/batch_output_seq_length_'+str(step)+'.npy', processed_batch_output_seq_length)

            batch_data_information_file = open(self.save_dir+'/batch_data_information', "a")
            batch_data_information_file.write(self.save_dir+'/batch_inputs_'+str(step)+'.npy'+'\n')
            batch_data_information_file.write(self.save_dir+'/batch_targets_'+str(step)+'.npy'+'\n')
            batch_data_information_file.write(self.save_dir+'/batch_input_seq_length_'+str(step)+'.npy'+'\n')
            batch_data_information_file.write(self.save_dir+'/batch_output_seq_length_'+str(step)+'.npy'+'\n')
            batch_data_information_file.close()
            
            #if step == 1:
                #break


    def process_kaldi_batch_data(self, inputs, targets):
        
        #get a list of sequence lengths
        input_seq_length = [i.shape[0] for i in inputs]
        output_seq_length = [t.shape[0] for t in targets]

        #pad all the inputs qnd targets to the max_length and put them in
        #one array
        padded_inputs = np.array([np.append(
            i, np.zeros([self.max_length-i.shape[0], i.shape[1]]), 0)
                                  for i in inputs])

        padded_targets = np.array([np.append(
            t, np.zeros(self.max_length-t.shape[0]), 0)
                                   for t in targets])

        #transpose the inputs and targets so they fit in the placeholders
        batch_inputs = padded_inputs.transpose([1, 0, 2])
        batch_targets = padded_targets.transpose()

        batch_targets = batch_targets[:, :, np.newaxis]

        return batch_inputs, batch_targets, input_seq_length, output_seq_length

