import numpy as np
import gzip
import pickle
import random
import os

class features_extraction(object):

    def __init__(self, context_width, ark_file_name, save_dir, batch_size):

        self.context_width = context_width
        self.ark_file_name = ark_file_name
        self.save_dir = save_dir
        self.batch_size = batch_size

        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    def splice(self, utt):
        '''
        splice the utterance

        Args:
            utt: numpy matrix containing the utterance features to be spliced
            context_width: how many frames to the left and right should
                be concatenated

        Returns:
            a numpy array containing the spliced features, if the features are
            too short to splice None will be returned
        '''

        context_width = self.context_width

        #return None if utterance is too short
        if utt.shape[0]<1+2*context_width:
            return utt, False

        #create spliced utterance holder
        utt_spliced = np.zeros(
        shape=[utt.shape[0], utt.shape[1]*(1+2*context_width)],
        dtype=np.float32)

        #middle part is just the uttarnce
        utt_spliced[:, context_width*utt.shape[1]:
                (context_width+1)*utt.shape[1]] = utt

        for i in range(context_width):

            #add left context
            utt_spliced[i+1:utt_spliced.shape[0],
                    (context_width-i-1)*utt.shape[1]:
                    (context_width-i)*utt.shape[1]] = utt[0:utt.shape[0]-i-1, :]

            #add right context
            utt_spliced[0:utt_spliced.shape[0]-i-1,
                    (context_width+i+1)*utt.shape[1]:
                    (context_width+i+2)*utt.shape[1]] = utt[i+1:utt.shape[0], :]

        return utt_spliced, True


    def save_data(self, utt, utt_count):

        save_dir = self.save_dir

        with open(save_dir+"/utt_"+str(utt_count), "wb") as fp: 
            pickle.dump(utt, fp)


    def data_processing(self):
        
        save_dir = self.save_dir

        raw_data = open(self.ark_file_name)
        file_data = raw_data.read().split("\n")
        raw_data.close()

        utt_id = []
        seq_length_count = 0
        max_length = 0
        total_number_of_utterances = 0

        for line in file_data:
            list_line = line.split()

            if len(list_line) > 0:

                if list_line[1] == "[":

                        utt_id.append(list_line[0])
                        features_per_frame = []
                        utt_mat = []


                elif list_line[-1] == "]":
    
                    del list_line[-1]

                    seq_length_count += 1

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    fetures_per_utt = np.array(features_per_frame)
                    splice_fetures_per_utt, splice_done = self.splice(fetures_per_utt)

                    if splice_done:

                        utt_mat.append(splice_fetures_per_utt)
                        self.save_data(utt_mat, total_number_of_utterances)
                        
                        if seq_length_count > max_length:
                            max_length = seq_length_count

                        seq_length_count = 0
                        total_number_of_utterances += 1

                    else:

                        del utt_id[-1]
                        seq_length_count = 0

                else:

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    seq_length_count += 1

        with open(save_dir+"/utt_id", "wb") as fp:
            pickle.dump(utt_id, fp)

        important_info = {'test_utt_max_length': max_length,  
                   'total_test_utterances': total_number_of_utterances}

        with open(save_dir+"/test_important_info", "wb") as fp:
            pickle.dump(important_info, fp)

        return important_info

