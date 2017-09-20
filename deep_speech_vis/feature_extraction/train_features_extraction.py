import numpy as np
import gzip
import pickle
import random
import os
import shutil

class features_extraction(object):

    def __init__(self, config):

        self.context_width = int(config.get('simple_NN', 'context_width'))
        self.ark_file_name = config.get('directories', 'train_ark')
        self.save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
        self.batch_size = int(config.get('simple_NN', 'batch_size'))
        self.pdf_file_dir = config.get('directories', 'exp_dir') + '/' + config.get('general', 'gmm_name')
        self.pdf_file_total = int(config.get('general', 'num_pdf_files'))


        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)


        self.temp_dir = self.save_dir + '/temp_dir'
        if not os.path.isdir(self.temp_dir):
            os.mkdir(self.temp_dir)


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

    def make_target_dict(self):
        '''
        read the file containing the state alignments

        Args:
            target_path: path to the alignment file

        Returns:
            A dictionary containing
                - Key: Utterance ID
                - Value: The state alignments as a space seperated string
        '''

        #put all the alignments in one file
        all_ali_files = [self.pdf_file_dir + '/pdf.' + str(i+1) + '.gz' for i in range(self.pdf_file_total)]
        
        alignment_file_names_list = ""
        for line_index in range(len(all_ali_files)):
            if line_index == 0:
                alignment_file_names_list = alignment_file_names_list + all_ali_files[line_index]
            else:
                alignment_file_names_list = alignment_file_names_list + " " + all_ali_files[line_index]

        alignment_file_name = self.pdf_file_dir + '/pdf.all'
        align_file = open(alignment_file_name, "w")
        align_file.write(alignment_file_names_list)
        align_file.close()

        target_dict = {}

        raw_data = open(alignment_file_name)
        file_data = raw_data.read().split(' ')
        raw_data.close()

        for zip_file in file_data:
            zip_file = zip_file.replace("\n", "")
	    with gzip.open(zip_file, 'rb') as fid:
                for line in fid:
                    splitline = line.strip().split(' ')
                    target_dict[splitline[0]] = ' '.join(splitline[1:])

        return target_dict

    def save_target_prior(self):
        '''
        compute the count of the targets in the data

        Returns:
            a numpy array containing the counts of the targets
        '''

        #get number of output labels
        numpdfs = open(self.pdf_file_dir + '/graph/num_pdfs')
        num_labels = numpdfs.read()
        self.num_labels = int(num_labels[0:len(num_labels)-1])
        numpdfs.close()

        self.max_target_length = 0
        target_array_list = []

        for target_string in self.target_dict.values():
            target_list = target_string.split(' ')
            if self.max_target_length < len(target_list):
                self.max_target_length = len(target_list)
            target_array = np.array(target_list, dtype=np.uint32)
            target_array_list.append(target_array)

        #create a big vector of stacked targets
        all_targets = np.concatenate(target_array_list)

        #count the number of occurences of each target
        prior = np.bincount(all_targets, minlength=self.num_labels)

        prior = prior.astype(np.float32)

        prior = prior/prior.sum()

        np.save(self.pdf_file_dir + '/prior_my_new_calculation.npy', prior)


    def get_target_array(self, utt_id):

        target_sequence = self.target_dict[utt_id]
        target_sequence_string_list = target_sequence.strip().split(' ')
        targets_list = [int(i) for i in target_sequence_string_list]
        targets = np.array(targets_list)
 
        return targets

    def get_utterance_array(self):

        seq_length_count = 0
        max_length = 0
        total_number_of_utterances = 0
        target_match = False
        input_dim = 0
        utt_id = ""
        input_dim_check = False
        utt_dict = {}

        raw_data = open(self.ark_file_name)
        file_data = raw_data.read().split("\n")
        raw_data.close()

        for line in file_data:
            list_line = line.split()

            if len(list_line) > 0:
  
                if list_line[1] == "[" and list_line[0] in self.target_dict:

                    target_match = True
                    utt_id = list_line[0]
                    features_per_frame = []
                    

                elif list_line[-1] == "]" and target_match:
    
                    del list_line[-1]
                    seq_length_count += 1

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    fetures_per_utt = np.array(features_per_frame)

                    splice_fetures_per_utt, splice_done = self.splice(fetures_per_utt)

                    if splice_done:
                        utt_dict[utt_id] = total_number_of_utterances
                        np.save(self.temp_dir + '/utt_'+str(total_number_of_utterances)+'.npy', splice_fetures_per_utt)

                        target_match = False

                        if seq_length_count > max_length:
                            max_length = seq_length_count

                        seq_length_count = 0
                        total_number_of_utterances += 1
                        print 'utt_no: ' + str(total_number_of_utterances) + ' utt_id: ' + utt_id

                        #get the input dim number only once
                        if input_dim_check == False:
                            input_dim = splice_fetures_per_utt.shape[1]
                            input_dim_check = True

                    else:
                        seq_length_count = 0


                elif target_match:

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    seq_length_count += 1

        self.max_length = max_length
        self.total_number_of_utterances = total_number_of_utterances
        self.input_dim = input_dim

        with open(self.save_dir+"/utt_dict", "wb") as fp:
            pickle.dump(utt_dict, fp)

        return utt_dict

    def save_batch_data(self, kaldi_batch_data, kaldi_batch_labels, step):
        
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

        print 'Batch ' + str(step + 1) + ' completed'


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


    def batch_data_processing(self):

        self.target_dict = self.make_target_dict()

        self.save_target_prior()

        utt_dict = self.get_utterance_array()

        utt_id_list = utterances.keys()
        random.shuffle(utt_id_list)

        utt_mat = []
        target_mat = []
        batch_count = 0

        for id_count in range(len(utt_id_list)):

            utt_key = utt_id_list[id_count]
            utt_id = utt_dict[utt_key]
            utt_array = np.load(self.temp_dir + '/utt_'+str(utt_id)+'.npy')
            target_array = self.get_target_array(utt_key)
            utt_mat.append(utt_array)
            target_mat.append(target_array)

            if (id_count + 1) % self.batch_size == 0:
 
                self.save_batch_data(utt_mat, target_mat, batch_count)
                utt_mat = []
                target_mat = []
                batch_count += 1

        important_info = {'train_utt_max_length': self.max_length, 
                   'training_batch_total': batch_count + 1, 
                   'total_training_utterances': self.total_number_of_utterances, 
                   'input_dim': self.input_dim,
                   'num_labels':self.num_labels,
                   'train_label_max_length':self.max_target_length}

        with open(self.save_dir+"/train_important_info", "wb") as fp:
            pickle.dump(important_info, fp)

        #shutil.rmtree(self.temp_dir)

        return important_info


def get_important_info(save_dir):

    with open(save_dir+"/train_important_info", "rb") as fp:
        important_info = pickle.load(fp)

    return important_info

