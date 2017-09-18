import numpy as np
import gzip
import pickle
import random
import os

class features_extraction(object):

    def __init__(self, context_width, ark_file_name, save_dir, batch_size, pdf_file_dir, pdf_file_total):

        self.context_width = context_width
        self.ark_file_name = ark_file_name
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.pdf_file_dir = pdf_file_dir
        self.pdf_file_total = pdf_file_total

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

    def save_data(self, utt_mat, target_mat, batch_count):

        save_dir = self.save_dir

        with open(save_dir+"/utt_mat_"+str(batch_count), "wb") as fp: 
            pickle.dump(utt_mat, fp)
                   
        with open(save_dir+"/target_mat_"+str(batch_count), "wb") as fp:
            pickle.dump(target_mat, fp)

    def save_target_prior(self, target_dict):
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

        for target_string in target_dict.values():
            target_list = target_string.split(' ')
            if self.max_target_length < len(target_list):
                self.max_target_length = len(target_list)
            target_array = np.array(target_list, dtype=np.uint32)
            target_array_list.append(target_array)

        #create a big vector of stacked targets
        all_targets = np.concatenate(target_array_list)

        #count the number of occurences of each target
        prior = np.bincount(all_targets, minlength=self.num_labels)

        prior = prior/prior.sum()

        np.save(self.pdf_file_dir + '/prior.npy', prior)


    def data_processing(self):

        target_dict = self.make_target_dict()
        self.save_target_prior(target_dict)

        save_dir = self.save_dir
        batch_size = self.batch_size

        raw_data = open(self.ark_file_name)
        file_data = raw_data.read().split("\n")
        raw_data.close()
        
        seq_length_count = 0
        max_length = 0
        batch_count = 0
        batch_utt_count = -1
        total_number_of_utterances = 0
        target_match = False
        input_dim = 0
        utt_id = ""
        input_dim_check = False

        for line in file_data:
            list_line = line.split()

            #print list_line

            if len(list_line) > 0:

                if batch_utt_count == -1:
            
                    utt_mat = []
                    target_mat = []
                    batch_utt_count = 0


                elif batch_utt_count == batch_size:
                
                    ##################################
                    #before saving randomize the data
                    combined = list(zip(utt_mat, target_mat))
                    random.shuffle(combined)
                    utt_mat, target_mat = zip(*combined)
                    #################################

                    self.save_data(utt_mat, target_mat, batch_count)
                    
                    #save the input dim number only once
                    if input_dim_check == False:
                        input_dim = utt_mat[0].shape[1]
                        input_dim_check = True
                        #print "Input dim: " + str(input_dim)
                    
                    utt_mat = []
                    target_mat = []
                    batch_utt_count = 0
                    batch_count += 1

                if list_line[1] == "[" and list_line[0] in target_dict:

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

                        utt_mat.append(splice_fetures_per_utt)

                        #####################################
                        # target matrix code
                        target_sequence = target_dict[utt_id]
                        target_sequence_string_list = target_sequence.strip().split(' ')
                        targets_list = [int(i) for i in target_sequence_string_list]
                        targets = np.array(targets_list)
                        target_mat.append(targets)
                        #####################################

                        target_match = False

                        if seq_length_count > max_length:
                            max_length = seq_length_count

                        seq_length_count = 0
                        batch_utt_count += 1
                        total_number_of_utterances += 1

                    else:
                        seq_length_count = 0


                elif target_match:

                    fetures_list = [float(i) for i in list_line]
                    features_per_frame.append(fetures_list)
                    seq_length_count += 1

        important_info = {'train_utt_max_length': max_length, 
                   'training_batch_total': batch_count, 
                   'total_training_utterances': total_number_of_utterances, 
                   'input_dim': input_dim,
                   'num_labels':self.num_labels,
                   'train_label_max_length':self.max_target_length}

        with open(save_dir+"/train_important_info", "wb") as fp:
            pickle.dump(important_info, fp)

        return important_info

