'''
@author: Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@date: 21.09.2017
@version: 1.0+
@copyright: Copyright (c)  2017-2018, Mahbub Ul Alam (alammb@ims.uni-stuttgart.de)
@license : MIT License
'''

'''@file main.py
run this file to go through the neural net training procedure, look at the 'deep_speech_vis_configuration.cfg' file to modify the settings'''

import os
from six.moves import configparser
from feature_extraction import train_features_extraction, test_features_extraction
from neural_network import simple_NN_training, simple_NN_decoding
from kaldi_processes import ark
from visualization import utterance_visualization, analyze_relevance
import numpy as np
import pickle

#select which one to run by 'True'. It is arranged sequentially and independently. So, no need to re-run
# any module succefully more than once

# All modules
TRAIN_FEATURE_EXTRACTION = False
TRAIN_NN = True
TEST_FEATURE_EXTRACTION = False
DECODE_NN = False
DECODE_KALDI = False
VISUALIZE_UTTERANCE = False

#read config file
config = configparser.ConfigParser()
config.read('deep_speech_vis_configuration.cfg')

current_dir = os.getcwd()

train_important_information = {}
test_important_information = {}

if TRAIN_FEATURE_EXTRACTION:
   
    train_features = train_features_extraction.features_extraction(config)

    train_important_information = train_features.batch_data_processing()

    print train_important_information

    print "Train feature extraction is completed."



if TRAIN_NN:
    
    if not train_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
        train_important_information = train_features_extraction.get_important_info(save_dir)

    trainer = simple_NN_training.Simple_multy_layer_perceptron()
    trainer.train_NN(config, train_important_information)

    print "Neural network training is completed"


if TEST_FEATURE_EXTRACTION:

    test_features = test_features_extraction.features_extraction(config)

    test_important_information = test_features.data_processing()

    print test_important_information

    print "Test feature extraction is completed."


if DECODE_NN:

    if not train_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
        train_important_information = train_features_extraction.get_important_info(save_dir)

    if not test_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        test_important_information = test_features_extraction.get_important_info(save_dir)
            
    decoder = simple_NN_decoding.Decode(config, train_important_information, test_important_information)

    decode_dir = config.get('directories', 'exp_dir') + '/NN_decode_dir'
    #create an ark writer for the likelihoods
    if os.path.isfile(decode_dir + '/likelihoods.ark'):
        os.remove(decode_dir + '/likelihoods.ark')
    
    writer = ark.ArkWriter(decode_dir + '/feats.scp', decode_dir + '/likelihoods.ark')

    decoder.decode_data(writer)

    print "Neural network decoding is completed"

if DECODE_KALDI:

    print '------- decoding testing sets using kaldi decoder ----------'

    decode_dir = config.get('directories', 'exp_dir') + '/NN_decode_dir'

    #copy the gmm model and some files to speaker mapping to the decoding dir
    os.system('cp %s %s' %(config.get('directories', 'exp_dir') + '/' + config.get('general', 'gmm_name') + '/final.mdl', decode_dir))
    os.system('cp -r %s %s' %(config.get('directories', 'exp_dir') + '/' + config.get('general', 'gmm_name') + '/graph', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/utt2spk', decode_dir))
    os.system('cp %s %s' %(config.get('directories', 'test_data') + '/text', decode_dir))

    #change directory to kaldi egs
    os.chdir(config.get('directories', 'kaldi_egs'))

    #decode using kaldi
    os.system('%s/kaldi_processes/decode.sh --cmd %s --nj %s %s/graph %s %s/kaldi_decode | tee %s/decode.log || exit 1;' % (current_dir, config.get('general', 'cmd'), config.get('general', 'num_jobs'), decode_dir, decode_dir, decode_dir, decode_dir))
   

    #go back to working dir
    os.chdir(current_dir)

    print "Kaldi decoding is completed"

if VISUALIZE_UTTERANCE:

    if not train_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
        train_info = train_features_extraction.get_important_info(save_dir)

    if not test_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        test_info = test_features_extraction.get_important_info(save_dir)

    random_utterance_id = int (config.get('visualization', 'random_utterance_id'))
    total_uttarences = test_important_information['total_test_utterances']

    if random_utterance_id < 0 or random_utterance_id >= total_uttarences:

        print "Wrong random utterance ID. Please pick a number between 0 & "+str(total_uttarences - 1)+'.'

    else:

        relevance_methods = ['simple_lrp', 'flat_lrp', 'ww_lrp', 'epsilon_lrp', 'alphabeta_lrp']

        if relevance_method_name in relevance_methods:

            visualize = utterance_visualization.Visualize_single_utterance(config, train_info, test_info)

            epochs = int(config.get('simple_NN', 'training_epochs'))

            for epoch in range (epochs):

                results, weights, biases = visualize.decode_data(epoch+1)

                analyze = analyze_relevance.relevance_analyzer(results, weights, biases, epoch+1, config)

            print "Single utterance visualization is completed"

        else:

            print "Invalid relevance method name. Please provide a valid name. options are:"
            print relevance_methods

    
