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
TRAIN_FEATURE_EXTRACTION = True
TRAIN_NN = False
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

    context_width = int(config.get('simple_NN', 'context_width'))
    ark_file_name = config.get('directories', 'train_ark')
    save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
    batch_size = int(config.get('simple_NN', 'batch_size'))
    pdf_file_dir = config.get('directories', 'exp_dir') + '/' + config.get('general', 'gmm_name')
    pdf_file_total = int(config.get('general', 'num_pdf_files'))

    train_features = train_features_extraction.features_extraction(context_width, ark_file_name, save_dir, batch_size, pdf_file_dir, pdf_file_total)

    #before calling please check in the save directory the utterance dictionary is saved
    #already or not and give the 'utterances_save' value accordingly.
    #It will save a lot of time.
    train_important_information = train_features.batch_data_processing(utterances_save=False)

    print train_important_information

    print "Train feature extraction is completed."


if TRAIN_NN:
    
    if not train_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/train_features_dir'
        train_important_information = train_features_extraction.get_important_info(save_dir)

    n_classes = train_important_information['num_labels']
    input_dim = train_important_information['input_dim']
    total_batch = train_important_information['training_batch_total']
    max_input_length = train_important_information['train_utt_max_length']
    max_target_length = train_important_information['train_label_max_length']

    trainer = simple_NN_training.Simple_multy_layer_perceptron()
    trainer.train_NN(config, n_classes, input_dim, total_batch, max_input_length, max_target_length)

    print "Neural network training is completed"


if TEST_FEATURE_EXTRACTION:

    context_width = int(config.get('simple_NN', 'context_width'))
    ark_file_name = config.get('directories', 'test_ark')
    save_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
    batch_size = int(config.get('simple_NN', 'batch_size'))

    test_features = test_features_extraction.features_extraction(context_width, ark_file_name, save_dir, batch_size)

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

    max_length = test_important_information['test_utt_max_length']
    input_dim = train_important_information['input_dim']
    total_uttarences = test_important_information['total_test_utterances']
    load_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
    decode_dir = config.get('directories', 'exp_dir') + '/NN_decode_dir'
            
    decoder = simple_NN_decoding.Decode(config, max_length, input_dim, total_uttarences, load_dir, decode_dir)

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
        train_important_information = train_features_extraction.get_important_info(save_dir)

    if not test_important_information:
        save_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        test_important_information = test_features_extraction.get_important_info(save_dir)

    random_utterance_id = int (config.get('visualization', 'random_utterance_id'))
    total_uttarences = test_important_information['total_test_utterances']

    if random_utterance_id < 0 or random_utterance_id >= total_uttarences:

        print "Invalid random utterance ID. Please select a number between 0 & "+str(total_uttarences - 1)+'.'

    else:

        utt_dir = config.get('directories', 'exp_dir') + '/test_features_dir'
        save_dir = config.get('directories', 'exp_dir') + '/NN_train_dir'
        max_length = test_important_information['test_utt_max_length']
        input_dim = train_important_information['input_dim']
        image_dir = config.get('directories', 'exp_dir') + '/heat_map_image_dir'

        relevance_method_name = config.get('visualization', 'visualization_method')
        epsilon_value = float(config.get('visualization', 'epsilon_value'))
        alpha_value = float(config.get('visualization', 'alpha_value'))
        pixel_scaling_factor = int(config.get('visualization', 'pixel_scaling_factor'))
        extension = config.get('visualization', 'extension')

        relevance_methods = ['simple_lrp', 'flat_lrp', 'ww_lrp', 'epsilon_lrp', 'alphabeta_lrp']

        if relevance_method_name in relevance_methods:

            visualize = utterance_visualization.Visualize_single_utterance(random_utterance_id, utt_dir, save_dir, max_length, input_dim, image_dir)

            epochs = int(config.get('simple_NN', 'training_epochs'))

            for epoch in range (epochs):

                results, weights, biases = visualize.decode_data(epoch+1)

                analyze = analyze_relevance.relevance_analyzer(results, weights, biases, epoch+1, relevance_method_name, epsilon_value, alpha_value, pixel_scaling_factor, extension, image_dir)

            print "Single utterance visualization is completed"

        else:

            print "Invalid relevance method name. Please provide a valid name. options are:"
            print relevance_methods

    
