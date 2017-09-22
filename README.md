Description:

    1. A baseline speech recognition software where the neural net is trained with 
    tensorFlow and GMM training and decoding are done in kaldi toolkit.

    2. Visualization of the relevance heatmap of the utterances.
    
    3. Process of adding noise to the input features to create acoustic simulation


Requirement:

    1. kaldi
    2. tensorflow


How to Run:

1. Complete the HMM-GMM model using kaldi,

	a. Use 'run.sh' script from the 'kaldi/egs/corpus-name/s5'

	b. Take the 'alignment directory' (inside 'exp' directory) and convert all 
	the '.ali' files to '.pdf' files by doing the following,
   	(the range can be found from 'num_jobs' file)

        # run the command from the s5 directory
        . ./cmd.sh
        . ./path.sh
        . utils/parse_options.sh
        train_cmd=run.pl
        decode_cmd=run.pl

        for i in {1..10}
        do
        gunzip -c exp/tri2b_ali_si284/ali.$i.gz 
        | ali-to-pdf exp/tri2b_ali_si284/final.mdl ark:- ark,t:- 
        | gzip >  exp/tri2b_ali_si284/pdf.$i.gz
        # please keep all the pdf file in the same 'alignment directory'.
    
	c. Take the 'graph_xyz' directory (for example 'graph_nosp_tgpr' from 'tri2b' directory) and rename it 
	   to only 'graph' and copy it inside of the 'alignment directory' (for example 'tri2b_ali_si284')

	d. Copy the whole 'alignment directory' inside of the 'your-exp-dir' directory. Provide the name of this 'alignment directory' 
	   in 'gmm_name' in the 'deep_speech_vis_configuration.cfg' file. I renamed the 'tri2b_ali_si284' as 'gmm_lda_mllt' for
	   clear understanding.
	   
	e. Convert the kaldi ark files (raw test and train ark files and cmvn test and train ark files) to text files and append those in a single 
	   text file for both train and test ark files. Provide the full path of these 4 ark files in the 'deep_speech_vis_configuration.cfg' file.
	   
        # run from s5 directory
        . ./cmd.sh
        . ./path.sh
        . utils/parse_options.sh
        train_cmd=run.pl
        decode_cmd=run.pl
        
        for i in {1..20}
        do
        copy-feats ark:data/test_eval92/data_fbank_20/raw_fbank_test_eval92.$i.ark ark,t:- >> 'your-directory'/test-feats-ark.txt
        done


2. Change the required parameters in 'deep_speech_vis/deep_speech_vis_configuration.cfg' file 
   according to your choice.

3. Select required operations in 'deep_speech_vis/deep_speech_vis.py' file by selecting 'True' or 
   'False'.

4. Run the 'deep_speech_vis/deep_speech_vis.py' file.

5. See results in 'your-exp-dir/NN_train_dir/logdir/accuracy_log' file 
   and 
   'your-exp-dir/decode_dir/kaldi_decode/scoring/best_wer' file.

6. Some important information can be found in print information if you run the
   'deep_speech_vis/deep_speech_vis.py' file with a saving log.

7. You can visualize the data and graph from tensorboard in a web browser by 
   using the following command (in a terminal),

                  tensorboard --logdir=your-exp-dir/NN_train_dir/logdir
                  
8. There are several methods for calculating relevance, choose them from the configuration file.

9. You can scale the image according to your choice in terms of heat-map points by giving
   value in the configuration file.

10. See the image containing heat-map in 'your-exp-dir/heat_map_image_dir'.
