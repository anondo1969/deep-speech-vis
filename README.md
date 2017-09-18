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

        #run the command from the s5 directory
        for i in {1..10}
        do
        gunzip -c exp/tri2b_ali_si284/ali.$i.gz 
        | ali-to-pdf exp/tri2b_ali_si284/final.mdl ark:- ark,t:- 
        | gzip >  exp/tri2b_ali_si284/pdf.$i.gz

	c. Take the 'graph_xyz' directory (for example of 'LDA+MLLT') and rename it 
	   to only 'graph' and copy it inside of the 'alignment directory'

	d. Copy the whole 'alignment directory' inside of the 
	   'tfkaldi/expdir/your-corpus-name/' directory. 
	   (for details see the 'config_main' file)

2. Change the required parameters in 'tfkaldi/config/config_main.cfg' file 
   according to your choice

3. Select required operations in 'tfkaldi/main.py' file by selecting 'True' or 
   'False'.

4. Run the 'tfkaldi/main.py' file

5. See results in 'expdir/your-corpus-name/NN_train_dir/logdir/accuracy_log' file 
   and 
   'expdir/your-corpus-name/decode_dir/kaldi_decode/scoring/best_wer' file

6. Some important information can be found in print information if you run the
   'tfkaldi/main.py' file with a saving log.

7. You can visualize the data and graph from tensorboard in a web browser by 
   using the following command (in a terminal),

                  tensorboard --logdir=expdir/corpus-name/NN_train_dir/logdir
