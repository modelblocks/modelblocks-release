1. cue2vec_preprocessing.py (preprocessing)
- Example command: python3 cue2vec_preprocessing.py dir_for_cuegraphs dir_for_output
- Function: reads in all .cuegraphs files in dir_for_cuegraphs and outputs RGCN training data in dir_for_output

2. cuegraphs2vectors (main RGCN trainer)
- Example command: python3 my_link_predict.py -l bce -d wsj_cue2vec_reduced_sents -s batch -r vector --n-epochs 2000
- Function: reads in training data from dir_for_output and outputs a tensor of node embeddings in .pt format (e.g. wsj_cue2vec_reduced_sents_vectorrel_bceloss_20dim_2samples_1000epochs_nodes.pt)
- Note: currently uses argparse to input parameters, could easily be modified to read off parameters from a config file

3. O model trainer
- Example command: python3 train_o_models.py dir_for_output node_vector_file
- Function: reads in node embedding tensor (.pt file) and RGCN training examples to output the predicate embeddings and O-model functions in TENSORNAME_emat_omodel.txt
