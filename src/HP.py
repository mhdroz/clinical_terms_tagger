#####################################################
#            Parameters for this project            #
#####################################################

#Parameters
dataset = 'CVD_train'
note_type = 'office_notes_2012_2016'
bucket_name = 'cvd-cohort-20210102'
blob = 'office_notes/progress_notes_2012_2016_3.parquet'
labeling_model = 'LM'
BatchID = 'BATCHID'
threads = 'THREADS'
restart_batch = 4
batchid = 93

#Paths
#path_project = '/share/pi/stamang/covid/'
path_termino = '/home/jupyter/res/term/'
path_output = '/home/jupyter/marie/outputs/meddra_extracter/'+dataset+'/'+note_type+'/'



#Datasets
#TOCHECK tagged_dataset = path_project + 'data/subset/dataset'+dataset+BATCHID+'_medDRA_CUI_tagged.parquet'
base_termino = path_termino + 'clever_base_terminology.txt'
meddraID_CUI_table = path_termino + 'termMedDRA.txt'
llt_to_pt = path_termino + 'llt_to_pt.txt'
meddra_hier = path_termino + 'med_hier.txt'
termino_matcher = path_termino + 'meddra_termino.txt'
#X_lstm = path_project + 'data/tagged/Dataset'+dataset+BATCHID+'_X_lstm.p'
#train_set = path_project + 'data/mini_batches/testD_batch'+BATCHID+'.parquet'
#raw_notes = path_project + 'data/notes_'+dataset+'/mini_batches/notes_covidBATCHID.parquet'

#NLP
spacy_model = '/home/jupyter/res/matchers/matcher_en_core_sci_md/'
matcher = '/home/jupyter/res/matchers/matcher_en_core_sci_md/meddra_v23_matcher.p'


#Outputs
extracted_hier = path_output + 'extracted_notes_BATCHID.parquet'

snorkel_preds = path_output + 'labeling_models/'+labeling_model+'_preds_dataset'+dataset+'BATCHID.p'
snorkel_probs = path_output + 'labeling_models/'+labeling_model+'_probs_dataset'+dataset+'BATCHID.p'
labeled_dataset = path_output + '/WL_BATCHID.parquet'
labeled_clean = path_output + 'WL_BATCHID_clean.parquet'

