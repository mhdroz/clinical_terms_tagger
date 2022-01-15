#####################################################
#            Parameters for this project            #
#####################################################

#Parameters
spacy_language = 'en_core_sci_md'
dataset = 'example'
note_type = 'example'
bucket_name = 'bucket'
blob = 'notes_proto.parquet'
labeling_model = 'LM'
BatchID = 'BATCHID'
threads = 'THREADS'
restart_batch = 4
batchid = 0

#Paths
path_termino = '../../res/'
path_output = '../output/'+dataset+'/'+note_type+'/'


#Datasets
base_termino = path_termino + 'clever_base_terminology.txt'
meddraID_CUI_table = path_termino + 'termMedDRA.txt'
llt_to_pt = path_termino + 'llt_to_pt.txt'
meddra_hier = path_termino + 'med_hier.txt'
termino_matcher = path_termino + 'meddra_termino.txt'

#NLP
spacy_model = '../../res/matchers/matcher_en_core_sci_md/'
matcher = '../../res/matchers/matcher_en_core_sci_md_test/meddra_v23_matcher.p'


#Outputs
extracted_hier = path_output + 'extracted_notes_BATCHID.parquet'
snorkel_preds = path_output + 'labeling_models/'+labeling_model+'_preds_dataset'+dataset+'BATCHID.p'
snorkel_probs = path_output + 'labeling_models/'+labeling_model+'_probs_dataset'+dataset+'BATCHID.p'
labeled_dataset = path_output + '/WL_BATCHID.parquet'
labeled_clean = path_output + 'WL_BATCHID_clean.parquet'

