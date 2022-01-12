import pandas as pd
from multiprocessing import Pool
import spacy
from spacy.matcher import PhraseMatcher
import numpy as np
import argparse
import sys, os
from os.path import exists
import pickle
import time
import HP
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model.baselines import MajorityLabelVoter
from snorkel.labeling.model.label_model import LabelModel
from snorkel.utils import probs_to_preds
from utils import get_meddra_concepts_df, map_meddra_hier, normalize_dtype, create_matcher
from maps import *

#To run the tagger: nohup python meddra_extract.py --task 'tag' --label_model 'label_model' --threads 28 --datasource bucket > extract_meddra_NO_existing_matcher.out&

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_model', dest='label_model', help='Define which weak labeling model to use: majority_voter or label_model', default='majority_voter', type=str)
    parser.add_argument('--task', dest='task', help='Define which task: tag, label', default='tag', type=str)
    parser.add_argument('--threads', dest='threads', help='Select number of processes', default=1, type=int)
    parser.add_argument('--datasource', dest='datasource', help='Select data source: local or bucket', default='local', type=str)
    parser.add_argument('--restart', dest='restart', help='Restart processing. Only valid of using bucket datasource.', default=False, type=str2bool)

    if len(sys.argv) == 1:
        parser.print_help()
        print("Using Default Settings...")

    args = parser.parse_args()
    return args


def extract_concepts(df):
    """
    Clinical concepts extraction and padding of sequences for LSTM training for the labeling step

    Args:
        df: raw clinical notes to process. Pandas dataframe format
        nlp: SpaCy language model
        matcher: SpaCy PhraseMatcher built with nlp model
        termino: MedDRA-CUI mapping

    Returns:
        extracted_df: Pandas dataframe with extracted clinical concepts
        X_train: Padded sequences for LSTM labeling
    """

    print('Extracting all MedDRA concepts from the notes:')
    start = time.time()
    extracted_df = get_meddra_concepts_df(df, nlp, matcher, window=10)
    print("done extracting medDRA terms  in time {:.2f}".format(time.time()-start))

    print("Mapping MedDRA IDs up the hierarchy:")
    extracted_hier = map_meddra_hier(extracted_df, meddraID_CUI_table, llt_to_pt, meddra_hier, termino_matcher)
    extracted_hier = normalize_dtype(extracted_hier)

    return extracted_hier

def label_extractions(df_tagged, label_model):

    df_terms = pd.read_csv(HP.base_termino, sep='|', header=None)
    df_terms = df_terms.rename(columns={0:'_id', 1:'label', 2:'_class'})

    negex = df_terms.loc[df_terms['_class'] == 'NEGEX']
    negex = negex['label'].to_list()

    fam = df_terms.loc[df_terms['_class'] == "FAM"]
    fam = fam['label'].to_list()

    hx = df_terms.loc[df_terms['_class'] == 'HX']
    hx = hx['label'].to_list()

    #Define LFs
    ABSTAIN = -1
    PRESENT = 1
    ABSENT = 0

    negation = set(negex)
    prior_knowledge = {'know', 'known'}
    history = set(hx)
    present = {'present', 'consistent', 'persists'}
    family = set(fam)

    @labeling_function(resources=dict(negation=negation))
    def negated(x, negation):
        if len(negation.intersection(set(x.context_left))) > 0:
            return ABSENT
        else:
            return ABSTAIN

    @labeling_function(resources=dict(prior_knowledge=prior_knowledge))
    def known_issue(x, prior_knowledge):
        if len(prior_knowledge.intersection(set(x.context_left))) > 0 or len(prior_knowledge.intersection(set(x.context_right))) > 0:
            return PRESENT
        else:
            return ABSTAIN

    @labeling_function(resources=dict(history=history))
    def clinical_history(x, history):
        if len(history.intersection(set(x.context_left))) > 0:
            return ABSENT
        else:
            return ABSTAIN

    @labeling_function(resources=dict(present=present))
    def present_issue(x, present):
        if len(present.intersection(set(x.context_left))) > 0:
            return PRESENT
        else:
            return ABSTAIN

    @labeling_function(resources=dict(family=family))
    def experiencer_is_family(x, family):
        if len(family.intersection(set(x.context_left))) > 0 or len(family.intersection(set(x.context_right))) > 0:
            return ABSENT
        else:
            return ABSTAIN

    lfs = [negated, clinical_history, experiencer_is_family]

    start = time.time()
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df_tagged)
    print("done applying LFs  in time {:.2f}".format(time.time()-start))


    df_L = pd.DataFrame(L_train)
    df_L.columns = ['negated', 'clinical_history', 'experiencer_is_family']

    df_tagged['negated'] = df_L['negated']
    df_tagged['clinical_history'] = df_L['clinical_history']
    df_tagged['experiencer_is_family'] = df_L['experiencer_is_family']


    print("LF analysis summary:")
    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    if label_model == 'majority_voting':

        print("Computing majority voting label:")
        majority_model = MajorityLabelVoter()
        preds_train = majority_model.predict(L=L_train)

        df_tagged['MV'] = preds_train
        prob_train = None

        print("Done labeling. Saving labels to: "+HP.snorkel_preds)
        pickle.dump(preds_train, open(HP.snorkel_preds, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    if label_model == 'label_model':

        print("Computing label model:")
        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

        prob_train = label_model.predict_proba(L=L_train)

        preds_train = probs_to_preds(probs=prob_train)

        df_tagged['LM'] = preds_train


    return df_tagged, preds_train, prob_train

def get_labels(df):

    df = df.assign(polarity=np.nan)
    df.loc[df['negated'] == 0, 'polarity'] = 'NEGATIVE'
    df.loc[df['negated'] == -1, 'polarity'] = 'POSITIVE'

    df = df.assign(temporality=np.nan)
    df.loc[df['clinical_history'] == 0, 'temporality'] = 'HISTORY'
    df.loc[df['clinical_history'] == -1, 'temporality'] = 'PRESENT'

    df = df.assign(experiencer=np.nan)
    df.loc[df['experiencer_is_family'] == 0, 'experiencer'] = 'FAMILY'
    df.loc[df['experiencer_is_family'] == -1, 'experiencer'] = 'PATIENT'

    df = df.drop(['negated', 'clinical_history', 'experiencer_is_family'], axis=1)

    return df

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'y', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'n', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    args = parse_args()

    task = args.task
    label_model = args.label_model
    threads = args.threads
    datasource = args.datasource
    restart = args.restart

    print("Check for matcher:")
    if exists(HP.matcher):
        print("Matcher exists")
    else:
        print("No matcher found, creating matcher")
        
        create_matcher(HP.spacy_language, HP.termino_matcher, HP.spacy_model, HP.matcher)

    

    
    if task == 'tag':    
        
        print("Multithreaded tagging")
        
        print("Loading SpaCy model and MedDRA matcher:")
        nlp = spacy.load(HP.spacy_model)
        matcher = pickle.load(open(HP.matcher, 'rb'))

        meddraID_CUI_table = pd.read_csv(HP.meddraID_CUI_table, sep='|')
        llt_to_pt = pd.read_csv(HP.llt_to_pt, sep='|')
        meddra_hier = pd.read_csv(HP.meddra_hier, sep='|')
        termino_matcher = pd.read_csv(HP.termino_matcher, sep='|')

        print('Loading dataset:')
        if datasource == 'local':
            print("Loading data from local disk")

            df = pd.read_parquet(HP.raw_notes)
            df = df.reset_index(drop=True)
            
            extracted_hier = parallelize_dataframe(df, extract_concepts, n_cores=threads)

            print("Saving extractions...")
            extracted_hier.to_parquet(HP.extracted_hier)
            
        if datasource == 'bucket':
            print("Loading data from gcp bucket %s" % HP.bucket_name)
            
            df = pd.read_parquet('gs://%s/%s' % (HP.bucket_name, HP.blob))
            
            #df['length'] = df['note_text'].apply(lambda x: len(x))
            #df = df.loc[df['length'] > 10]
            
            print(df.shape)
            
            batches = int(df.shape[0] / 20000)
            if batches == 0:
                interval = int(df.shape[0] / 1)
            else:
                interval = int(df.shape[0] / batches)
            
            if restart == False:
                print('start processing data from batch 0')
                for i in range(0, batches+1):
                #for i in range(0, 4):
                    if i % 10 == 0:
                        print('processing batch %03d' % (i))
                    select = df.iloc[i*interval:(i+1)*interval]

                    extracted_hier = parallelize_dataframe(select, extract_concepts, n_cores=threads)

                    print("Saving extractions...")
                    extracted_hier.to_parquet(HP.path_output + 'extracted_notes_%03d.parquet' % (i+HP.batchid))

            if restart == True:
                print('Restarting processing from batch %s' % HP.restart_batch)
                for i in range(HP.restart_batch, batches+1):
                    if i % 10 == 0:
                        print('processing batch %03d' % (i))
                    select = df.iloc[i*interval:(i+1)*interval]

                    extracted_hier = parallelize_dataframe(select, extract_concepts, n_cores=threads)

                    print("Saving extractions...")
                    extracted_hier.to_parquet(HP.path_output + 'extracted_notes_%03d.parquet' % (i+HP.batchid))


    if task == 'label':
        print("Selected task: Weak labeling only.")
        print("Loading tagged dataset:")

        extracted_hier = pd.read_parquet(HP.extracted_hier)

        df_tagged, preds_train, prob_train = label_extractions(extracted_hier, label_model)

        df_tagged = get_labels(df_tagged)

        df_tagged.to_parquet(HP.labeled_dataset)
        pickle.dump(preds_train, open(HP.snorkel_preds, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(prob_train, open(HP.snorkel_probs, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        print("All done!")