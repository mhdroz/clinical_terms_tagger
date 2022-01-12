import pandas as pd
import pickle
import numpy as np
from maps import *
import HP
import spacy
from spacy.matcher import PhraseMatcher

flatten = lambda l: [item for sublist in l for item in sublist]

def create_matcher(spacy_language, termino, path_model, path_matcher):
    """
    Function that creates and saves a SpaCy PhraseMatcher from a given terminology
    
    Args:
        spacy_language: spacy language or nlp pipeline to load
        termino: source terminology for the matcher creation
        path_model: path to save the SpaCy language used for the matcher
        path_matcher: path to save the matcher
    """

    nlp = spacy.load(spacy_language)
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

    term = pd.read_csv(termino, sep='|')

    medid_list = term.code.to_list()
    term_list = term.term_text.to_list()
    meddra_dict = dict(zip(medid_list, term_list))
    len(meddra_dict)

    for item in meddra_dict.items():
        matcher.add(str(item[0]), None, nlp(str(item[1])))

    print(len(meddra_dict))

    nlp.to_disk(path_model)
    f = open(path_matcher, 'wb')
    pickle.dump(matcher, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Matcher saved!")
    f.close()


def get_meddra_concepts_df(original_notes, nlp, matcher, window=10): #TODO: Update the function for a flexible metadata for the dataframe
    """
    Function that performs clinical concepts extraction based on medDRA terminology using a SpaCy language model and phrase matcher

    Args:
        original_notes: clinical notes to process in pandas dataframe format. The column containing the note_text MUST be the LAST column of the input dataframe.
        nlp: SpaCy language model
        matcher: SpaCy PhraseMatcher built with nlp model
        window: window of tokens for left and right contexts. Default: 10

    Returns:
        concepts_df: Pandas dataframe with extracted clinical term, their position in the text, as well as left- and right-context
    """
    
    concepts_df = pd.DataFrame()
    print('original_notes shape: ', original_notes.shape)
    for i,  row in original_notes.iterrows():
        doc = nlp(row[-1])

        matches = matcher(doc)
        medID = []
        start_idx = []
        end_idx = []
        term = []
        loc_concept = []
        pos = []

        tokens = [token.text.lower() for token in doc]
        left_tokens = []
        right_tokens = []
    
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            medID.append(string_id)
            start_idx.append(start)
            end_idx.append(end)
            term.append(span.text)
            loc = (start, end)
            loc_concept.append(loc)

        for idx in loc_concept:
            start = idx[0]+1
            end = idx[1]
            left_tokens.append(tokens[0:start][-1 -window : -1])
            right_tokens.append(tokens[end:-1][0 : 1+window])
            pos.append(start)
        
        index_list = range(0, len(matches))    
        new_df = pd.DataFrame(np.nan, index=index_list, columns=row.index[0:-1])
    
        for j in range(0, len(row.index)-1):
            new_df[row.index[j]] = row[j]
       
        new_df['medID'] = medID
        new_df['concept_text'] = term
        new_df['pos'] = pos
        new_df['context_right'] = right_tokens
        new_df['context_left'] = left_tokens
    
        concepts_df = concepts_df.append(new_df)
        concepts_df = concepts_df.reset_index(drop=True)

    return concepts_df


def map_meddra_hier(df, meddraID_CUI_table, llt_to_pt, meddra_hier, termino_matcher):

    print('mapping medID to ttype:')
    df['ttype'] = df['medID'].apply(lambda x: lookup_ttype(x, termino_matcher))
    df['extracted_CUI'] = df['medID'].apply(lambda x: lookup_cui(x, meddraID_CUI_table))

    print('Done! Climbing the hierarchy:')
    df_llt = df.loc[df['ttype'] == 'LLT']
    df_hlt = df.loc[df['ttype'] == 'HT']
    df_hlgt = df.loc[df['ttype'] == 'HG']
    df_soc = df.loc[df['ttype'] == 'OS']

    print('Mapping LLT to PT')
    df_llt['PT'] = df_llt['medID'].apply(lambda x: match_llt_pt(x, llt_to_pt))
    df_llt['PT_text'] = df_llt['PT'].apply(lambda x: get_pt_text(x, meddra_hier))
    df_llt['PT_CUI'] = df_llt['PT'].apply(lambda x: lookup_cui(x, meddraID_CUI_table))

    print('mapping PT to HLT')
    df_llt['HLT'] = df_llt['PT'].apply(lambda x: match_pt_hlt(x, meddra_hier))
    df_hlt['HLT'] = df_hlt['medID']
    df_llt['HLT_text'], df_hlt['HLT_text'] = (df_all['HLT'].apply(lambda x: get_hlt_text(x, meddra_hier)) for df_all in [df_llt, df_hlt])
    df_llt['HLT_CUI'], df_hlt['HLT_CUI'] = (df_all['HLT'].apply(lambda x: lookup_cui(x, meddraID_CUI_table)) for df_all in [df_llt, df_hlt])

    print('mapping HLT to HLGT')
    df_llt['HLGT'], df_hlt['HLGT'] = (df_all['HLT'].apply(lambda x: match_hlt_hlgt(x, meddra_hier)) for df_all in [df_llt, df_hlt])
    df_hlgt['HLGT'] = df_hlgt['medID']
    df_llt['HLGT_text'], df_hlt['HLGT_text'], df_hlgt['HLGT_text'] = (df_all['HLGT'].apply(lambda x: get_hlgt_text(x, meddra_hier)) for df_all in [df_llt, df_hlt, df_hlgt])
    df_llt['HLGT_CUI'], df_hlt['HLGT_CUI'], df_hlgt['HLGT_CUI'] = (df_all['HLGT'].apply(lambda x: lookup_cui(x, meddraID_CUI_table)) for df_all in [df_llt, df_hlt, df_hlgt])

    print('mapping HLGT to SOC')
    df_llt['SOC'], df_hlt['SOC'], df_hlgt['SOC'] = (df_all['HLGT'].apply(lambda x: match_hlgt_soc(x, meddra_hier)) for df_all in [df_llt, df_hlt, df_hlgt])
    df_soc['SOC'] = df_soc['medID']
    df_llt['SOC_text'], df_hlt['SOC_text'], df_hlgt['SOC_text'], df_soc['SOC_text'] = (df_all['SOC'].apply(lambda x: get_soc_text(x, meddra_hier)) for df_all in [df_llt, df_hlt, df_hlgt, df_soc])
    df_llt['SOC_CUI'], df_hlt['SOC_CUI'], df_hlgt['SOC_CUI'], df_soc['SOC_CUI'] = (df_all['SOC'].apply(lambda x: lookup_cui(x, meddraID_CUI_table)) for df_all in [df_llt, df_hlt, df_hlgt, df_soc])

    df_hlt['PT'] = np.nan
    df_hlt['PT_text'] = np.nan
    df_hlt['PT_CUI'] = np.nan

    df_hlgt['PT'] = np.nan
    df_hlgt['PT_text'] = np.nan
    df_hlgt['PT_CUI'] = np.nan
    df_hlgt['HLT'] = np.nan
    df_hlgt['HLT_text'] = np.nan
    df_hlgt['HLT_CUI'] = np.nan

    df_soc['PT'] = np.nan
    df_soc['PT_text'] = np.nan
    df_soc['PT_CUI'] = np.nan
    df_soc['HLT'] = np.nan
    df_soc['HLT_text'] = np.nan
    df_soc['HLT_CUI'] = np.nan
    df_soc['HLGT'] = np.nan
    df_soc['HLGT_text'] = np.nan
    df_soc['HLGT_CUI'] = np.nan

    print('Merging all ttypes')
    df_hier = df_llt.append(df_hlt)
    df_hier = df_hier.append(df_hlgt)
    df_hier = df_hier.append(df_soc)

    return df_hier

def normalize_dtype(df):

    df['PT_text'] = df['PT_text'].astype(str)
    df['HLT_text'] = df['HLT_text'].astype(str)
    df['HLGT_text'] = df['HLGT_text'].astype(str)
    df['SOC_text'] = df['SOC_text'].astype(str)

    df = df.fillna(0)

    df['PT'] = df['PT'].astype(int)
    df['HLT'] = df['HLT'].astype(int)
    df['HLGT'] = df['HLGT'].astype(int)
    df['SOC'] = df['SOC'].astype(int)
    
    df['PT_CUI'] = df['PT_CUI'].astype(str)
    df['HLT_CUI'] = df['HLT_CUI'].astype(str)
    df['HLGT_CUI'] = df['HLGT_CUI'].astype(str)
    df['SOC_CUI'] = df['SOC_CUI'].astype(str)
    df['extracted_CUI'] = df['extracted_CUI'].astype(str)

    return df
