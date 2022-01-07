import pandas as pd
import numpy as np
from multiprocessing import Pool
import HP
import re

def tokenize(sent):
    """Tokenize the sentence
    Args: 
        sent: a sentence
    return:
        tokens: a list of tokens
    """

    tokenizer = re.compile('\w+|\*\*|[^\s\w]')
    tokens = tokenizer.findall(sent.lower())
    cleaned_tokens = []
    for tok in tokens:
        cleaned_tokens.append(tok)

    return cleaned_tokens

#def remove_duplicate_extractions(df):
#    
#    df['extracted_tokens'] = df['concept_text'].apply(lambda x: tokenize(x))
#    df['token_length'] = df['extracted_tokens'].apply(lambda x: len(x))
#    
#    new_df = df.copy()
#    for index, row in df.iterrows():
#        if row['token_length'] > 1:
#            #print('multi-token concept at index ', index)
#            if index+1 in df.index:
#                #print(df['pos'][index] + 1)
#                #print(df['pos'][index+1])
#                if df['pos'][index] + 1 == df['pos'][index+1]:
#                    #print("double occurrence")
#                    #print(new_df.shape)
#                    new_df = new_df.drop(index+1)
#                    #print(new_df.shape)
#            else:
#                print('No double occurrence')
#    return new_df

def remove_duplicate_forward(df):
    
    df['extracted_tokens'] = df['concept_text'].apply(lambda x: tokenize(x))
    df['token_length'] = df['extracted_tokens'].apply(lambda x: len(x))
    
    new_df = df.copy()
    for index, row in df.iterrows():
        if row['token_length'] > 1:
            if index+1 in df.index:
                if df['pos'][index] + 1 == df['pos'][index+1]:
                    new_df = new_df.drop(index+1)
            else:
                print('No double occurrence')
    return new_df

def remove_duplicate_backward(df):
    
    df['extracted_tokens'] = df['concept_text'].apply(lambda x: tokenize(x))
    df['token_length'] = df['extracted_tokens'].apply(lambda x: len(x))
    
    new_df = df.copy()
    for index, row in df.iterrows():
        if row['token_length'] > 1:
            if index-1 in df.index:
                if df['pos'][index] == df['pos'][index-1]:
                    new_df = new_df.drop(index-1)
            else:
                print('No double occurrence')
    return new_df

def remove_duplicates(df):
    df_clean = parallelize_dataframe(df, remove_duplicate_forward, n_cores=int(HP.threads))
    df_clean2 = parallelize_dataframe(df_clean, remove_duplicate_backward, n_cores=int(HP.threads))
    return df_clean2

def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df



if __name__ == '__main__':
    
    print('Loading batch %s' % HP.BatchID)
    df = pd.read_parquet(HP.labeled_dataset)
    print(df.shape)
    df = df.reset_index(drop=True)
    print("Removing duplicate extractions...")
    #df_clean = parallelize_dataframe(df, remove_duplicate_extractions, n_cores=int(HP.threads))
    df_clean = remove_duplicates(df)
    print(df_clean.shape)
    
    print("Done! Saving...")
    df_clean.to_parquet(HP.labeled_clean)
    