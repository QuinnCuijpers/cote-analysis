import pandas as pd
import numpy as np 
import spacy
import networkx as nx 
import matplotlib.pyplot as plt 

def ner(text_file):
    """
    use the spacy module to apply npl to the input txt file and return the text with named entities

    Args:
        text_file (DirEntry): a txt file on which to find named entities

    Returns:
        spacy.tokens.doc.Doc: a spacy object of the txt file where named entities are flagged
    """    
    # download from the command line with python3 -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    book_text = open(text_file, encoding='utf-8').read()
    book_ner = nlp(book_text)
    
    return book_ner

def create_entity_per_sentence_df(book_ner):
    """
    using the ner object inputted this creates a dataframe of sentences and ner per sentence

    Args:
        book_ner (spacy.tokens.doc.Doc): takes in an ner processed text

    Returns:
        dataframe: a dataframe that contains every sentence and the ner's per sentence of the input text
    """    
    sentence_entity_df = []

    for sentence in book_ner.sents:
        ent_list = [ent.text for ent in sentence.ents]
        sentence_entity_df.append({
            "sentence": sentence,
            "entities":ent_list
        })
    return pd.DataFrame(sentence_entity_df)

def filter_character_entities(ent_list: list, character_df: pd.DataFrame):
    """
    a function that filters the given ent_list to only include characters from the given character_df

    Args:
        ent_list: a list containing ner's that you want to be filtered
        character_df: a dataframe with the following columns: display name, character, firstname, lastname and character reverse

    Returns:
        list: a list that only contains the character names (as defined in the "display name" column) that match in the ent_list
    """    
    characters = []
    for ent in ent_list:
        ent = str(ent)
        # in japan these are common suffixes
        ent = ent.replace('-san', '')
        ent = ent.replace('-kun', '')
        ent = ent.replace('-chan', '')
        ent = ent.replace('-sensei', '')
        if ent in list(character_df['character']) or ent in list(character_df['firstname']) or ent in list(character_df['lastname']) or ent in list(character_df['character reverse']):
            idx =  character_df.index[character_df['character'] == ent].tolist()+ character_df.index[character_df['firstname'] == ent].tolist() + character_df.index[character_df['lastname'] == ent].tolist() + character_df.index[character_df['character reverse'] == ent].tolist()
            if idx:
                idx = idx[0]
            characters.append(character_df['display name'][idx])
    return characters

def create_relations(sentence_entity_df_filtered: pd.DataFrame, window_size:int=5):
    """
    using a dataframe with characters we define an edgelist using a moving window

    Args:
        sentence_entity_df_filtered (dataframe): a dataframe containing a character_entities column, which are arrays of characters
        window_size (int, optional): the size of the moving window. Defaults to 10.

    Returns:
        dataframe: returns a dataframe in the form of an edgelist for networkx
    """    
    relations = []

    for i in range(sentence_entity_df_filtered.index[-1]):
        end_i = min(i+window_size, sentence_entity_df_filtered.index[-1])
        char_list = (sum((sentence_entity_df_filtered.loc[i:end_i].character_entities), []))
        
        # remove duplicate values in the char_list
        char_uniques = [char_list[i] for i in range(len(char_list))
                        if (i==0) or char_list[i] != char_list[i-1]]
        
        if len(char_uniques) > 1:
            for idx, char in enumerate(char_uniques[:-1]):
                target = char_uniques[idx + 1]
                relations.append({
                    "source":char,
                    "target": target
                })
    relations_df = pd.DataFrame(relations)
    relations_df = pd.DataFrame(np.sort(relations_df.values, axis = 1), columns=relations_df.columns)
    
    # init a new value column to all 1's
    relations_df['value'] = 1
    # aggregate over all source, target pairs, making value become the count
    relations_df = relations_df.groupby(['source', 'target'], sort=False, as_index=False).sum()
    
    return relations_df