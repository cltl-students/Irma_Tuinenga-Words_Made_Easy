from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from transformers import pipeline
import bert_score
import string
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import spacy
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import OrderedDict


# initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")


def get_str_for_file_name(model):
    """
    Convert explicit model names to names used for saving in files.
    
    :param model: the explicit name of the model.
    :return: the condensed name of the model for saving in files.
    """
    if model == 'roberta-base':
        model_name_str = 'robertabase'
    elif model == 'roberta-large':
        model_name_str = 'robertalarge'
    elif model == 'google/electra-large-generator':
        model_name_str = 'electralarge'
    elif model == 'google/electra-base-generator':
        model_name_str = 'electrabase'
    elif model == 'bert-base-uncased':
        model_name_str = 'bertbase'
    elif model == 'bert-large-uncased':
        model_name_str = 'bertlarge'
    else:
        raise ValueError

    return model_name_str


def get_data_and_create_empty_df():
    """
    Read data from a file and create an empty dataframe to store substitutes in.
    
    :return: - data (pandas dataframe): the data from the loaded file, containing the sentences and the complex words.
             - substitutes_df (pandas dataframe): an empty dataframe with columns for sentence, complex word, and 10 substitutes
    """
    filename = "./data/test/tsar2022_en_test_none_no_noise.tsv"
    data = pd.read_csv(filename, sep='\t', header=None, names=["sentence", "complex_word"])

    # create an empty dataframe to store the substitutes for evaluation
    substitutes_df = pd.DataFrame(columns=["sentence", "complex_word"] + [f"substitute_{i + 1}" for i in range(10)])

    return data, substitutes_df


def instantiate_spacy_tokenizer_model_pipeline(model): # "roberta-base", "google/electra-large-generator", ...
    """
    Instantiate a SpaCy tokenizer, language model tokenizer, language model, and a fill-mask pipeline.
    
    :param model (str): the explicit name of the model.
    :return: - nlp: a SpaCy language model.
             - lm_tokenizer: a language model tokenizer based on pre-trained transformers.
             - lm_model: a pretrained masked language model.
             - fill_mask: a pipeline for filling masked instances in the masked language model.

    """
    nlp = spacy.load("en_core_web_sm")
    lm_tokenizer = AutoTokenizer.from_pretrained(model)
    lm_model = AutoModelForMaskedLM.from_pretrained(model)
    fill_mask = pipeline("fill-mask", lm_model, tokenizer=lm_tokenizer)

    return nlp, lm_tokenizer, lm_model, fill_mask


def substitute_generation_including_noise_removal_excluding_original_unmasked(data, substitutes_df, lm_tokenizer,
                                                                              fill_mask, model_name_str):
    """
    Generate 30 substitutes, remove noise, and store the top 10 results in a dataframe.
    
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instanced in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        # in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentence_masked_word, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        # remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        # and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

        punctuation_set = set(string.punctuation) - set(
            '-')  # retained hyphens in case tokenizers don't split on hyphenated compounds
        punctuation_set.update({'“',
                                '”'})  # as these curly quotes appeared in the Electra (SG step) results but were not
        # part of the string set

        try:
            substitutes = [substitute["token_str"].lower().strip() for substitute in result if not any(
                char in punctuation_set for char in
                substitute["token_str"])  # added .strip as roberta uses a leading space before each substitute
                           and not substitute["token_str"].startswith('##') and substitute["token_str"].strip() != ""]
            # print(f" Substitute list without unwanted punctuation characters: {substitutes}\n")
        except TypeError as error:
            continue

        # print(f"Substitute Generation (SG) step b): substitute list without empty elements and unwanted characters: "f"{substitutes}\n")

        # limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = substitutes[:10]
        # print(
            # f"Substitute Generation (SG) final step c): top-10 substitutes for the complex word '{complex_word}': "
            # f"{top_10_substitutes}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/test/SG_{model_name_str}_maskedsentenceonly.tsv", sep="\t", index=False,
                          header=False)
    print(
        f"SG_{model_name_str}_maskedsentenceonly exported to csv in path "
        f"'./predictions/test/SG_{model_name_str}_maskedsentenceonly.tsv'\n")

    return substitutes_df


def substitute_generation_including_noise_removal_including_original_unmasked(data, substitutes_df, lm_tokenizer,
                                                                              fill_mask, model_name_str):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise, and store the top 10 results in a dataframe.
    
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataframe to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        # in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        # generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        # remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        # and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

        punctuation_set = set(string.punctuation) - set(
            '-')  # retained hyphens in case tokenizers don't split on hyphenated compounds
        punctuation_set.update({'“',
                                '”'})  # as these curly quotes appeared in the Electra (SG step) results but were not
        # part of the string set

        try:
            substitutes = [substitute["token_str"].lower().strip() for substitute in result if not any(
                char in punctuation_set for char in
                substitute["token_str"])  # added .strip as roberta uses a leading space before each substitute
                           and not substitute["token_str"].startswith('##') and substitute["token_str"].strip() != ""]
            # print(f" Substitute list without unwanted punctuation characters: {substitutes}\n")
        except TypeError as error:
            continue

        # print(
            # f"Substitute Generation (SG) step b): substitute list without empty elements and unwanted characters: "
            # f"{substitutes}\n")

        # limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = substitutes[:10]
        # print(
        #     f"Substitute Generation (SG) final step c): top-10 substitutes for the complex word '{complex_word}': "
        #     f"{top_10_substitutes}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/test/SG_incl_orig_sentence_{model_name_str}.tsv", sep="\t", index=False, header=False)
    print(f"SG_incl_orig_sentence_{model_name_str} exported to csv in path './predictions/test/SG_incl_orig_sentence_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_1(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
     Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected
     forms, and antonyms of complex word, and store the top 10 results in a dataframe.
    
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        # in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        # generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        # remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        # and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

        punctuation_set = set(string.punctuation) - set(
            '-')  # retained hyphens in case tokenizers don't split on hyphenated compounds
        punctuation_set.update({'“',
                                '”'})  # as these curly quotes appeared in the Electra (SG step) results but were not
        # part of the string set

        try:
            substitutes = [substitute["token_str"].lower().strip() for substitute in result if not any(
                char in punctuation_set for char in
                substitute["token_str"])  # added .strip as roberta uses a leading space before each substitute
                           and not substitute["token_str"].startswith('##') and substitute["token_str"].strip() != ""]
            # print(f"Substitute list without unwanted punctuation characters: {substitutes}\n")
        except TypeError as error:
            continue

        # print(f"Substitute Generation (SG) final step b): substitute list without empty elements and unwanted
        # characters: {substitutes}\n")

        # 2. Substitute Selection (SS) phase 1: remove duplicates of substitutes; as well as duplicates, inflected
        # forms, and antonyms of complex word:

        # a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # such as Roberta that did not lowercase by default)
        # the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        substitutes_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
            else:
                substitutes_dupl.append(sub)
        # print(f"Duplicate substitutes: {substitutes_dupl}\n")

        # print(
            # f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes: "
            # f"{substitutes_no_dupl}\n")

        # b) remove duplicates and inflected forms of the complex word from the substitute list
        # Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        # remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        substitutes_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
            else:
                substitutes_dupl_complex_word.append(substitute)
        # print(f"duplicates and inflected forms of complex word removed: {substitutes_dupl_complex_word} \n")
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        # c) remove antonyms of the complex word from the substitute list
        # get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        # remove antonyms of the complex word from the substitute list
        substitutes_no_antonyms = []
        for substitute in substitutes_no_dupl_complex_word:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma not in antonyms_complex_word:
                substitutes_no_antonyms.append(substitute)
            # else:
            #     print(f"Removed antonym: {substitute}")
        # print(f"Substitute Selection (SS) phase 1, step c): substitute list without antonyms of the complex word
        # '{complex_word}': {substitutes_no_antonyms}\n")

        # limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = substitutes_no_antonyms[:10]
        # print(
            # f"Substitute Selection (SS) phase 1, final step d): top-10 substitutes for the complex word "
            # f"'{complex_word}': {top_10_substitutes}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/test/SS_phase1_{model_name_str}.tsv", sep="\t", index=False, header=False)
    print(f"SS_phase1_{model_name_str} exported to csv in path './predictions/test/SS_phase1_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_1(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected
     forms, and antonyms of complex word. Sort the substitutes first that are synonyms with the complex word, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        # in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        # generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        # remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        # and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

        punctuation_set = set(string.punctuation) - set(
            '-')  # retained hyphens in case tokenizers don't split on hyphenated compounds
        punctuation_set.update({'“',
                                '”'})  # as these curly quotes appeared in the Electra (SG step) results but were not
        # part of the string set

        try:
            substitutes = [substitute["token_str"].lower().strip() for substitute in result if not any(
                char in punctuation_set for char in
                substitute["token_str"])  # added .strip as roberta uses a leading space before each substitute
                           and not substitute["token_str"].startswith('##') and substitute["token_str"].strip() != ""]
            # print(f"Substitute list without unwanted punctuation characters: {substitutes}\n")
        except TypeError as error:
            continue

        # print(f"Substitute Generation (SG) final step b): substitute list without empty elements and unwanted
        # characters: {substitutes}\n")

        # 2. Substitute Selection (SS) phase 1: remove duplicates, inflected forms, and antonyms of complex word:

        # a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        # the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        # b) remove duplicates and inflected forms of the complex word from the substitute list
        # Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        # remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        # c) remove antonyms of the complex word from the substitute list
        # get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        # remove antonyms of the complex word from the substitute list
        substitutes_no_antonyms = []
        for substitute in substitutes_no_dupl_complex_word:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma not in antonyms_complex_word:
                substitutes_no_antonyms.append(substitute)
            # else:
                # print(f"Removed antonym: {substitute}")
        # print(f"Substitute Selection (SS) phase 1, step c): substitute list without antonyms of the complex word
        # '{complex_word}': {substitutes_no_antonyms}\n")

        # 3. Substitute Selection (SS) phase 2, option 1: sort the substitutes that are synonyms with the complex word
        # first:
        synonyms = []
        non_synonyms = []

        for substitute in substitutes_no_antonyms:
            substitute_synsets = wn.synsets(substitute)

            # get all the synsets for the complex word
            complex_word_synsets = wn.synsets(complex_word_lemma)

            # check if the substitute and the complex word share the same synset
            if any(substitute_synset == complex_word_synset for substitute_synset in substitute_synsets for
                   complex_word_synset in complex_word_synsets):
                # add substitute to synonyms list
                synonyms.append(substitute)
            else:
                # add substitute to non_synonyms list
                non_synonyms.append(substitute)

        # combine the lists with synonyms appearing first
        final_list = synonyms + non_synonyms
        # print(
            # f"Substitute Selection (SS) phase 2, option 1, step a): substitutes sorted on synonyms for the complex word"
            # f" '{complex_word}' first: {final_list}\n")

        # limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = final_list[:10]
        # print(
            # f"Substitute Selection (SS) phase 2, option 1, final step b): top 10 substitutes, sorted on synonyms for "
            # f"the complex word '{complex_word}' first: {top_10_substitutes}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/test/SS_phase2_option1_SharedSyns_{model_name_str}.tsv", sep="\t", index=False,
                          header=False)
    print(
        f"SS_phase2_option1_SharedSyns_{model_name_str} exported to csv in path "
        f"'./predictions/test/SS_phase2_option1_SharedSyns_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_2a(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected
     forms, and antonyms of complex word. Sort the substitutes first that have the same one-level up hypernyms as the complex word, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        # in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        # generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        # remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        # and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

        punctuation_set = set(string.punctuation) - set(
            '-')  # retained hyphens in case tokenizers don't split on hyphenated compounds
        punctuation_set.update({'“',
                                '”'})  # as these curly quotes appeared in the Electra (SG step) results but were not
        # part of the string set

        try:
            substitutes = [substitute["token_str"].lower().strip() for substitute in result if not any(
                char in punctuation_set for char in
                substitute["token_str"])  # added .strip as roberta uses a leading space before each substitute
                           and not substitute["token_str"].startswith('##') and substitute["token_str"].strip() != ""]
            # print(f"Substitute list without unwanted punctuation characters: {substitutes}\n")
        except TypeError as error:
            continue

        # print(f"Substitute Generation (SG) final step b): substitute list without empty elements and unwanted
        # characters: {substitutes}\n")

        # 2. Substitute Selection (SS) phase 1: remove duplicates, inflected forms, and antonyms of complex word:

        # a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        # the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        # b) remove duplicates and inflected forms of the complex word from the substitute list
        # Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        # remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        # c) remove antonyms of the complex word from the substitute list
        # get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        # remove antonyms of the complex word from the substitute list
        substitutes_no_antonyms = []
        for substitute in substitutes_no_dupl_complex_word:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma not in antonyms_complex_word:
                substitutes_no_antonyms.append(substitute)
            # else:
            #     print(f"Removed antonym: {substitute}")
        # print(f"Substitute Selection (SS) phase 1, step c): substitute list without antonyms of the complex word
        # '{complex_word}': {substitutes_no_antonyms}\n")

        # 3. Substitute Selection (SS) phase 2, option 2-a: sort the substitutes that share their direct (1 level up)
        # hypernyms with the complex word first:

        # step a): get first and second level hypernyms of the complex word
        complex_word_synsets = wn.synsets(complex_word_lemma)
        complex_word_hypernyms_1 = [h for syn in complex_word_synsets for h in syn.hypernyms()]
        complex_word_hypernyms_2 = [h2 for h1 in complex_word_hypernyms_1 for h2 in h1.hypernyms()]
        complex_word_hypernyms_1_lemmas = [lemma for h in complex_word_hypernyms_1 for lemma in h.lemma_names()]
        complex_word_hypernyms_2_lemmas = [lemma for h in complex_word_hypernyms_2 for lemma in h.lemma_names()]
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-a, step a): complex_word_hypernyms_lemmas (1st level "
            # f"hypernyms) for complex word '{complex_word}': {complex_word_hypernyms_1_lemmas}\n")
        # print(f"Substitute Selection (SS) phase 2, option 2-b, step a): complex_word_hypernyms_lemmas (2nd level
        # hypernyms) for complex word '{complex_word}': {complex_word_hypernyms_2_lemmas}\n")

        # step b): get the first level hypernyms of the substitutes and check for shared hypernyms
        intersection_1_substitutes = []
        other_1_substitutes = []

        for substitute in substitutes_no_antonyms:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            substitute_synsets = wn.synsets(substitute_lemma)

            substitute_hypernyms_1 = [h for syn in substitute_synsets for h in syn.hypernyms()]
            substitute_hypernyms_2 = [h2 for h1 in substitute_hypernyms_1 for h2 in h1.hypernyms()]
            substitute_hypernyms_1_lemmas = [lemma for h in substitute_hypernyms_1 for lemma in h.lemma_names()]
            substitute_hypernyms_2_lemmas = [lemma for h in substitute_hypernyms_2 for lemma in h.lemma_names()]

            intersection_1 = set(complex_word_hypernyms_1_lemmas).intersection(set(substitute_hypernyms_1_lemmas))

            if intersection_1:
                # print(
                    # f"Substitute {substitute} has the same one-level hypernym as the complex word {complex_word} in "
                    # f"Wordnet. Matching hypernym: {intersection_1}\n")
                intersection_1_substitutes.append(substitute)
            else:
                other_1_substitutes.append(substitute)
                
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-a, step b): list of substitutes that share the same "
            # f"one-level hypernym with the complex word '{complex_word}' in Wordnet: {intersection_1_substitutes}\n")
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-a, step b): list of substitutes that DO NOT share the same "
            # f"one-level hypernym with the complex word '{complex_word}' in Wordnet: {other_1_substitutes}\n")

        # step c): create the final list, by putting the intersection first
        final_list = intersection_1_substitutes + other_1_substitutes
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-a, step c): substitutes sorted on shared one-level hypernyms "
            # f"with the complex word '{complex_word}' in Wordnet first:  {final_list}\n")

        # step d): limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = final_list[:10]
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-a, final step d): top 10 of substitutes, sorted on shared "
            # f"one-level up hypernyms with the complex word '{complex_word}' in Wordnet first: {top_10_substitutes}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/test/SS_phase2_option2a_SharedHyper1_{model_name_str}.tsv", sep="\t",
                          index=False, header=False)
    print(
        f"SS_phase2_option2a_SharedHyper1_{model_name_str} exported to csv in path "
        f"'./predictions/test/SS_phase2_option2a_SharedHyper1_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_2b(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
     Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected
     forms, and antonyms of complex word. Sort the substitutes first that have the same two-level up hypernyms as the complex word, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        # in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        # generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        # remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        # and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

        punctuation_set = set(string.punctuation) - set(
            '-')  # retained hyphens in case tokenizers don't split on hyphenated compounds
        punctuation_set.update({'“',
                                '”'})  # as these curly quotes appeared in the Electra (SG step) results but were not
        # part of the string set

        try:
            substitutes = [substitute["token_str"].lower().strip() for substitute in result if not any(
                char in punctuation_set for char in
                substitute["token_str"])  # added .strip as roberta uses a leading space before each substitute
                           and not substitute["token_str"].startswith('##') and substitute["token_str"].strip() != ""]
            # print(f"Substitute list without unwanted punctuation characters: {substitutes}\n")
        except TypeError as error:
            continue

        # print(f"Substitute Generation (SG) final step b): substitute list without empty elements and unwanted
        # characters: {substitutes}\n")

        # 2. Substitute Selection (SS) phase 1: remove duplicates, inflected forms, and antonyms of complex word:

        # a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        # the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        # b) remove duplicates and inflected forms of the complex word from the substitute list
        # Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        # remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        # c) remove antonyms of the complex word from the substitute list
        # get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        # remove antonyms of the complex word from the substitute list
        substitutes_no_antonyms = []
        for substitute in substitutes_no_dupl_complex_word:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma not in antonyms_complex_word:
                substitutes_no_antonyms.append(substitute)
            # else:
            #     print(f"Removed antonym: {substitute}")
        # print(f"Substitute Selection (SS) phase 1, step c): substitute list without antonyms of the complex word
        # '{complex_word}': {substitutes_no_antonyms}\n")

        # 3. Substitute Selection (SS) phase 2, option 2-b: sort the substitutes that share their indirect (2 levels up)
        # hypernyms with the complex word first:

        # step a): get first and second level hypernyms of the complex word
        complex_word_synsets = wn.synsets(complex_word_lemma)
        complex_word_hypernyms_1 = [h for syn in complex_word_synsets for h in syn.hypernyms()]
        complex_word_hypernyms_2 = [h2 for h1 in complex_word_hypernyms_1 for h2 in h1.hypernyms()]
        complex_word_hypernyms_1_lemmas = [lemma for h in complex_word_hypernyms_1 for lemma in h.lemma_names()]
        complex_word_hypernyms_2_lemmas = [lemma for h in complex_word_hypernyms_2 for lemma in h.lemma_names()]
        # print(f"Substitute Selection (SS) phase 2, option 2-a, step a): complex_word_hypernyms_lemmas (1st level
        # hypernyms) for complex word '{complex_word}': {complex_word_hypernyms_1_lemmas}\n")
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-b, step a): complex_word_hypernyms_lemmas (2nd level "
            # f"hypernyms) for complex word '{complex_word}': {complex_word_hypernyms_2_lemmas}\n")

        # step b): get second level hypernyms of the substitutes and check for shared hypernyms
        intersection_2_substitutes = []
        other_2_substitutes = []

        for substitute in substitutes_no_antonyms:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            substitute_synsets = wn.synsets(substitute_lemma)

            substitute_hypernyms_1 = [h for syn in substitute_synsets for h in syn.hypernyms()]
            substitute_hypernyms_2 = [h2 for h1 in substitute_hypernyms_1 for h2 in h1.hypernyms()]
            substitute_hypernyms_1_lemmas = [lemma for h in substitute_hypernyms_1 for lemma in h.lemma_names()]
            substitute_hypernyms_2_lemmas = [lemma for h in substitute_hypernyms_2 for lemma in h.lemma_names()]

            intersection_2 = set(complex_word_hypernyms_2_lemmas).intersection(set(substitute_hypernyms_2_lemmas))

            if intersection_2:
                # print(
                    # f"Substitute {substitute} has the same two-level hypernym as the complex word {complex_word} in "
                    # f"Wordnet. Matching hypernym: {intersection_2}\n")
                intersection_2_substitutes.append(substitute)
            else:
                other_2_substitutes.append(substitute)

        # print(
            # f"Substitute Selection (SS) phase 2, option 2-b, step b): list of substitutes that share the same two-level"
            # f" hypernym with the complex word '{complex_word}' in Wordnet: {intersection_2_substitutes}\n")
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-b, step b): list of substitutes that DO NOT share the same "
            # f"two-level hypernym with the complex word '{complex_word}' in Wordnet: {other_2_substitutes}\n")

        # step c): create the final list, by putting the intersection first
        final_list = intersection_2_substitutes + other_2_substitutes
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-b, step c): substitutes sorted on shared two-level hypernyms "
            # f"with the complex word '{complex_word}' in Wordnet first:  {final_list}\n")

        # step d): limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = final_list[:10]
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-b, final step d): top 10 of substitutes, sorted on shared "
            # f"two-level hypernyms with the complex word '{complex_word}' in Wordnet first: {top_10_substitutes}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/test/SS_phase2_option2b_SharedHyper2_{model_name_str}.tsv", sep="\t",
                          index=False, header=False)
    print(
        f"SS_phase2_option2b_SharedHyper2_{model_name_str} exported to csv in path "
        f"'./predictions/test/SS_phase2_option2b_SharedHyper2_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_2c(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected forms, and antonyms of complex word. Sort the substitutes first that have the same one- or two-level up hypernyms as the complex word, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        # in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        # generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        # remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        # and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

        punctuation_set = set(string.punctuation) - set(
            '-')  # retained hyphens in case tokenizers don't split on hyphenated compounds
        punctuation_set.update({'“',
                                '”'})  # as these curly quotes appeared in the Electra (SG step) results but were not
        # part of the string set

        try:
            substitutes = [substitute["token_str"].lower().strip() for substitute in result if not any(
                char in punctuation_set for char in
                substitute["token_str"])  # added .strip as roberta uses a leading space before each substitute
                           and not substitute["token_str"].startswith('##') and substitute["token_str"].strip() != ""]
            # print(f"Substitute list without unwanted punctuation characters: {substitutes}\n")
        except TypeError as error:
            continue

        # print(f"Substitute Generation (SG) final step b): substitute list without empty elements and unwanted
        # characters: {substitutes}\n")

        # 2. Substitute Selection (SS) phase 1: remove duplicates, inflected forms, and antonyms of complex word:

        # a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        # the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        # b) remove duplicates and inflected forms of the complex word from the substitute list
        # Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        # remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        # c) remove antonyms of the complex word from the substitute list
        # get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        # remove antonyms of the complex word from the substitute list
        substitutes_no_antonyms = []
        for substitute in substitutes_no_dupl_complex_word:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma not in antonyms_complex_word:
                substitutes_no_antonyms.append(substitute)
            # else:
            #     print(f"Removed antonym: {substitute}")
        # print(f"Substitute Selection (SS) phase 1, step c): substitute list without antonyms of the complex word
        # '{complex_word}': {substitutes_no_antonyms}\n")

        # 3. Substitute Selection (SS) phase 2, option 2-c: sort the substitutes that share their direct (1 level up) or
        # indirect (2 levels up) hypernyms with the complex word first:

        # step a): get first and second level hypernyms of the complex word
        complex_word_synsets = wn.synsets(complex_word_lemma)
        complex_word_hypernyms_1 = [h for syn in complex_word_synsets for h in syn.hypernyms()]
        complex_word_hypernyms_2 = [h2 for h1 in complex_word_hypernyms_1 for h2 in h1.hypernyms()]
        complex_word_hypernyms_1_lemmas = [lemma for h in complex_word_hypernyms_1 for lemma in h.lemma_names()]
        complex_word_hypernyms_2_lemmas = [lemma for h in complex_word_hypernyms_2 for lemma in h.lemma_names()]
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-c, step a): complex_word_hypernyms_lemmas (1st level "
            # f"hypernyms) for complex word '{complex_word}': {complex_word_hypernyms_1_lemmas}\n")
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-c, step a): complex_word_hypernyms_lemmas (2nd level "
            # f"hypernyms) for complex word '{complex_word}': {complex_word_hypernyms_2_lemmas}\n")

        # step b): get first and second level hypernyms of the substitutes and check for shared hypernyms
        intersection_1_2_substitutes = []
        other_1_2_substitutes = []

        for substitute in substitutes_no_antonyms:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            substitute_synsets = wn.synsets(substitute_lemma)

            substitute_hypernyms_1 = [h for syn in substitute_synsets for h in syn.hypernyms()]
            substitute_hypernyms_2 = [h2 for h1 in substitute_hypernyms_1 for h2 in h1.hypernyms()]
            substitute_hypernyms_1_lemmas = [lemma for h in substitute_hypernyms_1 for lemma in h.lemma_names()]
            substitute_hypernyms_2_lemmas = [lemma for h in substitute_hypernyms_2 for lemma in h.lemma_names()]

            intersection_1_2 = set(complex_word_hypernyms_1_lemmas + complex_word_hypernyms_2_lemmas).intersection(
                set(substitute_hypernyms_1_lemmas + substitute_hypernyms_2_lemmas))

            if intersection_1_2:
                # print(
                #     f"Substitute {substitute} has the same one-level or two-level hypernym as the complex word "
                #     f"{complex_word} in Wordnet. Matching hypernym: {intersection_1_2}\n")
                intersection_1_2_substitutes.append(substitute)
            else:
                other_1_2_substitutes.append(substitute)
        # print(
            # f" intersection one and two level hypernyms complex word with one and two level hypernyms of substitute:  "
            # f"{intersection_1_2}\n")

        # print(
            # f"Substitute Selection (SS) phase 2, option 2-c, step b): list of substitutes that share the same one- or "
            # f"two-level hypernym with the complex word '{complex_word}' in Wordnet: {intersection_1_2_substitutes}\n")
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-c, step b): list of substitutes that DO NOT share the same "
            # f"one- or two-level hypernym with the complex word '{complex_word}' in Wordnet: {other_1_2_substitutes}\n")

        # step c): create the final list, by putting the intersection first
        final_list = intersection_1_2_substitutes + other_1_2_substitutes
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-c, step c): substitutes sorted on shared one- or two-level "
            # f"hypernyms with the complex word '{complex_word}' in Wordnet first:  {final_list}\n")

        # step d): limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = final_list[:10]
        # print(
        #     f"Substitute Selection (SS) phase 2, option 2-c, final step d): top 10 of substitutes, sorted on shared "
        #     f"one- or two-level hypernyms with the complex word '{complex_word}' in Wordnet first: "
            # f"{top_10_substitutes}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/test/SS_phase2_option2c_SharedHyper1_2_{model_name_str}.tsv", sep="\t",
                          index=False, header=False)
    print(
        f"SS_phase2_option2c_SharedHyper1_2_{model_name_str} exported to csv in path "
        f"'./predictions/test/SS_phase2_option2c_SharedHyper1_2_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_3(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp,
                                          score_model, letter=''):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected forms, and antonyms of complex word. Sort the substitutes by their BERTScores, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param score_model: a pre-trained model to calculate BERTScores, used for scoring and ranking the substitutes.
    :param letter: allowing differentiation by option 3 type ('a','b','c','d','e','f')

    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        # in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        # generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (step: initial substitute list: {substitutes}\n")

        # 2: Morphological Generation and Context Adaptation (Morphological Adaptation):
        # a) remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        # and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

        punctuation_set = set(string.punctuation) - set(
            '-')  # retained hyphens in case tokenizers don't split on hyphenated compounds
        punctuation_set.update({'“',
                                '”'})  # as these curly quotes appeared in the Electra (SG step) results but were not
        # part of the string set

        try:
            substitutes = [substitute["token_str"].lower().strip() for substitute in result if not any(
                char in punctuation_set for char in
                substitute["token_str"])  # added .strip as roberta uses a leading space before each substitute
                           and not substitute["token_str"].startswith('##') and substitute["token_str"].strip() != ""]
            # print(f"Morphological Adaptation step a): substitute list without unwanted punctuation characters:
            # {substitutes}\n")
        except TypeError as error:
            continue

        # b) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        # the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Morphological Adaptation step b): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        # c) remove duplicates and inflected forms of the complex word from the substitute list

        # first Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to
        # see if their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        # remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Morphological Adaptation step c): substitute list without duplicates of the complex word nor inflected
        # forms of the complex word: {substitutes_no_dupl_complex_word}\n")

        # d) remove antonyms of the complex word from the substitute list
        # step 1: get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        # step 2: remove antonyms of the complex word from the substitute list
        substitutes_no_antonyms = []
        for substitute in substitutes_no_dupl_complex_word:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma not in antonyms_complex_word:
                substitutes_no_antonyms.append(substitute)
            # else:
            #     print(f"Removed antonym: {substitute}")
        # print(f"Morphological Adaptation step d): substitute list without antonyms of the complex word:
        # {substitutes_no_antonyms}\n")

        # 3: Substitute Selection (SS) phase 2, option 3, by calculating Bert scores:

        # create sentence with the complex word replaced by the substitutes
        sentence_with_substitutes = [sentence.replace(complex_word, sub) for sub in substitutes_no_antonyms]
        # print(f"List with sentences where complex word is substituted: {sentence_with_substitutes}\n")

        # calculate BERTScores, and rank the substitutes based on these scores
        score_model_name_str = get_str_for_file_name(score_model)
        if len(sentence_with_substitutes) > 0:  # to make sure the substitute list is always filled
            logging.getLogger('transformers').setLevel(
                logging.ERROR)  # to prevent the same warnings from being printed x times
            scores = bert_score.score([sentence] * len(sentence_with_substitutes), sentence_with_substitutes, lang="en",
                                      model_type=score_model, verbose=False)
            logging.getLogger('transformers').setLevel(
                logging.WARNING)  # to reset the logging level back to printing warnings

            # create a list of tuples, each tuple containing a substitute and its score
            substitute_score_pairs = list(zip(substitutes_no_antonyms, scores[0].tolist()))

            # sort the list of tuples by the scores (the second element of each tuple), in descending order
            sorted_substitute_score_pairs = sorted(substitute_score_pairs, key=lambda x: x[1], reverse=True)

            # print each substitute with its score
            # for substitute, score in sorted_substitute_score_pairs:
            #     print(f"Substitute: {substitute}, BertScore: {score}")

            # extract the list of substitutes from the sorted pairs
            bertscore_ranked_substitutes_only = [substitute for substitute, _ in sorted_substitute_score_pairs]
            # print(f"substitutes based on bertscores in context: {bertscore_ranked_substitutes_only}\n")

            # limit the substitutes to the 10 first ones for evaluation
            bertscore_top_10_substitutes = bertscore_ranked_substitutes_only[:10]
            # print(f"top-10 substitutes based on bertscores in context: {bertscore_top_10_substitutes}\n")

        else:
            bertscore_top_10_substitutes = []

        # add the sentence, complex_word, and substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + bertscore_top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/test/SS_phase2_option3{letter}_BS{score_model_name_str}_{model_name_str}.tsv",
                          sep="\t", index=False, header=False)
    print(
        f"SS_phase2_option3{letter}_BS{score_model_name_str}_{model_name_str} exported to csv in path "
        f"'./predictions/test/SS_phase2_option3{letter}_BS{score_model_name_str}_{model_name_str}.tsv'\n")

    return substitutes_df, score_model_name_str




def get_abbrev(input):
    """
    Based on a specific file path, return the corresponding abbreviation. 

    :param input: the specific file path.
    :return: the corresponding abbreviation.
    """
    if input == './cefrj/cefrj_all_treebank.tsv':
        abbrev = '2aCEFR_J_robertabase'
    elif input == './cefr_ls/uchida_pos.tsv':
        abbrev = '2bCEFR_ls_robertabase'
    elif input == './cefr_efllex/EFLLex_mostfreq.tsv':
        abbrev = '2cCEFR_efl_mostfreq_robertabase'
    elif input == './cefr_efllex/EFLLex_weighted.tsv':
        abbrev = '2dCEFR_efl_weighted_robertabase'
    elif input == './cefr_all/cefr_all_combined.tsv':
        abbrev = '2eCEFR_all_robertabase'
    elif input == './predictions/test/SS_phase2_option2b_SharedHyper2_robertabase.tsv':
        abbrev = 'SS_no1'
    elif input == './predictions/test/SS_phase2_option1_SharedSyns_robertabase.tsv':
        abbrev = 'SS_no2'
    elif input == './predictions/test/SS_phase2_option3f_BSrobertalarge_robertabase.tsv':
        abbrev = 'SS_no3'

    return abbrev


    
def substitute_ranking_option1_hyper2(prediction):
    """
    Rank the substitutes of a complex word based on whether they function as a two-level up hypernym of the complex word.
    
    param prediction: the file path to the tsv file containing the sentences, complex words, and their substitutes.
    :return: None. The results are saved to a tsv file.
    """
    predict_abbrev = get_abbrev(prediction)
    pred_df = pd.read_csv(prediction, sep='\t', header=None)

    for index, row in pred_df.iterrows():
        sentence = row[0]
        complex_word = row[1]
        substitutes = row[2:12]
        # print(f"complex word: {complex_word}\n")
        # print(f"substitutes: {substitutes}\n")

        # step a): get the complex word lemma, the complex word synsets, and its first level hypernym
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        complex_word_synsets = wn.synsets(complex_word_lemma)
        complex_word_hypernyms_1 = [h for syn in complex_word_synsets for h in syn.hypernyms()]
        complex_word_hypernyms_1_lemmas = [lemma for h in complex_word_hypernyms_1 for lemma in h.lemma_names()]
        complex_word_hypernyms_2 = [h2 for h1 in complex_word_hypernyms_1 for h2 in h1.hypernyms()]
        complex_word_hypernyms_2_lemmas = [lemma for h in complex_word_hypernyms_2 for lemma in h.lemma_names()]
        # print(f"Substitute Ranking (SR), option 1-b, step a): complex_word_hypernyms_lemmas (2nd level hypernyms) for complex word '{complex_word}': {complex_word_hypernyms_2_lemmas}\n")

        # step b): get the lemma and synsets of the substitutes, and store the original substitutes with the lemmas and synsets
        substitute_lemmas_synsets = []
        for substitute in substitutes:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            substitute_synsets = wn.synsets(substitute_lemma)
            substitute_lemmas_synsets.append((substitute, substitute_lemma, substitute_synsets))
        # print(f"Substitute Ranking (SR), option 1-b, step b): substitute lemmas synsets: {substitute_lemmas_synsets}\n")

        # step c): get the intersection of the substitute synsets with the 2nd level hypernyms of the complex word
        intersection_2_substitutes = []
        other_substitutes = []

        for substitute, substitute_lemma, substitute_synsets in substitute_lemmas_synsets:
            # get the lemmas of the substitute synsets
            substitute_synsets_lemmas = [lemma for syn in substitute_synsets for lemma in syn.lemma_names()]

            # check if the substitute belongs to a synset that is the same as the 1st level hypernym of the complex word
            intersection_2 = set(complex_word_hypernyms_2_lemmas).intersection(set(substitute_synsets_lemmas))
            if intersection_2:
                intersection_2_substitutes.append(substitute)  # append original substitute
            else:
                other_substitutes.append(substitute)  # append original substitute

        # print(f"Substitute Ranking (SR) option 1b, step c): list of substitutes of which their synsets are the same as the second level hypernyms of the complex word '{complex_word}' in Wordnet: {intersection_2_substitutes}\n")
        # print(f"Substitute Ranking (SR) option 1b, step c): list of substitutes of which their synsets are NOT the same as the second-level hypernym of the complex word '{complex_word}' in Wordnet: {other_substitutes}\n")

        # step d): create the final list, by putting the intersection first in the list, appending the list with the other substitutes
        final_list = intersection_2_substitutes + other_substitutes
    #     print(f"Substitute Ranking (SR) option 1b, step d): substitutes sorted on whether they belong to a synset that is the same as the second level hypernym of the complex word '{complex_word}' in Wordnet first:  {final_list}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        pred_df.loc[index] = [sentence, complex_word] +  final_list

    # export the dataframe to tsv for evaluation
    pred_df.to_csv(f"./predictions/test/{predict_abbrev}_SR_option1b_Hyper2Hypo_robertabase.tsv", sep="\t", index=False, header=False)
    print(f"{predict_abbrev}_SR_option1b_Hyper2Hypo_robertabase exported to csv in path './predictions/test/{predict_abbrev}_SR_option1b_Hyper2Hypo_robertabase.tsv'\n")



def map_pos_spacy_wordnet(pos_spacy):
    """
    map spaCy PoS tags to WordNet PoS tags.
    
    :param pos_spacy: the Spacy PoS tag to be converted.
    :return: the corresponding Wordnet PoS tag, or wn.NOUN if no match is found. 
    """
    pos_map = {
        'NOUN': wn.NOUN,
        'VERB': wn.VERB,
        'ADJ': wn.ADJ,
        'ADV': wn.ADV
    }
    return pos_map.get(pos_spacy, wn.NOUN) 



def substitute_ranking_option2_cefr(cefr_dataset, prediction):
    """
    Rank the substitutes of a complex word based on their respective CEFR levels.
    
    :param cefr_dataset: the file path to the tsv file containing the CEFR levels of words.
    :param prediction:  the file path to the tsv file containing sentences, complex words, and their substitutes
    :return:  None. The results are saved to a tsv file.
    """
    dataset_abbrev = get_abbrev(cefr_dataset)
    predict_abbrev = get_abbrev(prediction)

    # read the CEFR levels file into a dataframe
    cefr_df = pd.read_csv(cefr_dataset, sep='\t', header=None, names=['word', 'pos', 'cefr'])

    # define a mapping from CEFR levels to numerical values
    cefr_level_mapping = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}

    # map the CEFR levels in the dataframe to numerical values using the mapping
    cefr_df['cefr'] = cefr_df['cefr'].map(cefr_level_mapping)

    # read the predictions file into a dataframe
    pred_df = pd.read_csv(prediction, sep='\t', header=None)

    # for each row in the predictions dataframe, map each substitute to its CEFR level, sort them, and save them into a new list
    predictions_cefr = []
    for index, row in pred_df.iterrows():
        sentence = row[0]
        complex_word = row[1]
        substitutes = row[2:12]
        # print(f"complex_word: {complex_word}\n")
        # print(f"substitutes: {substitutes}\n")

        # replace the complex word in the sentence with the substitute, and parse it to get the pos tag of the substitute
        substitute_pos = []
        for substitute in substitutes:
            replaced_sentence = sentence.replace(complex_word, substitute)
            doc = nlp(replaced_sentence)
            pos = [token.pos_ for token in doc if token.text == substitute][0]
            substitute_pos.append((substitute, pos))

        # get the lemma of the substitute 
        substitutes_lemmas = []
        for sub_pos in substitute_pos:
            substitute, pos_spacy = sub_pos
            pos_substitute_wordnet = map_pos_spacy_wordnet(pos_spacy)
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_ if substitute in [token.text for token in
                                                                          doc_substitute] else substitute
            substitutes_lemmas.append((substitute, substitute_lemma))
        # print(f"Substitutes with their lemmas: {substitutes_lemmas}\n")

        # map each lemmatized substitute to its CEFR level, or to a high number if it doesn't have a CEFR level
        substitutes_cefr = []
        for original, lemmatized in substitutes_lemmas:
            # get the pos of the original substitute by parsing the sentence where the complex word is replaced by the substitute
            sub_sentence = sentence.replace(complex_word, original)
            sub_pos = dict(pos_tag(word_tokenize(sub_sentence))).get(original)
            # if the lemmatized substitute equals a word that is found in cefrj_all_treebank.tsv AND the POS tag of that word (in cefrj_all_treebank.tsv) is the same as the POS tag of the substitute:
            if lemmatized in cefr_df['word'].values and cefr_df[cefr_df['word'] == lemmatized]['pos'].values[0] == sub_pos:
                substitutes_cefr.append((original, cefr_df[cefr_df['word'] == lemmatized]['cefr'].values[0]))
            else:
                substitutes_cefr.append(
                    (original, 7))  # assign a high value if it doesn't have a CEFR level or if pos don't match
        # print(f"substitutes_cefr: {substitutes_cefr}\n")

        # sort the substitutes based on their CEFR levels
        ranked_cefr_subs = sorted(substitutes_cefr, key=lambda x: x[1])
        # print(f"Substitute Ranking (SR), option 2: substitutes with cefr level ranked first: {ranked_cefr_subs}\n")

        # append the sorted list of substitutes to the new lists, keeping original form
        predictions_cefr.append([sentence, complex_word] + [sub for sub, _ in ranked_cefr_subs])

    # create a new dataframe from the new lists and export it to tsv for evaluation
    new_df = pd.DataFrame(predictions_cefr)
    new_df.to_csv(f'./predictions/test/{predict_abbrev}_SR_option{dataset_abbrev}.tsv', sep='\t', index=False, header=False)
    print(
        f"{predict_abbrev}_SR_option{dataset_abbrev} exported to csv in path './predictions/test/{predict_abbrev}_SR_option{dataset_abbrev}.tsv'\n")
    
  


    
def substitute_ranking_option2_cefr_weighted(weighted_dataset, prediction):
    """
    Rank the substitutes of a complex word based on their respective weighted CEFR levels.
    
    :param str cefr_dataset: the file path to the tsv file containing the CEFR levels of words.
    :param str prediction: the file path to the tsv file containing sentences, complex words, and their substitutes
    :return:  None. The results are saved to a tsv file.
    """
    
    dataset_abbrev = get_abbrev(weighted_dataset)
    predict_abbrev = get_abbrev(prediction)
    
    cefr_df = pd.read_csv(weighted_dataset, sep='\t', header=None, names=['word', 'pos', 'Weighted CEFR'])
    pred_df = pd.read_csv(prediction, sep='\t', header=None)

    # for each row in the predictions dataframe, map each substitute to its weighted CEFR level, sort them, and save them into a new list
    predictions_cefr = []
    for index, row in pred_df.iterrows():
        sentence = row[0]
        complex_word = row[1]
        substitutes = row[2:12]

         # replace the complex word in the sentence with the substitute, and parse it to get the pos tag of the substitute
        substitute_pos = []
        for substitute in substitutes:
            replaced_sentence = sentence.replace(complex_word, substitute)
            doc = nlp(replaced_sentence)
            pos = [token.pos_ for token in doc if token.text == substitute][0]
            substitute_pos.append((substitute, pos))

        # get the lemma of the substitute 
        substitutes_lemmas = []
        for sub_pos in substitute_pos:
            substitute, pos_spacy = sub_pos
            pos_substitute_wordnet = map_pos_spacy_wordnet(pos_spacy)
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_ if substitute in [token.text for token in doc_substitute] else substitute
            substitutes_lemmas.append((substitute, substitute_lemma))
        # print(f"Substitutes with their lemmas: {substitutes_lemmas}\n")

        # map each lemmatized substitute to its weighted CEFR level, or to a high number if it doesn't have a CEFR level
        substitutes_cefr = []
        for original, lemmatized in substitutes_lemmas:
            # get the pos of the original substitute by parsing the sentence where the complex word is replaced by the substitute
            sub_sentence = sentence.replace(complex_word, original)
            sub_pos = dict(pos_tag(word_tokenize(sub_sentence))).get(original)
            # if the lemmatized substitute equals a word that is found in EFLLex_weighted.tsv AND the POS tag of that word (in EFLLex_weighted.tsv) is the same as the POS tag of the substitute:
            if lemmatized in cefr_df['word'].values and cefr_df[cefr_df['word'] == lemmatized]['pos'].values[0] == sub_pos:
                substitutes_cefr.append((original, cefr_df[cefr_df['word'] == lemmatized]['Weighted CEFR'].values[0]))
            else:
                substitutes_cefr.append((original, 7))  # assign a high value if it doesn't have a CEFR level or if pos don't match

        # sort the substitutes based on their weighted CEFR levels
        ranked_cefr_subs = sorted(substitutes_cefr, key=lambda x: x[1])
        # print (f"substitutes_cefr ranked on weighted average: {ranked_cefr_subs}\n")

        # append the sorted list of substitutes to the new lists, keeping original form
        predictions_cefr.append([sentence, complex_word] + [sub for sub, _ in ranked_cefr_subs])

    # create a new dataframe from the new lists and export it to tsv for evaluation
    new_df = pd.DataFrame(predictions_cefr)
    new_df.to_csv(f'./predictions/test/{predict_abbrev}_SR_option{dataset_abbrev}.tsv', sep='\t', index=False, header=False)
    print(f"{predict_abbrev}_SR_option{dataset_abbrev} exported to csv in path './predictions/test/{predict_abbrev}_SR_option{dataset_abbrev}'\n")
    

    
def posteval_substitute_ranking_hyper(prediction, levels=[1, 2]):
    """
    Rank the substitutes of a complex word based on whether they function as a hypernym of the complex word up to a specific hypernym level.
    
    :param prediction: the file path to the tsv file containing the sentences, complex words, and their substitutes.
    :param levels: the specific hypernym levels as a list of integers. Default is [1, 2].
    :return: None. The results are saved to a tsv file.
    """
    pred_df = pd.read_csv(prediction, sep='\t', header=None)

    for index, row in pred_df.iterrows():
        sentence = row[0]
        complex_word = row[1]
        substitutes = row[2:12]
        # print(f"complex word: {complex_word}\n")
        # print(f"substitutes: {substitutes}\n")

        # step a): get the complex word lemma, the complex word synsets, and the hypernyms depending on the input for the 'levels' parameter
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        complex_word_synsets = wn.synsets(complex_word_lemma)

        complex_word_hypernyms = {i:[] for i in range(1, max(levels)+1)}
        complex_word_hypernyms[1] = [h for syn in complex_word_synsets for h in syn.hypernyms()]
        for i in range(2, max(levels)+1):
            complex_word_hypernyms[i] = [h for h_prev in complex_word_hypernyms[i - 1] for h in h_prev.hypernyms()]

        complex_word_hypernyms_lemmas = [lemma for level in levels for h in complex_word_hypernyms[level] for lemma in h.lemma_names()]

        # step b): get the lemma and synsets of the substitutes, and store the original substitutes with the lemmas and synsets
        substitute_lemmas_synsets = []
        for substitute in substitutes:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            substitute_synsets = wn.synsets(substitute_lemma)
            substitute_lemmas_synsets.append((substitute, substitute_lemma, substitute_synsets))

        # step c): get the intersection of the substitute synsets with the input for the 'levels' parameter for the complex word
        intersection_substitutes = []
        other_substitutes = []
        for substitute, substitute_lemma, substitute_synsets in substitute_lemmas_synsets:
            substitute_synsets_lemmas = [lemma for syn in substitute_synsets for lemma in syn.lemma_names()]

            intersection = set(complex_word_hypernyms_lemmas).intersection(set(substitute_synsets_lemmas))
            if intersection:
                intersection_substitutes.append(substitute)
            else:
                other_substitutes.append(substitute)
                
        # step d): create the final list, by putting the intersection first in the list, appending the list with the other substitutes
        final_list = intersection_substitutes + other_substitutes
        pred_df.loc[index] = [sentence, complex_word] + final_list

    pred_df.to_csv(f"./predictions/test/posteval_Hyper{'_'.join(map(str, levels))}-Hypo.tsv", sep="\t", index=False, header=False)
    print(f"posteval_Hyper{'_'.join(map(str, levels))}-Hypo exported to path './predictions/test/posteval_Hyper{'_'.join(map(str, levels))}-Hypo.tsv'\n")

    
    
def posteval_analysis_cefr_weighted_stats_complexword_substitutes(weighted_dataset, prediction):
    """
    Perform statistical measures of variation for complex word and substitute datasets.
    
    :param weighted_dataset: the file path to the tsv file containing the CEFR levels of words.
    :param prediction: the file path to the tsv file containing sentences, complex words, and their substitutes
    :return:  None. 
    """
    
    dataset_abbrev = get_abbrev(weighted_dataset)
    predict_abbrev = get_abbrev(prediction)

    cefr_df = pd.read_csv(weighted_dataset, sep='\t', header=None, names=['word', 'pos', 'Weighted CEFR'])
    pred_df = pd.read_csv(prediction, sep='\t', header=None)    
    

    # for each row in the predictions dataframe, map each substitute to its weighted CEFR level, sort them, and save them into a new list
    predictions_cefr = []
    count_7 = 0  # counter of substitutes with level 7
    total_rows = 0  # counter of total number of rows
    cefr_levels = []  # list to store CEFR levels of all substitutes (except those with level 7)

    complex_word_cefr_levels = []  # list to store CEFR levels of the complex words
    unavailable_complex_words = [] #list to store complex words with unavailable CEFR levels

    for index, row in pred_df.iterrows():
        sentence = row[0]
        complex_word = row[1]
        substitutes = row[2:12]

        # get the POS of the complex word and its lemma
        complex_word_pos = dict(pos_tag(word_tokenize(sentence))).get(complex_word)
        complex_word_lemma = nlp(complex_word)[0].lemma_ if complex_word in [token.text for token in nlp(complex_word)] else complex_word

        # get the CEFR level of the complex word and append it to complex_word_cefr_levels
        if complex_word_lemma in cefr_df['word'].values and cefr_df[cefr_df['word'] == complex_word_lemma]['pos'].values[0] == complex_word_pos:
            complex_word_cefr = cefr_df[cefr_df['word'] == complex_word_lemma]['Weighted CEFR'].values[0]
            complex_word_cefr_levels.append(complex_word_cefr)
        else:
            unavailable_complex_words.append(complex_word)
        
        # replace the complex word in the sentence with the substitute, and parse it to get the pos tag of the substitute
        substitute_pos = []
        for substitute in substitutes:
            replaced_sentence = sentence.replace(complex_word, substitute)
            doc = nlp(replaced_sentence)
            pos = [token.pos_ for token in doc if token.text == substitute][0]
            substitute_pos.append((substitute, pos))
            
        # get the lemma of the substitute 
        substitutes_lemmas = []
        for sub_pos in substitute_pos:
            substitute, pos_spacy = sub_pos
            pos_substitute_wordnet = map_pos_spacy_wordnet(pos_spacy)
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_ if substitute in [token.text for token in doc_substitute] else substitute
            substitutes_lemmas.append((substitute, substitute_lemma))
            
        #get the substitute cefr levels
        substitutes_cefr = []
        for original, lemmatized in substitutes_lemmas:
            sub_sentence = sentence.replace(complex_word, original)
            sub_pos = dict(pos_tag(word_tokenize(sub_sentence))).get(original)
            if lemmatized in cefr_df['word'].values and cefr_df[cefr_df['word'] == lemmatized]['pos'].values[0] == sub_pos:
                substitutes_cefr.append((original, cefr_df[cefr_df['word'] == lemmatized]['Weighted CEFR'].values[0]))
            else:
                substitutes_cefr.append((original, 7))  
                count_7 += 1  # add 1 to the counter per substitute with level 7

        total_rows += 1  # add 1 to the total number of rows

        ranked_cefr_subs = sorted(substitutes_cefr, key=lambda x: x[1])
        # append the sorted list of substitutes to the new lists, keeping original form
        predictions_cefr.append([sentence, complex_word] + [sub for sub, _ in ranked_cefr_subs])

        # get the CEFR levels of the substitutes, except those with level 7
        cefr_levels.extend(cefr for sub, cefr in ranked_cefr_subs if cefr < 7)

    # calculate the average number of substitutes with level 7
    average_7 = count_7 / total_rows
    print(f"On average, there were {average_7} substitutes of the provided 10 substitutes that had not been found in the combined CEFR dataset.")
    print()

    unavailable_complex_word_percentage = (1 - len(complex_word_cefr_levels) / len(pred_df)) * 100
    unavailable_complex_word_count = len(pred_df) - len(complex_word_cefr_levels)

    print(f"Count and Percentage of rows with unavailable CEFR level for the complex word: {unavailable_complex_word_count} and {unavailable_complex_word_percentage}%")
    # print(f"Complex words with unavailable CEFR levels: {unavailable_complex_words}")
    print()

    # calculate average CEFR level of the substitutes (except those with level 7)
    if cefr_levels:
        average_cefr_level = sum(cefr_levels) / len(cefr_levels)
        print(f"Average CEFR level for substitutes (except level 7): {average_cefr_level}")
        median_cefr_level = np.median(cefr_levels)
        print(f"Median CEFR level for substitutes (except level 7): {median_cefr_level}")
        std_dev_cefr_level = np.std(cefr_levels)
        print(f"Standard Deviation of CEFR level for substitutes (except level 7): {std_dev_cefr_level}")
        kurtosis_cefr_level = stats.kurtosis(cefr_levels)
        print(f"Kurtosis of CEFR level for substitutes (except level 7): {kurtosis_cefr_level}")

        # Q-Q plot for substitutes
        plt.figure()
        stats.probplot(cefr_levels, dist="norm", plot=plt)
        plt.title("Q-Q Plot for Substitutes")
        plt.show()

    # calculate average CEFR level of the complex words
    if complex_word_cefr_levels:
        average_complex_word_cefr = sum(complex_word_cefr_levels) / len(complex_word_cefr_levels)
        print(f"Average CEFR level for complex words: {average_complex_word_cefr}")
        median_complex_word_cefr = np.median(complex_word_cefr_levels)
        print(f"Median CEFR level for complex words: {median_complex_word_cefr}")
        std_dev_complex_word_cefr = np.std(complex_word_cefr_levels)
        print(f"Standard Deviation of CEFR level for complex words: {std_dev_complex_word_cefr}")
        kurtosis_complex_word_cefr = stats.kurtosis(complex_word_cefr_levels)
        print(f"Kurtosis of CEFR level for complex words: {kurtosis_complex_word_cefr}")

        # Q-Q plot for complex words
        plt.figure()
        stats.probplot(complex_word_cefr_levels, dist="norm", plot=plt)
        plt.title("Q-Q Plot for Complex Words")
        plt.show()
  
    
    
def posteval_analysis_cefr_weighted_stats_goldlabels(weighted_dataset, gold):
    """
    Perform statistical measures of variation for gold label dataset.
    
    :param weighted_dataset: the file path to the tsv file containing the CEFR levels of words.
    :param gold: the file path to the tsv file containing sentences, complex words, and the gold labels.
    :return:  None. 
    """

    cefr_df = pd.read_csv('./cefr_all/cefr_all_combined.tsv', sep='\t', header=None, names=['word', 'pos', 'Weighted CEFR'])
    gold_df = pd.read_csv('./data/test/tsar2022_en_test_gold_no_noise.tsv', sep='\t', header=None)

    gold_cefr_levels = []

    first_as_min_count = 0
    first_as_max_count = 0
    first_equals_last_count = 0
    counter_sorted = 0
    rows_0_cefr = 0
    rows_1_cefr = 0
    rows_2plus_cefr = 0
    first_as_lowest_count = 0

    for index, row in gold_df.iterrows():  
        sentence = row[0]
        complex_word = row[1]
        substitutes = row[2:].values

        #get the gold substitute pos tags
        substitute_pos = []
        for substitute in substitutes:
            # skip if substitute is not a string (e.g., NaN)
            if not isinstance(substitute, str):
                continue

            replaced_sentence = sentence.replace(complex_word, substitute)
            doc = nlp(replaced_sentence)
            pos = [token.pos_ for token in doc][0] 
            substitute_pos.append((substitute, pos))

        #get the gold substitute lemmas
        substitutes_lemmas = []
        for sub_pos in substitute_pos:
            substitute, pos_spacy = sub_pos
            pos_substitute_wordnet = map_pos_spacy_wordnet(pos_spacy)
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_ if substitute in [token.text for token in doc_substitute] else substitute
            substitutes_lemmas.append((substitute, substitute_lemma))

        # get the gold substitute cefr levels
        substitutes_cefr = []
        for original, lemmatized in substitutes_lemmas:
            sub_sentence = sentence.replace(complex_word, original)
            sub_pos = dict(pos_tag(word_tokenize(sub_sentence))).get(original)
            if lemmatized in cefr_df['word'].values and cefr_df[cefr_df['word'] == lemmatized]['pos'].values[0] == sub_pos:
                substitutes_cefr.append((original, cefr_df[cefr_df['word'] == lemmatized]['Weighted CEFR'].values[0]))
            else:
                substitutes_cefr.append((original, 7))  

        # remove duplicates but preserve order with OrderedDict
        substitutes_cefr = list(OrderedDict(substitutes_cefr).items())

        # get the CEFR levels of the gold substitutes, except those with level 7
        substitutes_cefr_levels = [cefr for sub, cefr in substitutes_cefr if cefr < 7]
        # print(f"substitutes_cefr_levels : {substitutes_cefr_levels} \n")
        gold_cefr_levels.extend(substitutes_cefr_levels)

        # update the CEFR level counters
        num_cefr = len(substitutes_cefr_levels)
        if num_cefr == 0:
            rows_0_cefr += 1
        elif num_cefr == 1:
            rows_1_cefr += 1
        else:
            rows_2plus_cefr += 1

        # check if first CEFR level is lower than, higher than, or equal to the last
        if num_cefr > 1:
            if substitutes_cefr_levels[0] < substitutes_cefr_levels[-1]:
                first_as_min_count += 1
            elif substitutes_cefr_levels[0] > substitutes_cefr_levels[-1]:
                first_as_max_count += 1
            else:  # this means substitutes_cefr_levels[0] == substitutes_cefr_levels[-1]
                first_equals_last_count += 1

        # check if first CEFR level is the lowest of all in that row
        if num_cefr > 1 and min(substitutes_cefr_levels) == substitutes_cefr_levels[0]:
            first_as_lowest_count += 1

        # check if CEFR levels are in ascending order
        if substitutes_cefr_levels == sorted(substitutes_cefr_levels) and len(substitutes_cefr_levels) > 1:
            counter_sorted += 1
            # print(f"\nRow {index}: CEFR levels are sorted in ascending order")

    #calculate statistics for the gold substitutes (except those with level 7)
    if gold_cefr_levels:
        average_cefr_level = sum(gold_cefr_levels) / len(gold_cefr_levels)
        median_cefr_level = np.median(gold_cefr_levels)
        std_dev_cefr_level = np.std(gold_cefr_levels)
        kurtosis_cefr_level = stats.kurtosis(gold_cefr_levels)
        print(f"Average CEFR level for gold substitutes (except level 7): {average_cefr_level}")
        print(f"Median CEFR level for gold substitutes (except level 7): {median_cefr_level}")
        print(f"Standard Deviation of CEFR level for gold substitutes (except level 7): {std_dev_cefr_level}")
        print(f"Kurtosis of CEFR level for gold substitutes (except level 7): {kurtosis_cefr_level}")

        # Q-Q plot for gold substitutes
        plt.figure()
        stats.probplot(gold_cefr_levels, dist="norm", plot=plt)
        plt.title("Q-Q Plot for Gold substitutes")
        plt.show()

    print(f"Rows with 0 CEFR level: {rows_0_cefr}")
    print(f"Rows with 1 CEFR level: {rows_1_cefr}")
    print(f"Rows with more than 1 CEFR level: {rows_2plus_cefr}")
    print(f"Rows where the first CEFR level is lower than the last: {first_as_min_count} ({first_as_min_count/rows_2plus_cefr*100}%)")
    print(f"Rows where the first CEFR level is higher than the last: {first_as_max_count} ({first_as_max_count/rows_2plus_cefr*100}%)")
    print(f"Rows where the first CEFR level is equal to the last: {first_equals_last_count} ({first_equals_last_count/rows_2plus_cefr*100}%)")
    print(f"Rows where the first CEFR level is the lowest of all: {first_as_lowest_count} ({first_as_lowest_count/rows_2plus_cefr*100}%)")
    print(f"Rows where CEFR levels are in ascending order: {counter_sorted} ({counter_sorted/gold_df.shape[0]*100}%)")




