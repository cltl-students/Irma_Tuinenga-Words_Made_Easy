from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from transformers import pipeline
import bert_score
import string
from nltk.corpus import wordnet as wn
import spacy
import logging


def get_str_for_file_name(model):
    """
    Convert explicit model names to names used for saving in files.
    
    :param model (str): the explicit name of the model.
    :return (str): the condensed name of the model for saving in files.
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
    filename = "./data/trial/tsar2022_en_trial_none_no_noise.tsv"
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

        # print the sentence and the complex word
        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        ## in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        # ## concatenate the original sentence and the masked sentence
        # sentences_concat= f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        ## generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentence_masked_word, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        ## remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        ## and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

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

        # # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

        # print(
            # '----------------------------------------------------------------------------------------------------------'
            # '-----------------------------------')

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/trial/SG_{model_name_str}_maskedsentenceonly.tsv", sep="\t", index=False,
                          header=False)
    print(
        f"SG_{model_name_str}_maskedsentenceonly exported to csv in path "
        f"'./predictions/trial/SG_{model_name_str}_maskedsentenceonly.tsv'\n")

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

        # print the sentence and the complex word
        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        ## in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        ## concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        ## generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        ## remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        ## and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

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
        # # print(
        #     f"Substitute Generation (SG) final step c): top-10 substitutes for the complex word '{complex_word}': "
        #     f"{top_10_substitutes}\n")

        # # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

        # print(
            # '----------------------------------------------------------------------------------------------------------'
            # '-----------------------------------')

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/trial/SG_{model_name_str}.tsv", sep="\t", index=False, header=False)
    print(f"SG_{model_name_str} exported to csv in path './predictions/trial/SG_{model_name_str}.tsv'\n")

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

        # print the sentence and the complex word
        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        ## in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        ## concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        ## generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        ## remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        ## and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

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

        ## a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # such as Roberta that did not lowercase by default)
        ## the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
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

        ## b) remove duplicates and inflected forms of the complex word from the substitute list
        ## Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        ## then, remove duplicates and inflected forms of the complex word from the substitute list
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

        ## c) remove antonyms of the complex word from the substitute list
        ## get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        ## remove antonyms of the complex word from the list with substitutes
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

        # # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

        # print(
            # '----------------------------------------------------------------------------------------------------------'
            # '-----------------------------------')

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/trial/SS_phase1_{model_name_str}.tsv", sep="\t", index=False, header=False)
    print(f"SS_phase1_{model_name_str} exported to csv in path './predictions/trial/SS_phase1_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_1(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected
     forms, and antonyms of complex word. Then, sort the substitutes first that are synonyms with the complex word, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        # print the sentence and the complex word
        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        ## in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        ## concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        ## generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        ## remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        ## and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

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

        ## a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        ## the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        ## b) remove duplicates and inflected forms of the complex word from the substitute list
        ## Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        ## then, remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        ## c) remove antonyms of the complex word from the substitute list
        ## get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        ## remove antonyms of the complex word from the list with substitutes
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
        ## create two lists to hold synonyms and non-synonyms
        synonyms = []
        non_synonyms = []

        ## iterate through each substitute
        for substitute in substitutes_no_antonyms:
            substitute_synsets = wn.synsets(substitute)

            # get all the synsets for the complex word
            complex_word_synsets = wn.synsets(complex_word_lemma)

            # check if the substitute and the complex word share the same synset
            if any(substitute_synset == complex_word_synset for substitute_synset in substitute_synsets for
                   complex_word_synset in complex_word_synsets):
                # Add substitute to synonyms list
                synonyms.append(substitute)
            else:
                # Add substitute to non_synonyms list
                non_synonyms.append(substitute)

        ## print the lists of synonyms and non-synonyms
        # print(f"List of substitutes that are synonyms with the complex word '{complex_word}' in Wordnet: {synonyms}\n")
        # print(
            # f"List of substitutes that are NO synonyms with the complex word '{complex_word}' in Wordnet: "
            # f"{non_synonyms}\n")

        ## combine the lists with synonyms appearing first
        final_list = synonyms + non_synonyms
        # print(
            # f"Substitute Selection (SS) phase 2, option 1, step a): substitutes sorted on synonyms for the complex word"
            # f" '{complex_word}' first: {final_list}\n")

        ## limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = final_list[:10]
        # print(
            # f"Substitute Selection (SS) phase 2, option 1, final step b): top 10 substitutes, sorted on synonyms for "
            # f"the complex word '{complex_word}' first: {top_10_substitutes}\n")

        # print(
            # '----------------------------------------------------------------------------------------------------------'
            # '----------------------------------')

        # # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/trial/SS_phase2_option1Synsfirst_{model_name_str}.tsv", sep="\t", index=False,
                          header=False)
    print(
        f"SS_phase2_option1Synsfirst_{model_name_str} exported to csv in path "
        f"'./predictions/trial/SS_phase2_option1Synsfirst_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_2a(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected
     forms, and antonyms of complex word. Then, sort the substitutes first that have the same one-level up hypernyms as the complex word, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        # print the sentence and the complex word
        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        ## in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        ## concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        ## generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        ## remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        ## and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

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

        ## a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        ## the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        ## b) remove duplicates and inflected forms of the complex word from the substitute list
        ## Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        ## then, remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        ## c) remove antonyms of the complex word from the substitute list
        ## get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        ## remove antonyms of the complex word from the list with substitutes
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

        # Step a: Get first and second level hypernyms of the complex word
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

        # Step b: Get the first level hypernyms of the substitutes and check for shared hypernyms
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
            # f" intersection of one level up hypernyms complex word with one level up hypernyms of substitute:  "
            # f"{intersection_1}\n")

        # print(
            # f"Substitute Selection (SS) phase 2, option 2-a, step b): list of substitutes that share the same "
            # f"one-level hypernym with the complex word '{complex_word}' in Wordnet: {intersection_1_substitutes}\n")
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-a, step b): list of substitutes that DO NOT share the same "
            # f"one-level hypernym with the complex word '{complex_word}' in Wordnet: {other_1_substitutes}\n")

        ## step c: create the final list, by putting the intersection first
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

        # print(
            # '----------------------------------------------------------------------------------------------------------'
            # '-----------------------------------')

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/trial/SS_phase2_option2aHyps1first_{model_name_str}.tsv", sep="\t",
                          index=False, header=False)
    print(
        f"SS_phase2_option2aHyps1first_{model_name_str} exported to csv in path "
        f"'./predictions/trial/SS_phase2_option2aHyps1first_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_2b(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
     Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected
     forms, and antonyms of complex word. Then, sort the substitutes first that have the same two-level up hypernyms as the complex word, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        # print the sentence and the complex word
        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        ## in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        ## concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        ## generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        ## remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        ## and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

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

        ## a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        ## the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        ## b) remove duplicates and inflected forms of the complex word from the substitute list
        ## Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        ## then, remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        ## c) remove antonyms of the complex word from the substitute list
        ## get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        ## remove antonyms of the complex word from the list with substitutes
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

        # Step a: Get first and second level hypernyms of the complex word
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

        # Step b: Get second level hypernyms of the substitutes and check for shared hypernyms
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
            # f" intersection two level up hypernyms complex word with two levelup  hypernyms of substitute:  "
            # f"{intersection_2}\n")

        # print(
            # f"Substitute Selection (SS) phase 2, option 2-b, step b): list of substitutes that share the same two-level"
            # f" hypernym with the complex word '{complex_word}' in Wordnet: {intersection_2_substitutes}\n")
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-b, step b): list of substitutes that DO NOT share the same "
            # f"two-level hypernym with the complex word '{complex_word}' in Wordnet: {other_2_substitutes}\n")

        ## step c: create the final list, by putting the intersection first
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

        # print(
            # '----------------------------------------------------------------------------------------------------------'
            # '-----------------------------------')

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/trial/SS_phase2_option2bHyps2first_{model_name_str}.tsv", sep="\t",
                          index=False, header=False)
    print(
        f"SS_phase2_option2bHyps2first_{model_name_str} exported to csv in path "
        f"'./predictions/trial/SS_phase2_option2bHyps2first_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_2c(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected forms, and antonyms of complex word. Then, sort the substitutes first that have the same one- or two-level up hypernyms as the complex word, and store the top 10 results in a dataframe.
     
    :param data: the data from the loaded file, containing the sentences and the complex words.
    :param substitutes_df: the dataFrame to store the substitutes in.
    :param lm_tokenizer: a language model tokenizer based on pre-trained transformers.
    :param fill_mask: a pipeline for filling masked instances in the masked language model.
    :param model_name_str: the condensed name of the used model for saving in files.
    :param nlp: a SpaCy language model.
    :return: a dataframe containing the sentence, complex word, and 10 substitutes.
    """
    for index, row in data.iterrows():

        # print the sentence and the complex word
        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        ## in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        ## concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        ## generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (SG) step a): initial substitute list: {substitutes}\n")

        ## remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        ## and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

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

        ## a) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        ## the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Substitute Selection (SS) phase 1, step a): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        ## b) remove duplicates and inflected forms of the complex word from the substitute list
        ## Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to see if
        # their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        ## then, remove duplicates and inflected forms of the complex word from the substitute list
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Substitute Selection (SS) phase 1, step b): substitute list without duplicates nor inflected forms of
        # the complex word '{complex_word}': {substitutes_no_dupl_complex_word}\n")

        ## c) remove antonyms of the complex word from the substitute list
        ## get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        ## remove antonyms of the complex word from the list with substitutes
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

        # Step a: Get first and second level hypernyms of the complex word
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

        # Step b: Get first and second level hypernyms of the substitutes and check for shared hypernyms
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
                # # print(
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

        ## step c: create the final list, by putting the intersection first
        final_list = intersection_1_2_substitutes + other_1_2_substitutes
        # print(
            # f"Substitute Selection (SS) phase 2, option 2-c, step c): substitutes sorted on shared one- or two-level "
            # f"hypernyms with the complex word '{complex_word}' in Wordnet first:  {final_list}\n")

        # step d): limit the substitutes to the 10 highest ranked ones for evaluation
        top_10_substitutes = final_list[:10]
        # # print(
        #     f"Substitute Selection (SS) phase 2, option 2-c, final step d): top 10 of substitutes, sorted on shared "
        #     f"one- or two-level hypernyms with the complex word '{complex_word}' in Wordnet first: "
            # f"{top_10_substitutes}\n")

        # add the sentence, complex_word, and the substitutes to the dataframe
        substitutes_df.loc[index] = [sentence, complex_word] + top_10_substitutes

        # print(
            # '----------------------------------------------------------------------------------------------------------'
            # '-----------------------------------')

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/trial/SS_phase2_option2cHyps1+2first_{model_name_str}.tsv", sep="\t",
                          index=False, header=False)
    print(
        f"SS_phase2_option2cHyps1+2first_{model_name_str} exported to csv in path "
        f"'./predictions/trial/SS_phase2_option2cHyps1+2first_{model_name_str}.tsv'\n")

    return substitutes_df


def substitute_selection_phase_2_option_3(data, substitutes_df, lm_tokenizer, fill_mask, model_name_str, nlp,
                                          score_model, letter=''):
    """
    Generate 30 substitutes (by including the original unmasked sentence), remove noise and duplicates of substitutes; as well as duplicates, inflected forms, and antonyms of complex word. Then, sort the substitutes by their BERTScores, and store the top 10 results in a dataframe.
     
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

        # print the sentence and the complex word
        sentence, complex_word = row["sentence"], row["complex_word"]
        # print(f"Sentence: {sentence}")
        # print(f"Complex word: {complex_word}")

        # 1. Substitute Generation (SG): perform masking and generate substitutes:

        ## in the sentence, replace the complex word with a masked word
        sentence_masked_word = sentence.replace(complex_word, lm_tokenizer.mask_token)

        ## concatenate the original sentence and the masked sentence
        sentences_concat = f"{sentence} {lm_tokenizer.sep_token} {sentence_masked_word}"

        ## generate and rank candidate substitutes for the masked word using the fill_mask pipeline (removing elements
        # without token_str key; as this gave errors in the ELECTRA models) .
        top_k = 30
        result = fill_mask(sentences_concat, top_k=top_k)
        substitutes = [substitute["token_str"] for substitute in result if "token_str" in substitute]
        # print(f"Substitute Generation (step: initial substitute list: {substitutes}\n")

        # 2: Morphological Generation and Context Adaptation (Morphological Adaptation):
        ## a) remove noise in the substitutes, by ignoring generated substitutes that are empty or that have unwanted
        # punctuation characters or that start with '##' (this returned errors with the ELECTRA model), and lowercase
        # the substitutes (as some models don't lowercase by default)
        ## and lowercase all substitutes. Use try/except statement to prevent other character-related problems to happen

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

        ## b) remove duplicates within the substitute list from the substitute list (duplicates are likely for models
        # that did not lowercase by default)
        ## the last mentioned duplicate is removed on purpose, as this may probably be the (previously) uppercased
        # variant of the lowercased substitute (lowercased subs are most likely higher ranked by the model)
        substitutes_no_dupl = []
        for sub in substitutes:
            if sub not in substitutes_no_dupl:
                substitutes_no_dupl.append(sub)
        # print(f"Morphological Adaptation step b): substitute list without duplicates of substitutes:
        # {substitutes_no_dupl}\n")

        ## c) remove duplicates and inflected forms of the complex word from the substitute list

        ## first Lemmatize the complex word with spaCy, in order to compare it with the lemmatized substitute later to
        # see if their mutual lemmas are the same
        doc_complex_word = nlp(complex_word)
        complex_word_lemma = doc_complex_word[0].lemma_
        # print(f"complex_word_lemma for complex word '{complex_word}': {complex_word_lemma}\n")

        ## then, remove duplicates and inflected forms of the complex word from the list with substitutes
        substitutes_no_dupl_complex_word = []
        for substitute in substitutes_no_dupl:
            doc_substitute = nlp(substitute)
            substitute_lemma = doc_substitute[0].lemma_
            if substitute_lemma != complex_word_lemma:
                substitutes_no_dupl_complex_word.append(substitute)
        # print(f"Morphological Adaptation step c): substitute list without duplicates of the complex word nor inflected
        # forms of the complex word: {substitutes_no_dupl_complex_word}\n")

        ## d) remove antonyms of the complex word from the substitute list
        ## step 1: get the antonyms of the complex word
        antonyms_complex_word = []
        for syn in wn.synsets(complex_word_lemma):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    antonyms_complex_word.append(antonym.name())

        # print(f"Antonyms for complex word '{complex_word}': {antonyms_complex_word}\n")

        ## step 2: remove antonyms of the complex word from the list with substitutes
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

        ## create sentence with the complex word replaced by the substitutes
        sentence_with_substitutes = [sentence.replace(complex_word, sub) for sub in substitutes_no_antonyms]
        # print(f"List with sentences where complex word is substituted: {sentence_with_substitutes}\n")

        ## calculate BERTScores, and rank the substitutes based on these scores
        score_model_name_str = get_str_for_file_name(score_model)
        if len(sentence_with_substitutes) > 0:  # to make sure the list with substitutes is always filled
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

            # # print each substitute with its score
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

        # print('------------------------------------------------------------------------------------------------------
        # ---------------------------------------')

    # export the dataframe to tsv for evaluation
    substitutes_df.to_csv(f"./predictions/trial/SS_phase2_option3{letter}_BS{score_model_name_str}_{model_name_str}.tsv",
                          sep="\t", index=False, header=False)
    print(
        f"SS_phase2_option3{letter}_BS{score_model_name_str}_{model_name_str} exported to csv in path "
        f"'./predictions/trial/SS_phase2_option3{letter}_BS{score_model_name_str}_{model_name_str}.tsv'\n")

    return substitutes_df, score_model_name_str
