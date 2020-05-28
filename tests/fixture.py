import os
import pypclda

def fixture_lda_config():

    corpus_filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
    stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
    sampler_folder = os.path.join(os.getcwd(), "tests/resources/stored_samplers", "stoplist.txt")

    lda_config = pypclda.create_simple_lda_config(
        dataset_filename=corpus_filename,
        stoplist_filename=stoplist_filename
    )

    return lda_config

def fixture_dataset(lda_config):

    filename = str(lda_config.getDatasetFilename())

    with open(filename) as f:
        train_corpus = f.readlines()

    dataset = pypclda.create_lda_dataset(
        train_corpus, None,
        lda_config.getStoplistFilename("")
    )
    return dataset

def fixture_sampler():
    config = fixture_lda_config()
    sampler_folder = str(config.getSavedSamplerDirectory(""))
    sampler = pypclda.load_lda_sampler(config, stored_dir=config.getSavedSamplerDirectory(""))
    return sampler