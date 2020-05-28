import os
import math
import pytest
import logging
import unittest
import types
import pypclda
import pypclda.config_default as config_default
import tests.fixture as fixture

PCPLDA_JAR_PATH = os.path.join(os.getcwd(), "lib", "PCPLDA-8.5.1.jar")

import jpype
import jpype.imports
from jpype.types import *

# jpype.startJVM(classpath=[PCPLDA_JAR_PATH])

cc = jpype.JPackage("cc")
java = jpype.JPackage("java")


def test_create_lda_util():
    util = pypclda.get_util()
    assert util is not None

def test_create_logging_util():
    log_folder = "/tmp/hello"
    logging_util = pypclda.get_logging_utils(log_folder)
    assert logging_util is not None

def test_create_lda_config():
    args = dict(
        nr_topics=99,
        alpha=0.05,
        beta=0.001
    )
    lda_config = pypclda.create_simple_lda(args)
    assert lda_config is not None
    assert args["nr_topics"] == lda_config.getNoTopics()

def test_create_lda_dataset():
    filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
    stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
    with open(filename) as f:
        train_corpus = f.readlines()

    lda_ds = pypclda.create_lda_dataset(train_corpus, None, stoplist_filename)

    assert lda_ds is not None

def test_load_lda_dataset():
    corpus_filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
    stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")

    util = pypclda.get_util()
    lda_config = pypclda.create_simple_lda_config(dataset_filename=corpus_filename, stoplist_filename=stoplist_filename)
    instances = pypclda.load_lda_dataset(corpus_filename, lda_config)

    assert instances is not None
    assert 10 == instances.size()

    assert "XYZ" == util.instanceLabelToString(instances.get(0))
    assert "doc#0" == util.instanceIdToString(instances.get(0))
    assert "XYZ" == util.instanceLabelToString(instances.get(9))
    assert "doc#9" == util.instanceIdToString(instances.get(9))

def test_load_zipped_dataset():
    dataset_filename = os.path.join(os.getcwd(), "tests/resources/datasets/small.txt.zip")
    stoplist_filename = os.path.join(os.getcwd(), "tests/stoplist.txt")

    util = pypclda.get_util()
    lda_config = pypclda.create_simple_lda_config(dataset_filename=dataset_filename, stoplist_filename=stoplist_filename)
    instances = pypclda.load_lda_dataset(dataset_filename, lda_config)

    assert 10 == instances.size()
    assert "X" == util.instanceLabelToString(instances.get(0))

    assert "docno:1" == util.instanceIdToString(instances.get(0))
    assert "X" == util.instanceLabelToString(instances.get(9))
    assert "docno:10" == util.instanceIdToString(instances.get(9))

def test_create_sampler_of_type_by_factory():
    lda_config = pypclda.create_simple_lda_config()
    sampler_type = "cc.mallet.topics.PolyaUrnSpaliasLDA"
    sampler = pypclda.create_lda_sampler_of_type_with_factory(lda_config, sampler_type)
    assert sampler is not None

def test_create_sampler_of_type():
    lda_config = pypclda.create_simple_lda_config()
    sampler_type = "cc.mallet.topics.PolyaUrnSpaliasLDA"
    sampler = pypclda.create_lda_sampler_of_type(lda_config, sampler_type)
    assert sampler is not None

def test_cast_lda_sampler_to_gibbs_sampler():

    filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
    stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
    lda_config = pypclda.create_simple_lda_config(dataset_filename=filename, stoplist_filename=stoplist_filename)
    sampler = pypclda.create_lda_sampler_of_type(lda_config, "cc.mallet.topics.PolyaUrnSpaliasLDA")

    gibbs_sampler = jpype.JObject(sampler, cc.mallet.topics.LDAGibbsSampler)

    assert gibbs_sampler is not None
    assert gibbs_sampler.sample is not None
    assert 'LDAGibbsSampler' in gibbs_sampler.__repr__()

def test_sample_pclda():

    sampler_type = "cc.mallet.topics.PolyaUrnSpaliasLDA"

    config = fixture.fixture_lda_config()
    dataset = fixture.fixture_dataset(config)

    sampler = pypclda.sample_pclda(
        config,
        dataset,
        iterations=2000,
        sampler_type=sampler_type,
        testset=None,
        save_sampler=True
    )

    assert sampler is not None

def test_load_lda_sampler():

    expected_sampler_type = "cc.mallet.topics.PolyaUrnSpaliasLDA"

    config = fixture.fixture_lda_config()
    sampler_folder = str(config.getSavedSamplerDirectory(""))

    sampler = pypclda.load_lda_sampler(config, stored_dir=config.getSavedSamplerDirectory(""))

    assert sampler is not None
    assert expected_sampler_type == sampler.getClass().getName()

def test_get_alphabet():

    sampler = fixture.fixture_sampler()

    alphabet = pypclda.get_alphabet(sampler)

    assert alphabet is not None
    assert alphabet.size() == 982

def test_extract_vocabulary():

    sampler = fixture.fixture_sampler()
    alphabet = sampler.getAlphabet()

    vocabulary = pypclda.extract_vocabulary(alphabet)

    assert vocabulary is not None
    assert 982 == len(vocabulary)
    assert 982 == len(set(vocabulary))

def test_extract_token_counts():

    expected_token_count = 982
    expected_max_count = 157

    sampler = fixture.fixture_sampler()

    token_counts = pypclda.extract_token_counts(sampler.getDataset())

    assert token_counts is not None
    assert expected_token_count == len(token_counts)
    assert expected_max_count == max(token_counts)

def test_extract_doc_lengths():

    expected_doc_count = 10
    expected_max_doc_length = 685

    sampler = fixture.fixture_sampler()

    doc_lengths = pypclda.extract_doc_lengths(sampler.getDataset())

    assert doc_lengths is not None
    assert expected_doc_count == len(doc_lengths)
    assert expected_max_doc_length == max(doc_lengths)

def test_get_token_topic_matrix():

    expected_token_count = 982
    expected_topic_count = 20
    expected_max_count = 157

    sampler = fixture.fixture_sampler()

    token_topic_matrix = pypclda.get_token_topic_matrix(sampler)

    assert token_topic_matrix is not None
    assert expected_token_count == len(token_topic_matrix)
    assert expected_topic_count == len(token_topic_matrix[0])
    assert expected_max_count == max([ max(x) for x in token_topic_matrix ])

def test_get_document_topic_matrix():

    expected_document_count = 10
    expected_topic_count = 20
    expected_max_count = 493

    sampler = fixture.fixture_sampler()

    document_topic_matrix = pypclda.get_document_topic_matrix(sampler)

    assert document_topic_matrix is not None
    assert expected_document_count == len(document_topic_matrix)
    assert expected_topic_count == len(document_topic_matrix[0])
    assert expected_max_count == max([ max(x) for x in document_topic_matrix ])

def test_get_topic_token_phi_matrix():

    expected_token_count = 982
    expected_topic_count = 20
    expected_max_phi = 1.0

    sampler = fixture.fixture.fixture_sampler()

    topic_token_phi_matrix = pypclda.get_topic_token_phi_matrix(sampler)

    assert topic_token_phi_matrix is not None
    assert expected_topic_count == len(topic_token_phi_matrix)
    assert expected_token_count == len(topic_token_phi_matrix[0])
    assert expected_max_phi >= max([ max(x) for x in topic_token_phi_matrix ])

    for t in range(0, expected_topic_count):
        assert math.isclose(1.0, sum(topic_token_phi_matrix[t]), rel_tol=1e-5)

def test_get_top_topic_tokens():

    expected_token_count = 30
    expected_topic_count = 20
    expected_max_phi = 1.0

    sampler = fixture.fixture_sampler()

    top_topic_words = pypclda.get_top_topic_tokens(sampler, expected_token_count)

    assert top_topic_words is not None
    assert expected_topic_count == len(top_topic_words)

    for t in top_topic_words:
        assert expected_token_count == len(t)


def test_get_top_topic_word_relevances():

    n_top_words = 20
    sampler = fixture.fixture_sampler()
    config = fixture.fixture_lda_config()

    relevances = pypclda.get_top_topic_word_relevances(sampler, config, n_top_words=n_top_words)

    assert relevances is not None
    assert int(sampler.getNoTopics()) == len(relevances)
    assert n_top_words == len(relevances[0])
    assert relevances is not None

def getRandomNumber():
    """ This number has been chosen by a fair dice roll and is hence guaranteed to be random. """
    return 4


def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0
datasetdatasetconfigconfig