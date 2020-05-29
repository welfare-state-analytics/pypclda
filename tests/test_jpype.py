import os
import math
import pytest
import logging
import unittest
import types
import numpy as np
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
    config = pypclda.create_simple_lda(args)
    assert config is not None
    assert args["nr_topics"] == config.getNoTopics()

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
    config = pypclda.create_simple_lda_config(dataset_filename=corpus_filename, stoplist_filename=stoplist_filename)
    instances = pypclda.load_lda_dataset(corpus_filename, config)

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
    config = pypclda.create_simple_lda_config(dataset_filename=dataset_filename, stoplist_filename=stoplist_filename)
    instances = pypclda.load_lda_dataset(dataset_filename, config)

    assert 10 == instances.size()
    assert "X" == util.instanceLabelToString(instances.get(0))

    assert "docno:1" == util.instanceIdToString(instances.get(0))
    assert "X" == util.instanceLabelToString(instances.get(9))
    assert "docno:10" == util.instanceIdToString(instances.get(9))

def test_create_sampler_of_type_by_factory():
    config = pypclda.create_simple_lda_config()
    sampler_type = "cc.mallet.topics.PolyaUrnSpaliasLDA"
    sampler = pypclda.create_lda_sampler_of_type_with_factory(config, sampler_type)
    assert sampler is not None

def test_create_sampler_of_type():
    config = pypclda.create_simple_lda_config()
    sampler_type = "cc.mallet.topics.PolyaUrnSpaliasLDA"
    sampler = pypclda.create_lda_sampler_of_type(config, sampler_type)
    assert sampler is not None

def test_cast_lda_sampler_to_gibbs_sampler():

    filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
    stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
    config = pypclda.create_simple_lda_config(dataset_filename=filename, stoplist_filename=stoplist_filename)
    sampler = pypclda.create_lda_sampler_of_type(config, "cc.mallet.topics.PolyaUrnSpaliasLDA")

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

def test_extract_id2token():

    sampler = fixture.fixture_sampler()
    alphabet = sampler.getAlphabet()

    id2token = pypclda.extract_vocabulary(alphabet)

    assert all(( id2token[i] == str(w) for i, w in enumerate(alphabet.toArray()) ))


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
    #assert expected_max_count == max([ max(x) for x in document_topic_matrix ])

def test_get_topic_token_phi_matrix():

    expected_token_count = 982
    expected_topic_count = 20
    expected_max_phi = 1.0

    sampler = fixture.fixture_sampler()

    topic_token_phi_matrix = pypclda.get_topic_token_phi_matrix(sampler)

    assert topic_token_phi_matrix is not None
    assert expected_topic_count == len(topic_token_phi_matrix)
    assert expected_token_count == len(topic_token_phi_matrix[0])
    assert expected_max_phi >= max([ max(x) for x in topic_token_phi_matrix ])

    for t in range(0, expected_topic_count):
        assert math.isclose(1.0, sum(topic_token_phi_matrix[t]), rel_tol=1e-5)

def test_compute_token_probabilities():

    #  Arrange
    lda_util = cc.mallet.util.LDAUtils()
    config = fixture.fixture_lda_config()
    sampler = fixture.fixture_sampler()
    beta = config.getBeta(config_default.BETA_DEFAULT)
    type_topic_counts = pypclda.get_token_topic_matrix(sampler)

    # Act
    word_probs_python = pypclda.compute_token_probabilities(type_topic_counts, beta)

    # Assert
    word_probs_java = lda_util.calcWordProb(type_topic_counts, beta)

    assert np.allclose(word_probs_java, word_probs_python, rtol=1e-05)

def test_compute_token_probabilities_given_topic():
    #  Arrange
    lda_util = cc.mallet.util.LDAUtils()
    config = fixture.fixture_lda_config()
    sampler = fixture.fixture_sampler()
    beta = config.getBeta(config_default.BETA_DEFAULT)
    type_topic_counts = pypclda.get_token_topic_matrix(sampler)

    # Act
    word_probs_python = pypclda.compute_token_probabilities_given_topic(type_topic_counts, beta)

    # Assert
    word_probs_java = lda_util.calcWordProbGivenTopic(type_topic_counts, beta)

    assert np.allclose(word_probs_java, word_probs_python, rtol=1e-05)

    """
	public static double[][] calcWordDistinctiveness(double [][] p_w_k, double [] p_w) {
		int nrTopics = p_w_k[0].length;
		int nrWords = p_w_k.length;
		double [][] wordDistinctiveness = new double[nrWords][nrTopics];
		for (int w = 0; w < nrWords; w++) {
			for (int k = 0; k < nrTopics; k++) {
				wordDistinctiveness[w][k] += p_w_k[w][k] * log(p_w_k[w][k] / p_w[w]);
			}
		}
		return wordDistinctiveness;
	}
    """

def test_compute_distinctiveness_matrix():

    #  Arrange
    config = fixture.fixture_lda_config()
    sampler = fixture.fixture_sampler()
    token_topic_count_matrix = sampler.getTypeTopicMatrix()

    beta     = config.getBeta(config_default.BETA_DEFAULT)
    p_w_k    = pypclda.compute_token_probabilities_given_topic(token_topic_count_matrix, beta)
    p_w      = pypclda.compute_token_probabilities(token_topic_count_matrix, beta)

    # Act
    python_matrix = pypclda.compute_distinctiveness_matrix(p_w_k, p_w)

    # Assert
    java_matrix = cc.mallet.util.LDAUtils.calcWordDistinctiveness(p_w_k, p_w)
    java_matrix = np.array([ list(x) for x in java_matrix ])

    assert np.allclose(java_matrix, python_matrix, rtol=1e-10)

def test_get_top_distinctive_topic_tokens2():
    """Tests a "port" of cc.mallet.util.LDAUtils.getTopDistinctiveWords
    """
    #  Arrange
    n_top_tokens = 20
    sampler = fixture.fixture_sampler()
    config = fixture.fixture_lda_config()

    # Act
    python_words = pypclda.get_top_distinctive_topic_tokens2(
        sampler, config, n_top_tokens
    )

    # Assert
    type_topic_count_matrix = sampler.getTypeTopicMatrix()
    java_words = cc.mallet.util.LDAUtils().getTopDistinctiveWords(
        n_top_tokens,
        len(type_topic_count_matrix),
        len(type_topic_count_matrix[0]),
        type_topic_count_matrix,
        config.getBeta(config_default.BETA_DEFAULT),
        sampler.getAlphabet()
    )

    python_words = [[ w[0] for w in row ] for row in python_words ]
    java_words = [ list(x) for x in java_words ]

    assert len(python_words) == len(java_words)
    assert len(python_words[0]) == len(java_words[0])


def __java_compute_token_relevance_matrix(numTypes, numTopics, typeTopicCounts, beta, vlambda):

    lda_util = cc.mallet.util.LDAUtils()

    p_w_k = lda_util.calcWordProbGivenTopic(typeTopicCounts, beta)
    p_w = lda_util.calcWordProb(typeTopicCounts, beta)

    relevance_matrix = []

    for topic in range(0, numTopics):

        topic_relevances = []

        for type in range(0, numTypes):
            relevance = vlambda * math.log(p_w_k[type][topic]) + (1-vlambda) * (math.log(p_w_k[type][topic]) - math.log(p_w[type]))
            topic_relevances.append(relevance)

        relevance_matrix.append(topic_relevances)

    return np.array(relevance_matrix)

def test_compute_token_relevance_matrix():

    #  Arrange
    config = fixture.fixture_lda_config()
    sampler = fixture.fixture_sampler()

    beta = config.getBeta(config_default.BETA_DEFAULT)
    vlambda = config.getLambda(config_default.LAMBDA_DEFAULT)
    type_topic_counts = pypclda.get_token_topic_matrix(sampler)
    n_types = len(type_topic_counts)
    n_topics = len(type_topic_counts[0])

    # Act
    token_relevance_matrix_python = pypclda.compute_token_relevance_matrix(type_topic_counts, beta, vlambda)

    # Assert
    token_relevance_matrix_java = __java_compute_token_relevance_matrix(n_types, n_topics, type_topic_counts, beta, vlambda)
    token_relevance_matrix_java = np.array([ list(x) for x in token_relevance_matrix_java ])

    assert np.allclose(token_relevance_matrix_java, token_relevance_matrix_python, rtol=1e-10)

def test_get_top_relevance_topic_tokens():
    """Tests call to cc.mallet.util.LDAUtils.getTopRelevanceWords
    TODO: fix equality test of word (different sort order when value the same)
    """
    n_top_words = 20
    sampler = fixture.fixture_sampler()
    config = fixture.fixture_lda_config()

    relevances = pypclda.get_top_topic_word_relevances(sampler, config, n_top_words=n_top_words)

    assert relevances is not None
    assert int(sampler.getNoTopics()) == len(relevances)
    assert n_top_words == len(relevances[0])
    assert relevances is not None

def test_get_top_relevance_topic_tokens2():
    """Tests a "port" of cc.mallet.util.LDAUtils.getTopRelevanceWords
    """
    #  Arrange
    n_top_tokens = 20
    sampler = fixture.fixture_sampler()
    config = fixture.fixture_lda_config()

    # Act
    top_token_relevance_python = pypclda.get_top_relevance_topic_tokens2(
        sampler, config, n_top_tokens
    )

    # Assert
    type_topic_count_matrix = sampler.getTypeTopicMatrix()
    java_words = cc.mallet.util.LDAUtils().getTopRelevanceWords(
        n_top_tokens,
        len(type_topic_count_matrix),
        len(type_topic_count_matrix[0]),
        type_topic_count_matrix,
        config.getBeta(config_default.BETA_DEFAULT),
        config.getLambda(config_default.LAMBDA_DEFAULT),
        sampler.getAlphabet()
    )

    python_words = [[ w[0] for w in row ] for row in top_token_relevance_python ]
    java_words = [ list(x) for x in java_words ]

    assert len(python_words) == len(java_words)
    assert len(python_words[0]) == len(java_words[0])

def test_get_top_topic_tokens():

    expected_token_count = 30
    expected_topic_count = 20

    sampler = fixture.fixture_sampler()

    top_topic_words = pypclda.get_top_topic_tokens(sampler, expected_token_count)

    assert top_topic_words is not None
    assert expected_topic_count == len(top_topic_words)

    for t in top_topic_words:
        assert expected_token_count == len(t)


def test_get_top_topic_tokens2():
    """Tests a "port" of cc.mallet.util.LDAUtils.getTopRelevanceWords
    """
    #  Arrange
    n_top_tokens = 20

    sampler = fixture.fixture_sampler()

    # Act
    top_tokens_python = pypclda.get_top_topic_tokens2(
        sampler, n_top_tokens
    )

    # Assert
    type_topic_counts = sampler.getTypeTopicMatrix()
    top_tokens_java = cc.mallet.util.LDAUtils().getTopWords(
        n_top_tokens,
        len(sampler.getTypeTopicMatrix()),
        len(sampler.getTypeTopicMatrix()[0]),
        sampler.getTypeTopicMatrix(),
        sampler.getAlphabet()
    )

    python_words = [ [ w[0] for w in row ] for row in top_tokens_python ]
    java_words = [ list(x) for x in top_tokens_java ]

    assert len(python_words) == len(java_words)
    assert len(python_words[0]) == len(java_words[0])

def getRandomNumber():
    """ This number has been chosen by a fair dice roll and is hence guaranteed to be random. """
    return 4

def test_zero_division():
    with pytest.raises(ZeroDivisionError):
        1 / 0
