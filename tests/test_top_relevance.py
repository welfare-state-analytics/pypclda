import os
import math
import pytest
import logging
import unittest
import types
import pypclda
import pypclda.config_default as config_default
import tests.fixture as fixture
import numpy as np
import toolz

PCPLDA_JAR_PATH = os.path.join(os.getcwd(), "lib", "PCPLDA-8.5.1.jar")

import jpype
import jpype.imports
from jpype.types import *

cc = jpype.JPackage("cc")
java = jpype.JPackage("java")

"""
	public static String[][] getTopRelevanceWords(int noWords, int numTypes, int numTopics,
			int[][] typeTopicCounts, double beta, double lambda, Alphabet alphabet) {
		if(noWords>numTypes) {
			throw new IllegalArgumentException("Asked for more words (" + noWords + ") than there are types (unique words = noTypes = " + numTypes + ").");
		}
		IDSorter[] sortedWords = new IDSorter[numTypes];
		String [][] topTopicWords = new String[numTopics][noWords];

		double [][] p_w_k = calcWordProbGivenTopic(typeTopicCounts, beta);
		double [] p_w = calcWordProb(typeTopicCounts, beta);

		for (int topic = 0; topic < numTopics; topic++) {
			for (int type = 0; type < numTypes; type++) {
				double relevance = lambda * log(p_w_k[type][topic]) + (1-lambda) * (log(p_w_k[type][topic]) - log(p_w[type]));
				sortedWords[type] = new IDSorter(type, relevance);
			}

			Arrays.sort(sortedWords);

			for (int i=0; i < noWords && i < topTopicWords[topic].length && i < numTypes; i++) {
				topTopicWords[topic][i] = (String) alphabet.lookupObject(sortedWords[i].getID());
			}
		}
		return topTopicWords;
	}

	public static String[][] getTopDistinctiveWords(int noWords, int numTypes, int numTopics,
			int[][] typeTopicCounts, double beta, Alphabet alphabet) {
		if(noWords>numTypes) {
			throw new IllegalArgumentException("Asked for more words (" + noWords + ") than there are types (unique words = noTypes = " + numTypes + ").");
		}
		IDSorter[] sortedWords = new IDSorter[numTypes];
		String [][] topTopicWords = new String[numTopics][noWords];

		double [][] p_w_k = calcTopicProbGivenWord(typeTopicCounts, beta);
		double [] p_w = calcWordProb(typeTopicCounts, beta);

		double [][] distinctiveness = calcWordDistinctiveness(p_w_k, p_w);

		for (int topic = 0; topic < numTopics; topic++) {
			for (int type = 0; type < numTypes; type++) {
				sortedWords[type] = new IDSorter(type, distinctiveness[type][topic]);
			}

			Arrays.sort(sortedWords);

			for (int i=0; i < noWords && i < topTopicWords[topic].length && i < numTypes; i++) {
				topTopicWords[topic][i] = (String) alphabet.lookupObject(sortedWords[i].getID());
			}
		}
		return topTopicWords;
	}

	public static String[][] getTopSalientWords(int noWords, int numTypes, int numTopics,
			int[][] typeTopicCounts, double beta, Alphabet alphabet) {
		if(noWords>numTypes) {
			throw new IllegalArgumentException("Asked for more words (" + noWords + ") than there are types (unique words = noTypes = " + numTypes + ").");
		}
		IDSorter[] sortedWords = new IDSorter[numTypes];
		String [][] topTopicWords = new String[numTopics][noWords];

		double [][] p_w_k = calcTopicProbGivenWord(typeTopicCounts, beta);
		double [] p_w = calcWordProb(typeTopicCounts, beta);

		double [][] saliency = calcWordSaliency(p_w_k,p_w);

		for (int topic = 0; topic < numTopics; topic++) {
			for (int type = 0; type < numTypes; type++) {
				sortedWords[type] = new IDSorter(type, saliency[type][topic]);
			}

			Arrays.sort(sortedWords);

			for (int i=0; i < noWords && i < topTopicWords[topic].length && i < numTypes; i++) {
				topTopicWords[topic][i] = (String) alphabet.lookupObject(sortedWords[i].getID());
			}
		}
		return topTopicWords;
	}

	/**
	 * Calculate word distinctiveness as defined in:
	 * Termite: Visualization Techniques for Assessing Textual Topic Models
	 * by Jason Chuang, Christopher D. Manning, Jeffrey Heer
	 * @param p_w_k probability of topic given word
	 * @param p_w probability of a word
	 * @return array with word distinctiveness measures dim(array) = nrWords x nrTopics
	 */
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

	/**
	 * Calculate word saliency as defined in:
	 * Termite: Visualization Techniques for Assessing Textual Topic Models
	 * by Jason Chuang, Christopher D. Manning, Jeffrey Heer
	 * @param p_w_k probability of topic given word
	 * @param p_w probability of a word
	 * @return array with word saliency measures
	 */
	public static double[][] calcWordSaliency(double [][] p_w_k, double [] p_w) {
		int nrTopics = p_w_k[0].length;
		int nrWords = p_w_k.length;
		double [][] wordDistinctiveness = calcWordDistinctiveness(p_w_k, p_w);
		double [][] wordSaliency = new double[nrWords][nrTopics];
		for (int w = 0; w < wordSaliency.length; w++) {
			for (int k = 0; k < nrTopics; k++) {
				wordSaliency[w][k] = p_w[w] * wordDistinctiveness[w][k];
			}
		}
		return wordSaliency;
	}
"""
def test_calculate_token_probs():

    #  Arrange
    lda_util = cc.mallet.util.LDAUtils()
    config = fixture.fixture_lda_config()
    sampler = fixture.fixture_sampler()
    beta = config.getBeta(config_default.BETA_DEFAULT)
    type_topic_counts = pypclda.get_token_topic_matrix(sampler)

    # Act
    word_probs_python = pypclda.calculate_token_probs(type_topic_counts, beta)

    # Assert
    word_probs_java = lda_util.calcWordProb(type_topic_counts, beta)

    assert np.allclose(word_probs_java, word_probs_python, rtol=1e-05)

def test_calculate_word_prob_given_topic():
    #  Arrange
    lda_util = cc.mallet.util.LDAUtils()
    config = fixture.fixture_lda_config()
    sampler = fixture.fixture_sampler()
    beta = config.getBeta(config_default.BETA_DEFAULT)
    type_topic_counts = pypclda.get_token_topic_matrix(sampler)

    # Act
    word_probs_python = pypclda.calculate_token_probs_given_topic(type_topic_counts, beta)

    # Assert
    word_probs_java = lda_util.calcWordProbGivenTopic(type_topic_counts, beta)

    assert np.allclose(word_probs_java, word_probs_python, rtol=1e-05)

def test_calculate_token_relevance_matrix():

    #  Arrange
    config = fixture.fixture_lda_config()
    sampler = fixture.fixture_sampler()

    beta = config.getBeta(config_default.BETA_DEFAULT)
    type_topic_counts = pypclda.get_token_topic_matrix(sampler)
    n_types = len(type_topic_counts)
    n_topics = len(type_topic_counts[0])
    v_lambda = config.getLambda(config_default.LAMBDA_DEFAULT)

    # Act
    token_relevance_matrix_python = pypclda.calculate_token_relevance_matrix(type_topic_counts, beta, v_lambda)

    # Assert
    token_relevance_matrix_java = _getRelevanceWordsMatrixByJava(n_types, n_topics, type_topic_counts, beta, v_lambda)

    token_relevance_matrix_java = np.array([ list(x) for x in token_relevance_matrix_java ])

    assert np.allclose(token_relevance_matrix_java, token_relevance_matrix_python, rtol=1e-10)

def test_calculate_top_relevance_tokens():
    #  Arrange
    n_top_tokens = 20

    lda_util = cc.mallet.util.LDAUtils()
    config = fixture.fixture_lda_config()
    sampler = fixture.fixture_sampler()
    beta = config.getBeta(config_default.BETA_DEFAULT)
    type_topic_counts = pypclda.get_token_topic_matrix(sampler)
    n_types = len(type_topic_counts)
    n_topics = len(type_topic_counts[0])
    v_lambda = config.getLambda(config_default.LAMBDA_DEFAULT)
    alphabet = sampler.getAlphabet()

    # Act
    top_token_relevance_python = calculate_top_relevance_tokens(
        n_top_tokens, type_topic_counts, beta, v_lambda, alphabet
    )

    # Assert
    top_token_relevance_java = lda_util.getTopRelevanceWords(
        n_top_tokens, n_types, n_topics, type_topic_counts,
        beta, v_lambda, sampler.getAlphabet()
    )

    python_words = [
        [ w[0] for w in row ]
            for row in top_token_relevance_python
    ]

    top_token_relevance_java = [ list(x) for x in top_token_relevance_java ]

    # assert python_words == top_token_relevance_java

def _getRelevanceWordsMatrixByJava(numTypes, numTopics, typeTopicCounts, beta, vlambda):

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