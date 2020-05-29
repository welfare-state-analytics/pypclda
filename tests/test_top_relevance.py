import os
import math
import pytest
import logging
import unittest
import types
import pypclda
import pypclda.config_default as config_default
import pypclda.utility as utility
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

