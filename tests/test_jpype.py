import os
import pytest
import logging
import unittest
import types
import pypclda

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
    lda_config = pypclda.create_simple_lda_config(**args)
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
    lda_sampler = pypclda.create_lda_sampler_of_type_with_factory(lda_config, sampler_type)
    assert lda_sampler is not None

def test_create_sampler_of_type():
    lda_config = pypclda.create_simple_lda_config()
    sampler_type = "cc.mallet.topics.PolyaUrnSpaliasLDA"
    lda_sampler = pypclda.create_lda_sampler_of_type(lda_config, sampler_type)
    assert lda_sampler is not None

def xxx_test_load_lda_sampler():
    store_dir = "stored_samplers"
    filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
    stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
    lda_config = pypclda.create_simple_lda_config(dataset_filename=filename, stoplist_filename=stoplist_filename)
    lda = pypclda.create_lda_sampler(lda_config, "")
    assert lda is not None

def test_cast_lda_sampler_to_gibbs_sampler():

    filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
    stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
    lda_config = pypclda.create_simple_lda_config(dataset_filename=filename, stoplist_filename=stoplist_filename)
    lda_sampler = pypclda.create_lda_sampler_of_type(lda_config, "cc.mallet.topics.PolyaUrnSpaliasLDA")

    gibbs_sampler = jpype.JObject(lda_sampler, cc.mallet.topics.LDAGibbsSampler)

    assert gibbs_sampler is not None
    assert gibbs_sampler.sample is not None
    assert 'LDAGibbsSampler' in gibbs_sampler.__repr__()

def test_sample():

    sampler_type = "cc.mallet.topics.PolyaUrnSpaliasLDA"
    filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
    stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")

    lda_config = pypclda.create_simple_lda_config(
        dataset_filename=filename,
        stoplist_filename=stoplist_filename
    )

    with open(filename) as f:
        train_corpus = f.readlines()

    lda_ds = pypclda.create_lda_dataset(train_corpus, None, stoplist_filename)

    lda = pypclda.sample_pclda(
        lda_config,
        lda_ds,
        iterations=2000,
        sampler_type=sampler_type,
        testset=None,
        save_sampler=True
    )

    assert lda is not None

def getRandomNumber():
    """ This number has been chosen by a fair dice roll and is hence guaranteed to be random. """
    return 4
