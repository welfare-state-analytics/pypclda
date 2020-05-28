import os
import types
import numpy as np
import toolz
import pypclda.utility as utility
import pypclda.config_default as config_default
import jpype
import jpype.imports
from jpype.types import *

logger = utility.get_logger()

PCPLDA_JAR_PATH = os.path.join(os.getcwd(), "lib", "PCPLDA-8.5.1.jar")

jpype.startJVM(classpath=[PCPLDA_JAR_PATH], convertStrings=False)

cc = jpype.JPackage("cc")
java = jpype.JPackage("java")

lda_util = cc.mallet.util.LDAUtils()

get_util = lambda: lda_util
jint = lambda n: java.lang.Integer(n)
jdbl = lambda d: java.lang.Double(d)

def create_logging_utils(output_folder=None):
    logging_util = cc.mallet.util.LoggingUtils()
    if not output_folder is None:
        logging_util.checkAndCreateCurrentLogDir(JString(output_folder))
    return logging_util

def create_simple_lda_config(**kwargs):
    """Creates a new LDA config file

    Parameters
    ----------
        dataset_filename string
            filename of dataset (in LDA format)
        nr_topics int
            number of topics to use
        alpha float
            symmetric alpha prior
        beta float
            symmetric beta prior
        iterations int
            number of iterations to sample
        rareword_threshold [type]
            min. number of occurences of a word to be kept
        optim_interval int
            how often to do hyperparameter optimization (default is off = -1)
        stoplist_filename string
            filenname of stoplist file (one word per line) (default "stoplist.txt")
        topic_interval int
            how often to print topic info during sampling
        tmp_folder string
            temporary directory for intermediate storage of logging data (default "tmp")
        topic_priors string
            text file with 'prior spec' with one topic per line with format: <topic nr(zero idxed)>, <word1>, <word2>, etc

    Returns
    -------
    cc.mallet.configuration.SimpleLDAConfiguration

    """

    timestamp = utility.get_timestamp2()
    output_folder = os.path.join("output", f"run_{timestamp}", "")
    log_folder = os.path.join(output_folder, "log", "")
    tmp_folder = kwargs.get("tmp_folder", "/tmp")

    logging_util = create_logging_utils(tmp_folder)

    slc = cc.mallet.configuration.SimpleLDAConfiguration()

    slc.setLoggingUtil(logging_util)

    if "experiment_output_directory" in kwargs:
        slc.setExperimentOutputDirectory(kwargs.get("experiment_output_directory", None) or "output")

    slc.setAlpha(java.lang.Double(kwargs.get("alpha", 0.01)))
    slc.setBeta(java.lang.Double(kwargs.get("beta", kwargs.get("nr_topics", 20) / 50.0)))
    slc.setDatasetFilename(kwargs.get("dataset_filename", "corpus.txt"))
    slc.setHyperparamOptimInterval(java.lang.Integer(kwargs.get("optim_interval", -1)))
    slc.setNoBatches(java.lang.Integer(5))
    slc.setNoIters(java.lang.Integer(kwargs.get("iterations", 2000)))
    slc.setNoPreprocess(kwargs.get("no_preprocess", True))
    slc.setNoTopics(kwargs.get("nr_topics", 20))
    slc.setRareThreshold(java.lang.Integer(kwargs.get("rareword_threshold", 10)))
    slc.setStartDiagnostic(java.lang.Integer(90))
    slc.setStoplistFilename(kwargs.get("stoplist_filename", "stoplist.txt"))
    slc.setTopicInterval(java.lang.Integer(kwargs.get("topic_interval", 10)))
    slc.setTopicPriorFilename(kwargs.get("topic_priors", "topic_priors.txt"))

    if "corpus_filename" in kwargs:
        slc.setSaveCorpus(True)
        slc.setCorpusFilename(kwargs.get("corpus_filename", "corpus.txt"))

    if "vocabulary_filename" in kwargs:
        slc.setSaveVocabulary(True)
        slc.setVocabularyFn(kwargs.get("vocabulary_filename", "vocabulary.txt"))

    if "document_topic_means_filename" in kwargs:
        slc.setSaveDocumentTopicMeans(True)
        slc.setDocumentTopicMeansOutputFilename(kwargs.get("document_topic_means_filename", "document_topic_means.txt"))

    if "phi_means_output_filename" in kwargs:
        slc.setSavePhi(True)
        slc.setPhiMeansOutputFilename(kwargs.get("phi_means_output_filename", "phi_means.txt"))

    if "doc_lengths_filename" in kwargs:
        slc.setSaveDocLengths(True)
        slc.setDocLengthsFilename(kwargs.get("doc_lengths_filename", "doc_lengths.txt"))

    if "document_topic_theta_output_filename" in kwargs:
        slc.setSaveDocumentTopicTheta(True)
        slc.setDocumentTopicThetaOutputFilename(kwargs.get("document_topic_theta_output_filename", "document_topic_theta.txt"))

    if "sampler_folder" in kwargs:
        slc.setSavedSamplerDirectory(kwargs["sampler_folder"])

    if "save_term_frequencies" in kwargs:
        slc.setSaveTermFrequencies(kwargs["save_term_frequencies"])

    logger.info("Logging to: " + output_folder)

    return slc

def create_lda_dataset(train_corpus, test_corpus=None, stoplist_filename="stoplist.txt"):
    """ Create an LDA dataset from existing string vector.

        Each entry in the vector must be a string with the following format:

            unique-id TAB `doc-class` TAB tokens....

        The document class is not used in by the LDA sampler.
        The document content CAN have TAB in it.

    Parameters
    ----------
    train_corpus : string vector
        Document data
    test_corpus : string vector
        Test document data
    stoplist_filename : string
        Stoplist filename

    Returns
    -------
    [type]
        [description]
    """

    train_corpus_iterator = cc.mallet.util.StringClassArrayIterator(train_corpus)
    util = cc.mallet.util.LDAUtils()
    pipe = util.buildSerialPipe(stoplist_filename, None, True)
    instances = cc.mallet.types.InstanceList(pipe)
    instances.addThruPipe(train_corpus_iterator)

    if test_corpus is not None:
        train_corpus_iterator = cc.mallet.util.StringClassArrayIterator(test_corpus)
        vocabulary = instances.getAlphabet()
        test_pipe = util.buildSerialPipe(stoplist_filename, vocabulary, True)
        test_instances = cc.mallet.types.InstanceList(pipe)
        test_instances.addThruPipe(train_corpus_iterator)
        return list(train=instances, test=test_instances)

    return instances

def load_lda_dataset(filename, config):
    """Loads an LDA dataset from file.

        The file should be in LDA format i.e.:
            <unique id> <tab> <doc-class> <tab> <document content> <nl>
        The document class is not used in by the LDA sampler. Each line
        must be concluded with a newline so all other newlines in the
        document must be removed. The document content CAN have \\t in it.

    Parameters
    ----------
    filename : str
        filename of dataset
    config : cc.mallet.configuration.SimpleLDAConfiguration
        LDA config object

    Returns
    -------
    list of list of str
        The loaded dataset
    """

    ds = cc.mallet.util.LDAUtils.loadDataset(config, filename)

    return ds

def create_lda_sampler_of_type_with_factory(config, sampler_type):
    """Creates sampler with factory

    Parameters
    ----------
    config : LDAConfiguration
        The sampler's configuration
    sampler_type : str
        Sampler's Java type

    Returns
    -------
    GibbsLDASampler
        The new sampler.
    """
    factory = cc.mallet.configuration.ModelFactory
    return factory.get(config, JString(sampler_type))

def create_lda_sampler_of_type(config, sampler_type):
    """Creates sample of given type using constructor ctor(LDAConfiguration)

    Parameters
    ----------
    config : LDAConfiguration
        The sampler's configuration
    model_class_name : str
        Sampler's Java type

    Returns
    -------
    GibbsLDASampler
        The new sampler.
    """
    sampler_class = java.lang.Class.forName(sampler_type)

    constructor = sampler_class \
        .getConstructor(cc.mallet.configuration.LDAConfiguration)

    sampler = constructor.newInstance(config)

    return sampler

create_lda_sampler = create_lda_sampler_of_type

def load_lda_sampler(config, stored_dir="stored_samplers"):
    """Load a previously saved LDA sampler from disk.

    Parameters
    ----------
    config : cc.mallet.configuration.SimpleLDAConfiguration
        LDA config object
    store_dir : str, optional
        directory name containing stored sampler, by default "stored_samplers"

    Returns
    -------
    LDASampler
        LDA sampler object

    """

    # Load the stored sampler
    stored_lda_sampler = cc.mallet.util.LDAUtils.loadStoredSampler(config, stored_dir)

    if stored_lda_sampler is None:
        raise FileNotFoundError()

    sampler_type = stored_lda_sampler.getClass().getName()
    sampler = create_lda_sampler_of_type(config, sampler_type)

    # Init the new sampler from the stored sampler
    sampler.initFrom(stored_lda_sampler) # cast to "cc.mallet.topics.LDAGibbsSampler"

    return sampler

def save_lda_sampler(sampler, config):
    """Saves an LDA sampler to disk

    Parameters
    ----------
    sampler : GibbsLDASampler
        The sampler to save
    config : LDAConfiguration
        Sampler's configuration
    """
    default_folder = config_default.STORED_SAMPLER_DIR_DEFAULT
    sampler_folder = config.getSavedSamplerDirectory(default_folder)
    cc.mallet.util.LDAUtils().saveSampler(sampler, config, sampler_folder)

def sample_pclda(config, dataset, iterations=2000, sampler_type="cc.mallet.topics.PolyaUrnSpaliasLDA", testset=None, save_sampler=True):
    """Run the PCLDA (default Polya Urn) sampler

    Parameters
    ----------
    config : [type]
        LDA config object
    dataset : [type]
        LDA dataset
    iterations : int, optional
        number of iterations to run, by default 2000
    samplerType : str, optional
        Java class of the sampler. Must implement the LDASampler interface, by default "cc.mallet.topics.PolyaUrnSpaliasLDA"
    testset : [type], optional
        If give, the left-to-right held out log likelihood will be calculated on this dataset, by default None
    save_sampler : bool, optional
        indicates that the sampler should be saved to file after finishing, by default True

    Returns
    -------
    LDASampler
        LDA sampler object
    """
    sampler = create_lda_sampler_of_type(config, sampler_type)
    casted_sampler = jpype.JObject(sampler, cc.mallet.topics.LDAGibbsSampler) # enable access to `sample` function

    sampler.addInstances(dataset)

    if testset is not None:
        sampler.addTestInstances(testset)

    casted_sampler.sample(java.lang.Integer(iterations))

    if save_sampler:
        save_lda_sampler(sampler, config)

    return sampler

def sample_pclda_continue(sampler,  iterations = 2000):
    """Continue sampling using a trained sampler

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler
    iterations : int, optional
        Number of sample iterations, default 2000

    Returns
    -------
    [type]
        The loaded sampler
    """
    sampler.sample(java.lang.Integer(iterations))
    return sampler

def extract_vocabulary(alphabet):
    """[summary] Extract the vocabulary as a string vector from an MALLET Alphabet

    Parameters
    ----------
    alphabet : Alphabet
        Java MALLET Alphabet object, obtained by 'get_alphabet(sampler)'

    Returns
    -------
    [type]
        Vocabulary
    """
    return cc.mallet.util.LDAUtils().extractVocabulaty(alphabet)

def extract_token_counts(dataset):
    """Extract how many times each token occurs. Same order as the vocabulary

    Parameters
    ----------
    dataset : [type]
        The dataset to query

    Returns
    -------
    "[I" list of integers
        token counts
    """
    return cc.mallet.util.LDAUtils.extractTermCounts(dataset)

def extract_doc_lengths(dataset):
    """Extract the the number of tokens in each document.
        Same order as the documents

    Parameters
    ----------
    dataset : [type]
        the dataset to query

    Returns
    -------
    "[I"
        Array of integers, number of tokens in each document
    """
    return cc.mallet.util.LDAUtils.extractDocLength(dataset)

def get_alphabet(sampler):
    """Get the Alphabet (as a java object) from an LDA sampler

    Parameters
    ----------
    sampler : [type]
        the dataset to query

    Returns
    -------
        Lcc/mallet/types/Alphabet
    """
    return sampler.getAlphabet()

def get_token_topic_matrix(sampler):
    """Extracts the token/topic matrix from and LDA sampler

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler

    Returns
    -------
     m x n integer matrix, where m is #tokens and n is #topics
        and (i,j) is count of token i sampled to topic j
     """
    ttm = sampler.getTypeTopicMatrix()
    return ttm

def get_document_topic_matrix(sampler):
    """Extracts document/topic matrix (counts)

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler
    Returns
    -------
    m x n integer matrix, where m is #tokens and n is #topics
        and (i,j) is count of topic j sampled to doc i   """

    dtm = sampler.getDocumentTopicMatrix()

    return dtm

def get_topic_token_phi_matrix(sampler):
    """Get the word/topic distribution (phi matrix) from an LDA sampler

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler

    Returns
    -------
    [[D
        Phi matrix
    """
    phi = sampler.getPhi()
    return phi

def calculate_token_probs(type_topic_counts, beta):
    """Computes token's overall probability

    Parameters
    ----------
     type_topic_counts : int[][]
        LDA type/topic counts
    beta : double
        beta value

    Returns
    -------
     m double array, where m is #tokens
        and (i) is word i's probability
    """
    n_topics = len(type_topic_counts[0])
    n_tokens = len(type_topic_counts)

    total_mass = np.sum(type_topic_counts, dtype=np.float64) + (n_topics * n_tokens) * beta
    token_total_mass = np.sum(type_topic_counts, axis=1, dtype=np.float64) + n_topics * beta

    token_probs = token_total_mass / total_mass

    return token_probs

def calculate_token_probs_given_topic(type_topic_counts, beta):
    """Compute token's probability given topic.

    This is a port of LDAUtil.calcWordProbGivenTopic()

    Parameters
    ----------
    type_topic_counts : int[][]
        LDA type/topic counts
    beta : double
        beta value

    Returns
    -------
     m x n double matrix, where m is #tokens and n is #topics
        and (i, j) is word i's probability given topic j
    """
    n_topics = len(type_topic_counts[0])
    n_tokens = len(type_topic_counts)

    topic_mass = np.sum(type_topic_counts, axis=0, dtype=np.float64) + n_tokens * beta

    p_w_k = (np.array(type_topic_counts, dtype=np.float64) + beta) / \
                    topic_mass[np.newaxis, :]

    return p_w_k

def calculate_token_relevance_matrix(type_topic_counts_matrix, beta, _lambda):
    """Calculates token relevances matix.

    This is a port of cc.mallet.util.LDAUtils.getTopRelevanceWords

    Parameters
    ----------
    type_topic_counts : int[][]
        LDA type/topic counts
    beta : double
        beta value used by sampler
    _lambda : double
        lambda value used in calculation

    Returns
    -------
    [type]
        [description]
    """

    n_topics = len(type_topic_counts_matrix[0])
    n_types  = len(type_topic_counts_matrix)

    p_w_k    = calculate_token_probs_given_topic(type_topic_counts_matrix, beta)
    p_w      = calculate_token_probs(type_topic_counts_matrix , beta)

    relevance_matrix = [
        _lambda * np.log(p_w_k[:,topic]) + (1 - _lambda) * (np.log(p_w_k[:,topic]) - np.log(p_w))
            for topic in range(0, n_topics)
    ]

    return relevance_matrix

def calculate_top_relevance_tokens(n_top_tokens, type_topic_counts_matrix, beta, _lambda, alphabet):

    n_topics = len(type_topic_counts_matrix[0])
    n_types  = len(type_topic_counts_matrix)
    n_top_tokens = min(n_top_tokens, n_types)

    relevance_matrix = np.array(
        calculate_token_relevance_matrix(type_topic_counts_matrix, beta, _lambda)
    )

    sorted_indices = relevance_matrix.argsort(axis=1)

    sorted_matrix = relevance_matrix[
        np.arange(np.shape(relevance_matrix)[0])[:,np.newaxis],
        sorted_indices
    ]

    sliced_indices = np.flip(sorted_indices[:,-n_top_tokens:], axis=1)
    sliced_matrix = np.flip(sorted_matrix[:,-n_top_tokens:], axis=1)

    descending_token_relevances = (
        (alphabet.lookupObject(w[1]), w[0])
            for w in zip(
                    sliced_matrix.ravel(),
                    sliced_indices.ravel()
                )
    )
    return [
        [ w for w in row ]
            for row in toolz.partition(n_top_tokens, descending_token_relevances)
    ]

def get_top_topic_tokens(sampler, n_words=20):
    """Get the top words per topic from an LDA sampler

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler
    n_words : int, optional
        Number of top words per topic to return, default 20

    Returns
    -------
    "[[Ljava/lang/String;"
        Per topic top words matrix
    """
    alphabet = sampler.getAlphabet()
    word_topic_matrix = sampler.getTypeTopicMatrix()
    alphabet_size = alphabet.size()
    n_topics = sampler.getNoTopics()

    top_topic_words = cc.mallet.util.LDAUtils.getTopWords(
                jint(n_words),
                jint(alphabet_size),
                jint(n_topics),
                word_topic_matrix, #dispatch = T),
                alphabet)

    return top_topic_words

def get_theta_estimate(sampler):
    """Get an estimate of the document topic distribution

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler

    Returns
    -------
    "[[D"
        The theta matrix) from an LDA sampler
    """
    theta = sampler.getThetaEstimate(sampler) # ,simplify = TRUE)
    return theta

def get_z_means(sampler):
    """Get the mean of the topic indicators from an LDA sampler

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler

    Returns
    -------
    "[[D"
    """
    zb = sampler.getZbar(sampler) # ,simplify = TRUE)
    return zb

def get_top_topic_word_relevances(sampler, config, n_top_words=20, v_lambda=None):
    """Get the 'top relevance words' (weighted version of top words) per topic from an LDA sampler

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler
    config : LDAConfiguration
        LDA config object
    n_words : int, optional
        Number of (top) words per topic to retrieve, default 20
    lambda_value : float, optional
        Lambda value to use when calculating the relevance, default 0.6
    """

    relevance_words = cc.mallet.util.LDAUtils.getTopRelevanceWords(
        jint(n_top_words),
        sampler.getAlphabet().size(),
        sampler.getNoTopics(),
        sampler.getTypeTopicMatrix(),
        config.getBeta(config_default.BETA_DEFAULT),
        jdbl(v_lambda) if v_lambda is not None else config.getLambda(config_default.LAMBDA_DEFAULT),
        sampler.getAlphabet()
    )
    return relevance_words

def calculate_type_topic_matrix_density(type_topic_matrix):
    """Calculate the density (sparsity) of the type topic matrix

    Parameters
    ----------
    type_topic_matrix : [type]
        Type topic matrix, obtained by 'get_type_topics(sampler)'

    Returns
    -------
    [type]
        [description]
    """
    return cc.mallet.util.LDAUtils.calculateMatrixDensity(type_topic_matrix) # ,dispatch = T)

def get_log_likelihood(sampler):
    """Extract the log likelihood for each iteration

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler, the trained LDA sampler

    Returns
    -------
    Double
        Log likelihood
    """
    log_likelihood = sampler.getLogLikelihood(simplify=True)
    return log_likelihood

def get_held_out_log_likelihood(sampler):
    """Extracts the heldout log likelihood for each iteration.
    The sampler must have been run with a test set

    Parameters
    ----------
    sampler : LDASampler
        LDA sampler, the trained LDA sampler

    Returns
    -------
    Double
        Log likelihood for test set
    """
    ll = sampler.getHeldOutLogLikelihood(simplify=True)
    return ll

def get_buid_version():
    """Returns build information (build, version, commit)

    Returns
    -------
        SimpleNamespace instance
    """
    return types.SimpleNamespace(
        build=cc.mallet.util.getManifestInfo("Implementation-Build","PCPLDA"),
        version=cc.mallet.util.getManifestInfo("Implementation-Version","PCPLDA"),
        commit=cc.mallet.util.getLatestCommit()
    )

def print_top_words(word_matrix):
    """Print the 'top words' from a sampled word matrix obtained using 'get_topwords(sampler)'

    Parameters
    ----------
    word_matrix : [type]
        top word matrix

    Returns
    -------
    [type]
        [description]
    """
    util = cc.mallet.util.LDAUtils

    # Vad betder $? Och är dispatch = t lik med transpose
    return None # util.formatTopWords(.jarray(word_matrix,dispatch = T))

