import os
import pypclda.utility as utility
import jpype
import jpype.imports
from jpype.types import *

logger = utility.get_logger()

PCPLDA_JAR_PATH = os.path.join(os.getcwd(), "lib", "PCPLDA-8.5.1.jar")

jpype.startJVM(classpath=[PCPLDA_JAR_PATH], convertStrings=False)

cc = jpype.JPackage("cc")
java = jpype.JPackage("java")


def get_util():
    util = cc.mallet.util.LDAUtils()
    return util

def get_logging_utils(output_path=None):
    logging_util = cc.mallet.util.LoggingUtils()
    if not output_path is None:
        logging_util.checkAndCreateCurrentLogDir(JString(output_path))
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
        tmpdir string
            temporary directory for intermediate storage of logging data (default "tmp")
        topic_priors string
            text file with 'prior spec' with one topic per line with format: <topic nr(zero idxed)>, <word1>, <word2>, etc

    Returns
    -------
    cc.mallet.configuration.SimpleLDAConfiguration
        [description]

    """

    timestamp = utility.get_timestamp2()
    output_path = os.path.join("output", f"run_{timestamp}", "")

    slc = cc.mallet.configuration.SimpleLDAConfiguration()

    slc.setLoggingUtil(get_logging_utils(output_path))

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

    if "experiment_output_directory" in kwargs:
        slc.setExperimentOutputDirectory(kwargs.get("experiment_output_directory", "output"))

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
        slc.setSavedSamplerDirectory("stored_sampler")

    if "save_term_frequencies" in kwargs:
        slc.setSaveTermFrequencies(kwargs["save_term_frequencies"])

    logger.info("Logging to: " + output_path)

    return slc

def create_lda_dataset(train_corpus, test_corpus=None, stoplist_filename="stoplist.txt"):
    """ Create an LDA dataset from existing string vector.

        Each entry in the vector must be a string with the following format:
            <unique id>\\t<doc class>\\t<document content>
        The document class is not used in by the LDA sampler.
        The document content CAN have \\t in it.

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

def load_lda_dataset(filename, lda_config):
    """Loads an LDA dataset from file.

        The file should be in LDA format i.e.:
            <unique id> <tab> <doc-class> <tab> <document content> <nl>
        The document class is not used in by the LDA sampler. Each line
        must be concluded with a newline so all other newlines in the
        document must be removed. The document content CAN have \\t in it.

    Parameters
    ----------
    filename : [type]
        filename of dataset
    lda_config : cc.mallet.configuration.SimpleLDAConfiguration
        LDA config object

    Returns
    -------
    list of list of str
        [description]
    """

    util = get_util()

    ds = cc.mallet.util.LDAUtils.loadDataset(lda_config, filename)

    return ds

def create_lda_sampler_of_type_with_factory(lda_config, model_class_name):
    factory = cc.mallet.configuration.ModelFactory
    return factory.get(lda_config, JString(model_class_name))

def create_lda_sampler_of_type(lda_config, sampler_type):

    lda_sampler_class = java.lang.Class.forName(sampler_type)

    constructor = lda_sampler_class \
        .getConstructor(cc.mallet.configuration.LDAConfiguration)

    lda_sampler = constructor.newInstance(lda_config)

    gibbs_sampler = jpype.JObject(lda_sampler, cc.mallet.topics.LDAGibbsSampler)

    return gibbs_sampler

create_lda_sampler = create_lda_sampler_of_type

def load_lda_sampler(lda_config, stored_dir="stored_samplers"):
    """Load an LDA sampler from file.

    Parameters
    ----------
    lda_config : cc.mallet.configuration.SimpleLDAConfiguration
        LDA config object
    store_dir : str, optional
        directory name containing stored sampler, by default "stored_samplers"

    Returns
    -------
    LDASampler
        LDA sampler object

    Raises
    ------
    FileNotFoundError
        [description]
    """

    # Load the stored sampler
    stored_lda_sampler = get_util().loadStoredSampler(lda_config, stored_dir)

    if stored_lda_sampler is None:
        raise FileNotFoundError()

    sampler_type = stored_lda_sampler.getClass().getName()
    lda = create_lda_sampler_of_type(sampler_type)

    # Init the new sampler from the stored sampler
    lda.initFrom(stored_lda_sampler) # cast to "cc.mallet.topics.LDAGibbsSampler"

    return lda

def sample_pclda(lda_config, ds, iterations=2000, sampler_type="cc.mallet.topics.PolyaUrnSpaliasLDA", testset=None, save_sampler=True):
    """Run the PCLDA (default Polya Urn) sampler

    Parameters
    ----------
    lda_config : [type]
        LDA config object
    ds : [type]
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
    lda_sampler = create_lda_sampler_of_type(lda_config, sampler_type)

    lda_sampler.addInstances(ds)

    if testset is not None:
        lda_sampler.addTestInstances(testset)

    lda_sampler.sample(java.lang.Integer(iterations))

    if save_sampler:
        sampler_dir = "stored_samplers" # J("cc.mallet.configuration.LDAConfiguration")$STORED_SAMPLER_DIR_DEFAULT
        samplerFolder = lda_config.getSavedSamplerDirectory(sampler_dir)
        cc.mallet.util.LDAUtils().saveSampler(lda_sampler, lda_config, samplerFolder)

    return lda_sampler

def sample_pclda_continue(lda,  iterations = 2000):
    """Continue sampling using a trained sampler

    Parameters
    ----------
    lda : LDASampler
        LDA sampler
    iterations : int, optional
        [description]how many iterations to sample, by default 2000

    Returns
    -------
    [type]
        [description]
    """
    lda.sample(java.lang.Integer(iterations))
    return lda

def print_top_words(word_matrix):
    """Print the 'top words' from a sampled word matrix obtained using 'get_topwords(lda)'

    Parameters
    ----------
    word_matrix : [type]
        top word matrix

    Returns
    -------
    [type]
        [description]
    """
    util = get_util()

    # Vad betder $? Och är dispatch = t lik med transpose
    return None # util.formatTopWords(.jarray(word_matrix,dispatch = T))

def extract_vocabulary(alphabet):
    """[summary] Extract the vocabulary as a string vector from an MALLET Alphabet

    Parameters
    ----------
    alphabet : Alphabet
        Java MALLET Alphabet object, obtained by 'get_alphabet(lda)'

    Returns
    -------
    [type]
        Vocabulary
    """
    util = get_util()
    return util.extractVocabulat(alphabet)

def extract_term_counts(instances):
    """Extract how many times each word occurs. Same order as the vocabulary

    Parameters
    ----------
    instances : [type]
        The dataset to query

    Returns
    -------
    "[I"
        Word counts
    """
    util = get_util()
    return util.extractTermCounts(instances)

def extract_doc_lengths(instances):
    """Extract the the number of tokens in each document. Same order as the documents

    Parameters
    ----------
    instances : [type]
        the dataset to query

    Returns
    -------
    "[I"
        Array of integers, number of tokens in each document
    """
    util = get_util()
    return util.extractDocLength(instances)

def get_alphabet(lda):
    """Get the Alphabet (as a java object) from an LDA sampler

    Parameters
    ----------
    lda : [type]
        the dataset to query

    Returns
    -------
        Lcc/mallet/types/Alphabet
    """
    return lda.getAlphabet()

def get_theta_estimate(lda):
    """Get an estimate of the document topic distribution

    Parameters
    ----------
    lda : LDASampler
        LDA sampler

    Returns
    -------
    "[[D"
        The theta matrix) from an LDA sampler
    """
    theta = lda.getThetaEstimate(lda) # ,simplify = TRUE)
    return theta

def get_z_means(lda):
    """Get the mean of the topic indicators from an LDA sampler

    Parameters
    ----------
    lda : LDASampler
        LDA sampler

    Returns
    -------
    "[[D"
    """
    zb = lda.getZbar(lda) # ,simplify = TRUE)
    return zb

def get_type_topics(lda):
    """Get the type/topic matrix from and LDA sampler

    Parameters
    ----------
    lda : LDASampler
        LDA sampler

    Returns
    -------
    [[I
        the type/topic matrix
    """
    ttm = lda.getTypeTopicMatrix() # ",simplify = TRUE)
    return ttm

def get_phi(lda):
    """Get the word/topic distribution (phi matrix) from an LDA sampler

    Parameters
    ----------
    lda : LDASampler
        LDA sampler

    Returns
    -------
    [[D
        Phi matrix
    """
    phi = lda.getPhi() # ",simplify = TRUE)
    return phi

def get_topwords(lda,nr_words=20):
    """Get the top words per topic from an LDA sampler

    Parameters
    ----------
    lda : LDASampler
        LDA sampler
    nr_words : int, optional
        Number of top words per topic to retrieve, by default 20

    Returns
    -------
    "[[Ljava/lang/String;"
        Per topic top words matrix
    """
    util = get_util()
    alph = lda.getAlphabet()
    typeTopicMatrix = lda.getTypeTopicMatrix() # ", simplify = TRUE)
    alphSize = alph.size()
    nrTopics = lda.getNoTopics()

    tw = util.getTopWords(
                java.lang.Integer(nr_words),
                java.lang.Integer(alphSize),
                java.lang.Integer(nrTopics),
                typeTopicMatrix, #dispatch = T),
                alph) #,
                # simplify = TRUE)

    return tw

def get_top_relevance_words(lda, config, nr_words=20, lambda_value=0.6):
    """Get the 'top relevance words' (weighted version of top words) per topic from an LDA sampler

    Parameters
    ----------
    lda : LDASampler
        LDA sampler
    config : [type]
        LDA config object
    nr_words : int, optional
        Number of top words per topic to retrieve, by default 20
    lambda_value : float, optional
        Lambda value to use when calculating the relevance, by default 0.6
    """
    util = get_util()
    alph = lda.getAlphabet()
    typeTopicMatrix = lda.getTypeTopicMatrix() # ", simplify = TRUE)
    alphSize = alph.size()
    nrTopics = lda.getNoTopics()
    beta = config.getBeta(java.lang.Double(0.01)).doubleValue()

    rw = util.getTopRelevanceWords(
                java.lang.Integer(nr_words),
                java.lang.Integer(alphSize),
                java.lang.Integer(nrTopics),
                typeTopicMatrix, #dispatch = T),
                java.lang.Double(beta),
                java.lang.Double(lambda_value),
                alph) #,
                # simplify = TRUE)
    rw

def calculate_ttm_density(typeTopicMatrix):
    """Calculate the density (sparsity) of the type topic matrix

    Parameters
    ----------
    typeTopicMatrix : [type]
        Type topic matrix, obtained by 'get_type_topics(lda)'

    Returns
    -------
    [type]
        [description]
    """
    util = get_util()
    return util.calculateMatrixDensity(typeTopicMatrix) # ,dispatch = T)

def get_log_likelihood(lda):
    """]Extract the log likelihood for each iteration

    Parameters
    ----------
    lda : LDASampler
        LDA sampler, the trained lda model

    Returns
    -------
    Double
        Log likelihood
    """
    ll = lda.getLogLikelihood(simplify=True)
    return ll

def get_held_out_log_likelihood(lda):
    """Extracts the heldout log likelihood for each iteration.
    The sampler must have been run with a test set

    Parameters
    ----------
    lda : LDASampler
        LDA sampler, the trained lda model

    Returns
    -------
    Double
        Log likelihood for test set
    """
    ll = lda.getHeldOutLogLikelihood(simplify=True)
    return ll

#' run_gc
#'
#' Runs the garbage collectors in both R (first) and Java
#'
#' @param ... any arguments to gc
#'
#' @importFrom rJava J
#' @export
# run_gc <- function(...) {
#   gc(...)
#   J("java.lang.Runtime")$getRuntime()$gc()
#   invisible()
# }


def print_build_info():
    pass
    # print("We have: ?? processors avaiable")
    # buildVer = LoggingUtils.getManifestInfo("Implementation-Build","PCPLDA")
    # implVer  = LoggingUtils.getManifestInfo("Implementation-Version", "PCPLDA")
    # if buildVer==None or implVer==None:
    #     System.out.println("GIT info:" + LoggingUtils.getLatestCommit());
    # } else {
    #     System.out.println("Build info:"
    #             + "Implementation-Build = " + buildVer + ", "
    #             + "Implementation-Version = " + implVer);
