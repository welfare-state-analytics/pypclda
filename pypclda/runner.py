import os
import warnings
import pytest
import datetime
import time
import logging
import types
import jpype
import jpype.imports
from jpype.types import *

import pypclda
import pypclda.config_default as config_default
import pypclda.utility as utility

def print_build_info():
    pass

print_build_info()

logger = utility.getLogger()
lda_utils = pypclda.get_util()

def sample(is_continuation, **options):

    lda_config = pypclda.create_simple_lda_config(**options)

    # installAbortHandler()
    # This must probably be implemented:
    #exHandler = buildExceptionHandler();
    #Thread.setDefaultUncaughtExceptionHandler(exHandler);

    run_config_names = lda_config.getSubConfigs()

    for run_config_name in run_config_names:

        lda_config.getsetLoggingUtil().checkCreateAndSetSubLogDir(run_config_name)
        lda_config.activateSubconfig(run_config_name)

        sample_config(is_continuation, lda_config, run_config_name)

def sample_config(is_continuation, lda_config, sub_config_name):

    log_folder = lda_config.getsetLoggingUtil().getLogDir().getAbsolutePath()
    logger = utility.get_logger(log_folder=log_folder, prefix=sub_config_name)

    model_name = lda_config.getScheme() or config_default.DEFAULT_MODEL
    model_class_name = utility.MODEL_CLASS_NAME_MAP.get(model_name, None)

    logger.info(f"Topic count: {lda_config.getNoTopics(-1)}")
    logger.info(f"Using SCHEME: {model_name}")
    logger.info(f"Using TRAIN dataset: {lda_config.getDatasetFilename()}")
    logger.info(f"Using TEST dataset: {lda_config.getTestDatasetFilename() or None}")
    logger.info(f"Using CONFIG: {lda_config.whereAmI()}")
    logger.info(f"Using Sub-config: {sub_config_name}")

    if is_continuation:

        logger.info("Continuing sampling from previously stored model...");

        stored_dir = lda_config.getSavedSamplerDirectory(config_default.STORED_SAMPLER_DIR_DEFAULT);
        lda_sampler = pypclda.load_lda_sampler(lda_config, stored_dir=stored_dir);
        instances = lda_sampler.getDataset();

    else:

        lda_sampler = pypclda.create_lda_sampler(lda_config, model_class_name)
        instances = lda_utils.loadDataset(lda_config, lda_config.getDatasetFilename());
        instances.getAlphabet().stopGrowth();

        # Imports the data into the model
        logger.info("Loading data instances...");
        lda_sampler.addInstances(instances);

        if lda_config.getTestDatasetFilename() != None:
            test_instances = lda_utils.loadDataset(lda_config, lda_config.getTestDatasetFilename(), instances.getAlphabet())
            lda_sampler.addTestInstances(test_instances)

    if lda_sampler is None:
        logger.error(f"Aborting, unknown model {model_name}...")
        exit(-1)

    # iterListener = createIterationListener(config); # IterationListener
    # if model instanceof LDASamplerWithCallback and iterListener != None:
    #     logger.info("Setting callback...");
    #     ((LDASamplerWithCallback)model).setIterationCallback(iterListener);

    lda_sampler.setRandomSeed(lda_config.getSeed(0));
    #lda_sampler.setShowTopicsInterval(lda_config.getTopicInterval(config_default.TOPIC_INTER_DEFAULT));

    if lda_config.getTfIdfVocabSize(config_default.TF_IDF_VOCAB_SIZE_DEFAULT) > 0:
        threshold = lda_config.getTfIdfVocabSize(config_default.TF_IDF_VOCAB_SIZE_DEFAULT)
        logger.info(f"Top TF-IDF threshold: {threshold}")
    else:
        threshold = lda_config.getRareThreshold(config_default.RARE_WORD_THRESHOLD)
        logger.info(f"Rare word threshold: {threshold}")

    logger.info("Vocabulary size: " + instances.getDataAlphabet().size());
    logger.info("Instance list is: " + instances.size());
    logger.info("Dataset size: " + lda_sampler.getDataset().size())
    logger.info("Total #tokens: " + lda_sampler.getCorpusSize());
    logger.info("Config seed:" + lda_config.getSeed(config_default.SEED_DEFAULT))
    logger.info("Start seed: " + lda_sampler.getStartSeed())
    logger.info("Starting iterations (" + lda_config.getNoIterations(config_default.NO_ITER_DEFAULT) + " total).");

    # Runs the model
    start = time.time()
    logger.info(f"Starting: {start}")

    lda_sampler.sample(lda_config.getNoIterations(config_default.NO_ITER_DEFAULT));

    end = time.time()
    logger.info(f"Finished: {end}")

    #logger.info("Topic model diagnostics:");
    #logger.info(tmd.toString());

    if lda_config.saveSampler(False):
        samplerFolder = lda_config.getSavedSamplerDirectory(config_default.STORED_SAMPLER_DIR_DEFAULT)
        lda_utils.saveSampler(lda_sampler, lda_config, samplerFolder)

    if lda_config.saveDocumentTopicMeans():
        docTopicMeanFn = lda_config.getDocumentTopicMeansOutputFilename();
        means = lda_sampler.getZbar(); #double [][]
        lda_utils.writeASCIIDoubleMatrix(means, log_folder + "/" + docTopicMeanFn, ",");

    if lda_config.saveDocumentThetaEstimate():
        docTopicThetaFn = lda_config.getDocumentTopicThetaOutputFilename();
        means = lda_sampler.getThetaEstimate(); #double [][]
        lda_utils.writeASCIIDoubleMatrix(means, log_folder + "/" + docTopicThetaFn, ",");

    if lda_config.saveVocabulary(False):
        lda_utils.writeStringArray(
            lda_utils.extractVocabulaty(instances.getDataAlphabet()),
            lda_config.getVocabularyFilename()
        )

    if lda_config.saveCorpus(False):
        lda_utils.writeASCIIIntMatrix(
            lda_utils.extractCorpus(instances),
            os.path.join(log_folder, lda_config.getCorpusFilename()),
            ","
        )

    if lda_config.saveTermFrequencies(False):
        lda_utils.writeIntArray(
            lda_utils.extractTermCounts(instances),
            os.path.join(log_folder, lda_config.getTermFrequencyFilename())
        )

    if lda_config.saveDocLengths(False):
        lda_utils.writeIntArray(
            lda_utils.extractDocLength(instances),
            os.path.join(log_folder, lda_config.getDocLengthsFilename())
        )

    n_top_words = lda_config.getNrTopWords(config_default.NO_TOP_WORDS_DEFAULT)
    n_top_words = min(n_top_words, instances.getDataAlphabet().size())

    # if instanceof(model, LDASamplerWithPhi): # model instanceof LDASamplerWithPhi
    if lda_config.savePhiMeans(config_default.SAVE_PHI_MEAN_DEFAULT):

        means = lda_sampler.getPhiMeans() # double [][]
        if means is not None:

            lda_utils.writeASCIIDoubleMatrix(
                means,
                os.path.join(log_folder, lda_config.getPhiMeansOutputFilename()),
                ","
            )

            lda_utils.writeStringArray(
                lda_utils.extractVocabulaty(instances.getDataAlphabet()),
                os.path.join(log_folder, lda_config.getVocabularyFilename() or "phi_vocabulary.txt")
            )

        else:

            logger.error("WARNING: ParallelLDA: No Phi means where sampled, not saving Phi means! This is likely due to a combination of configuration settings of phi_mean_burnin, phi_mean_thin and save_phi_mean");

    print_top_words(lda_config, lda_sampler, log_folder)
    print_relevance_words(log_folder, lda_sampler, lda_config)
    print_salient_words(log_folder, lda_sampler, lda_config)
    print_kri_re_weighted_words(log_folder, lda_sampler, lda_config)

    # metadata = new ArrayList<String>(); # List<String>
    # metadata.add("No. Topics: " + model.getNoTopics());
    # metadata.add("Start Seed: " + model.getStartSeed());
    # # Save stats for this run
    # lu.dynamicLogRun("Runs", t, cp, config, None,
    #         ParallelLDA.class.getName(), "Convergence", "HEADING", "PLDA", 1, metadata)

    # if instanceof(model, HDPSamplerWithPhi):
    #     printHDPRsampler_class_folder);

    # if lda_config.saveDocumentTopicDiagnostics():
    #     tmd = new TopicModelDiagnosticsPlain(model, requestedWords)
    #     filename = config.getDocumentTopicDiagnosticsOutputFilename()
    #     out = new PrintWriter(log_folder + "/" + docTopicDiagFn)
    #     print(tmd.topicsToCsv())

    logger.info(f"{datetime.datetime.now()}: I am done!");

def print_top_words(lda_config, lda_sampler, log_folder, filename="top_topic_words.txt"):
    top_words = pypclda.get_top_topic_words(lda_sampler, lda_config, n_words=20, lambda_value=0.6)
    pass
    # PrintWriter out = new PrintWriter(log_folder + "/TopWords.txt");
    # out.println(lda_utils.formatTopWordsAsCsv(
    #         lda_utils.getTopWords(requestedWords,
    #                 model.getAlphabet().size(),
    #                 model.getNoTopics(),
    #                 model.getTypeTopicMatrix(),
    #                 model.getAlphabet())));
    # logger.info("Top words are: \n" +
    #         lda_utils.formatTopWords(lda_utils.getTopWords(requestedWords,
    #                 model.getAlphabet().size(),
    #                 model.getNoTopics(),
    #                 model.getTypeTopicMatrix(),
    #                 model.getAlphabet())));

def print_relevance_words(log_folder, model, config):
    pass
    # out = new PrintWriter(log_folder + "/RelevanceWords.txt");
    # out.println(lda_utils.formatTopWordsAsCsv(
    #         lda_utils.getTopRelevanceWords(requestedWords,
    #                 model.getAlphabet().size(),
    #                 model.getNoTopics(),
    #                 model.getTypeTopicMatrix(),
    #                 config.getBeta(config_default.BETA_DEFAULT),
    #                 config.getLambda(config_default.LAMBDA_DEFAULT),
    #                 model.getAlphabet())));
    # logger.info("Relevance words are: \n" +
    #     lda_utils.formatTopWords(lda_utils.getTopRelevanceWords(requestedWords,
    #             model.getAlphabet().size(),
    #             model.getNoTopics(),
    #             model.getTypeTopicMatrix(),
    #             config.getBeta(config_default.BETA_DEFAULT),
    #             config.getLambda(config_default.LAMBDA_DEFAULT),
    #             model.getAlphabet())));

def print_salient_words(log_folder, model, config):
    pass
    #				logger.info("Salient words are: \n" +
    #						lda_utils.formatTopWords(lda_utils.getTopSalientWords(20,
    #								model.getAlphabet().size(),
    #								model.getNoTopics(),
    #								model.getTypeTopicMatrix(),
    #								config.getBeta(config_default.BETA_DEFAULT),
    #								model.getAlphabet())));

def print_kri_re_weighted_words(log_folder, model, config):
    pass
    #				logger.info("KR1 re-weighted words are: \n" +
    #						lda_utils.formatTopWords(lda_utils.getK1ReWeightedWords(20,
    #								model.getAlphabet().size(),
    #								model.getNoTopics(),
    #								model.getTypeTopicMatrix(),
    #								config.getBeta(config_default.BETA_DEFAULT),
    #								model.getAlphabet())));
