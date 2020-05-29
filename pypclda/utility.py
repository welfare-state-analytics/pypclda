import os
import datetime
import logging
import numpy as np
import toolz

ADLDA_MODEL = "adlda"
UNCOLLAPSED_MODEL = "uncollapsed"
COLLAPSED_MODEL = "collapsed"
LIGHT_COLLAPSED_MODEL = "lightcollapsed"
EFFICIENT_UNCOLLAPSED_MODEL = "efficient_uncollapsed"
SPALIAS_MODEL = "spalias"
POLYAURN_MODEL =  "polyaurn"
POLYAURN_PRIORS_MODEL =  "polyaurn_priors"
PPU_HLDA_MODEL =  "ppu_hlda"
PPU_HDPLDA_MODEL =  "ppu_hdplda"
PPU_HDP_ALL_TOPICS_MODEL =  "ppu_hdplda_all_topics"
SPALIAS_PRIORS_MODEL =  "spalias_priors"
LIGHTPCLDA_MODEL =  "lightpclda"
LIGHTPCLDA_PROPOSAL_MODEL =  "lightpclda_proposal"
NZVSSPALIAS_MODEL =  "nzvsspalias"
DEFAULT_MODEL =  POLYAURN_MODEL

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_timestamp2():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_CLASS_NAME_MAP = {
    ADLDA_MODEL: "cc.mallet.topics.ADLDA",
    UNCOLLAPSED_MODEL: "cc.mallet.topics.UncollapsedParallelLDA",
    COLLAPSED_MODEL: "cc.mallet.topics.SerialCollapsedLDA",
    LIGHT_COLLAPSED_MODEL: "cc.mallet.topics.CollapsedLightLDA",
    EFFICIENT_UNCOLLAPSED_MODEL: "cc.mallet.topics.EfficientUncollapsedParallelLDA",
    SPALIAS_MODEL: "cc.mallet.topics.SpaliasUncollapsedParallelLDA",
    POLYAURN_MODEL: "cc.mallet.topics.PolyaUrnSpaliasLDA",
    POLYAURN_PRIORS_MODEL: "cc.mallet.topics.PolyaUrnSpaliasLDAWithPriors",
    PPU_HDP_ALL_TOPICS_MODEL: "cc.mallet.topics.PoissonPolyaUrnHDPLDAInfiniteTopics",
    SPALIAS_PRIORS_MODEL: "cc.mallet.topics.SpaliasUncollapsedParallelWithPriors",
    LIGHTPCLDA_MODEL: "cc.mallet.topics.LightPCLDA",
    LIGHTPCLDA_PROPOSAL_MODEL: "cc.mallet.topics.LightPCLDAtypeTopicProposal",
    NZVSSPALIAS_MODEL: "cc.mallet.topics.NZVSSpaliasUncollapsedParallelLDA"
}


def options2args(options):

    args = [ f"--{key}={value}" for key, value in options ]
    return args

    # "dbg",    "debug",              true,  "use debugging " );
    # "c",      "continue",           false, "continue sampling from a saved state (give in config file by option 'saved_sampler_dir') " );
    # "cm",     "comment",            true,  "a comment to be added to the logfile " );
    # "ds",     "dataset",            true,  "filename of dataset file" );
    # "ts",     "topics",             true,  "number of topics" );
    # "a",      "alpha",              true,  "uniform alpha prior" );
    # "b",      "beta",               true,  "uniform beta prior" );
    # "i",      "iterations",         true,  "number of sample iterations" );
    # "batch",  "batches",            true,  "the number of batches to split the data in" );
    # "r",      "rare_threshold",     true,  "the number of batches to split the data in" );
    # "ti",     "topic_interval",     true,  "topic interval" );
    # "sd",     "start_diagnostic",   true,  "start diagnostic" );
    # "sch",    "scheme",             true,  "sampling scheme " );
    # "cf",     "run_cfg",            true,  "full path to the RunConfiguration file " );

def get_logger(level=logging.DEBUG, log_folder=None, prefix="", suffix="console_output"):

    # PrintStream logOut = new PrintStream(new FileOutputStream(logFile, true));
    # PrintStream teeStdOut = new TeeStream(System.out, logOut);
    # PrintStream teeStdErr = new TeeStream(System.err, logOut);

    # System.setOut(teeStdOut);
    # System.setErr(teeStdErr);

    logger = logging.getLogger('pypclda')
    logger.setLevel(level)

    if log_folder is not None:
        log_file = os.path.join(log_folder, f"{prefix}_{suffix}.log")

        logger.info("FIXME: set log file handler")

    return logger


def extract_top_tokens_descending(matrix, n_top_tokens, alphabet):

    sorted_indices = matrix.argsort(axis=1)

    sorted_matrix = matrix[
        np.arange(np.shape(matrix)[0])[:,np.newaxis],
        sorted_indices
    ]

    n_top_tokens = min(n_top_tokens, len(matrix[0]))

    sliced_indices = np.flip(sorted_indices[:,-n_top_tokens:], axis=1)
    sliced_matrix = np.flip(sorted_matrix[:,-n_top_tokens:], axis=1)

    zipped_tokens = (
        (str(alphabet.lookupObject(w[1])), w[0])
            for w in zip(
                    sliced_matrix.ravel(),
                    sliced_indices.ravel()
                )
    )
    return [ [ w for w in row ] for row in toolz.partition(n_top_tokens, zipped_tokens)]

