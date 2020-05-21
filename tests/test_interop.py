import os
import javabridge
import warnings
import pytest

from pypclda import create_simple_lda_config, load_lda_dataset, load_lda_sampler, create_lda_dataset, sample_pclda

PCPLDA_JAR_PATH = os.path.join(os.getcwd(), "lib", "PCPLDA-8.5.1.jar")

def test_hello_word():
    javabridge.start_vm(run_headless=True)
    try:
        result = javabridge.run_script('java.lang.String.format("Hello, %s!", greetee);', dict(greetee='world'))
        assert result == "Hello, world!"
    finally:
        javabridge.kill_vm()

def test_can_create_logging_util_instance_using_JWrapper():
    javabridge.start_vm(run_headless=True, class_path=javabridge.JARS + [PCPLDA_JAR_PATH])
    try:
        logging_util = javabridge.JWrapper(javabridge.make_instance("cc/mallet/util/LoggingUtils", "()V"))
        fp = logging_util.checkAndCreateCurrentLogDir("/tmp/hello")
        assert logging_util is not None
    except Exception as ex:
        raise ex
    finally:
        javabridge.kill_vm()

def test_can_create_logging_util_instance_using_JClassWrapper():
    javabridge.start_vm(run_headless=True, class_path=javabridge.JARS + [PCPLDA_JAR_PATH])
    try:
        logging_util = javabridge.JClassWrapper("cc.mallet.util.LoggingUtils")()
        fp = logging_util.checkAndCreateCurrentLogDir("/tmp/hello")
        assert logging_util is not None
    except Exception as ex:
        raise ex
    finally:
        javabridge.kill_vm()

def test_create_simple_lda_config():

    javabridge.start_vm(run_headless=True, class_path=javabridge.JARS + [PCPLDA_JAR_PATH])
    try:
        args = dict(
            nr_topics=99,
            alpha=0.05,
            beta=0.001
        )

        slc = create_simple_lda_config(**args)

        assert args["nr_topics"] == slc.getNoTopics()
        # assert args["alpha"] == slc.getAlpha(0.05)
        # assert args["beta"] == slc.getBeta(0.001)

    except Exception as ex:
        raise ex
    finally:
        javabridge.kill_vm()

def test_load_lda_dataset():
    javabridge.start_vm(run_headless=True, class_path=javabridge.JARS + [PCPLDA_JAR_PATH])
    try:
        filename = os.path.join(os.getcwd(), "tests", "corpus.txt")

        lda_config = create_simple_lda_config(dataset_fn=filename)

        result = load_lda_dataset(filename, lda_config)

        assert result is not None

    except Exception as ex:
        raise ex
    finally:
        javabridge.kill_vm()

def test_load_lda_sampler():
    javabridge.start_vm(run_headless=True, class_path=javabridge.JARS + [PCPLDA_JAR_PATH])
    try:
        store_dir = "stored_samplers"
        filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
        stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
        lda_config = create_simple_lda_config(dataset_fn=filename, stoplist_fn=stoplist_filename)

        lda = load_lda_sampler(lda_config, store_dir)

        assert lda is not None

    except Exception as ex:
        raise ex
    finally:
        javabridge.kill_vm()

def test_create_lda_dataset():
    javabridge.start_vm(run_headless=True, class_path=javabridge.JARS + [PCPLDA_JAR_PATH])
    try:
        filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
        stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
        with open(filename) as f:
            train_corpus = f.readlines()

        lda_ds = create_lda_dataset(train_corpus, None, stoplist_filename)

        assert lda_ds is not None

    except Exception as ex:
        raise ex
    finally:
        javabridge.kill_vm()

def test_sample_pclda():
    javabridge.start_vm(run_headless=True, class_path=javabridge.JARS + [PCPLDA_JAR_PATH])
    try:

        store_dir = "stored_samplers"
        filename = os.path.join(os.getcwd(), "tests", "corpus.txt")
        stoplist_filename = os.path.join(os.getcwd(), "tests", "stoplist.txt")
        lda_config = create_simple_lda_config(dataset_fn=filename, stoplist_fn=stoplist_filename)

        with open(filename) as f:
            train_corpus = f.readlines()

        lda_ds = create_lda_dataset(train_corpus, None, stoplist_filename)

        lda = sample_pclda(lda_config, lda_ds, iterations=2000, samplerType="cc.mallet.topics.PolyaUrnSpaliasLDA", testset=None, save_sampler=True)

        assert lda is not None

    except Exception as ex:
        raise ex
    finally:
        javabridge.kill_vm()


def getRandomNumber():
    """ This number has been chosen by a fair dice roll and is hence guaranteed to be random. """
    return 4

#import pypclda.pypclda as pypclda
# >>> class Integer:
#         new_fn = javabridge.make_new("java/lang/Integer", "(I)V")
#         def __init__(self, i):
#             self.new_fn(i)
#         intValue = javabridge.make_method("intValue", "()I", "Retrieve the integer value")
# >>> i = Integer(435)
# >>> i.intValue()

# def test_can_activate_java():

#     javabridge.start_vm(run_headless=True, class_path=javabridge.JARS +
#                         [jar_path])
#     try:
#         # Bind a Java variable and run a script that uses it.
#         print(javabridge.run_script(
#             'java.lang.String.format("Hello, %s!", greetee);',
#             dict(greetee="world")))

#         # Wrap a Java object and call some of its methods.
#         array_list = javabridge.JWrapper(javabridge.make_instance(
#             "java/util/ArrayList", "()V"))
#         array_list.add("ArrayList item 1")
#         array_list.add("ArrayList item 2")
#         print("ArrayList size:", array_list.size())
#         print("First ArrayList item:", array_list.get(0))

#         # Wrap a Java object from our jar and call a method.
#         main1 = javabridge.JWrapper(javabridge.make_instance(
#             "net/talvi/pythonjavabridgedemos/Main",
#             "(Ljava/lang/String;)V", "Alice"))
#  lda_configint(main1.greet("Hello"))

#         # Wrap a Java object using JClassWrapper (no signature required)
#         main2 = javabridge.JClassWrapper(
#             "net.talvi.pythonjavabridgedemos.Main")("Bob")
#         print(main2.greet("Hi there"))
#         print("2 + 2 =", main2.add(2, 2))

#     finally:
#         javabridge.kill_vm()