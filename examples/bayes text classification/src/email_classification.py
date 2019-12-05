# encoding=utf-8

import glob
import os
import pickle
import re
import time
from collections import Counter

import jieba

import config


def split2words(content:str)->set:
    """
    Split text to word using jieba

    """
    words = set()
    raw_words = list(jieba.cut(content))
    for raw_word in raw_words:
        if raw_word.strip() == '':
            continue
        words.add(raw_word)
    return words


def filter_nonzh(content:str)->str:
    """
    Filter non-chinese str

    """
    rule = re.compile(config.CHN_CODE_MASK)
    content = rule.sub("", content)
    return content


def get_stopwords()->list:
    """
    Returns: all stop words

    """
    stopwords = []
    with open(config.STOPWORDS_TABLE_PATH, "r", encoding='utf-8') as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    return stopwords


STOP_WORDS = get_stopwords()


def is_not_stopword(word:str)->bool:
    """
    Function for stop word checking
    Returns: True/False

    """
    return word not in STOP_WORDS


def filter_stopwords(words:list)->iter:
    """
    Filter the stop words

    """
    words = filter(is_not_stopword, words)
    return words


def calc_dir_word_freq(dir)->dict:
    """
    Calculate the word frequency under the specified directory
    Args:
        dir: relative or absolute directory

    Returns: words frequency dictionary

    """
    start_time = time.time()
    words_of_dir = []
    files = glob.glob(dir + '/*')
    for file_name in files:
        with open(file_name, 'r') as f:
            context = filter_nonzh(''.join(f.readlines()))
            words = split2words(context)
            words = filter_stopwords(words)
            words_of_dir.extend(words)
    print("Time cost:{}".format(time.time() - start_time))

    words_count_dict = Counter(words_of_dir)
    return words_count_dict


def calc_file_word_freq(file_name):
    """
    Calculate the word frequency under in specified file
    Args:
        file_name: file name in relative or absolute path

    Returns: words frequency dictionary

    """
    words_of_file = []
    with open(file_name, 'r') as f:
        context = filter_nonzh(''.join(f.readlines()))
        words = split2words(context)
        words = filter_stopwords(list(words))
        words_of_file.extend(words)

    words_count_dict = Counter(words_of_file)
    return words_count_dict


def get_words_prob(test_words_list:list, normal_words_freq:dict, spam_words_freq:dict, normal_file_number:int, spam_file_number:int,
                   max_prob_num:int):
    """
    Calculate the probability of spam for every words, p(s|wi)
    Args:
        test_words_list: the words list to test
        normal_words_freq: the words frequency in normal mail
        spam_words_freq: the words frequency in spam mail
        normal_file_number: the file number of normal mail
        spam_file_number: the file number of spam mail
        max_prob_num: the number of most max probability to consideration

    Returns: the dictionary of the most max-N probability

    """
    words_prob = {}
    for word in set(test_words_list):
        prob_ws = spam_words_freq[word] / spam_file_number if word in spam_words_freq.keys() else 0.01
        prob_wn = normal_words_freq[word] / normal_file_number if word in normal_words_freq.keys() else 0.01

        prob_sw = 0.4 if prob_ws == 0.01 and prob_wn == 0.01 else prob_ws / (prob_ws + prob_wn)
        words_prob[word] = prob_sw
    words_prob = dict(sorted(words_prob.items(), key=lambda d: d[1], reverse=True)[0:max_prob_num])
    return words_prob


def calc_bayes_prob(words_prob_dict):
    """
    Calculate bayes probability, P(s|w)
    Args:
        words_prob_dict: the dictionary of the most max-N probability

    Returns: bayes probability

    """
    prob_sw = 1
    prob_sn = 1

    for word, prob in words_prob_dict.items():
        prob_sw *= (prob)
        prob_sn *= (1 - prob)

    p = prob_sw / (prob_sw + prob_sn)
    return p


def is_spam(words_prob_dict, threshold:float)->bool:
    """
    Whether is spam
    Args:
        words_prob_dict: the probability dictionary of words to test
        threshold: the threshold to judge whether is spam

    Returns: True/False

    """
    p = calc_bayes_prob(words_prob_dict)
    return p > threshold


def calc_accuracy(result_dict)->float:
    """
    Count accuracy
    Args:
        result_dict: result dictionary

    Returns: accuracy

    """
    right_num = 0
    error_num = 0
    for name, catagory in result_dict.items():
        if (eval(name) < 1000 and catagory == False) or (eval(name) >= 1000 and catagory == True):
            right_num += 1
        else:
            error_num += 1
    return right_num / (right_num + error_num)


if __name__ == "__main__":
    normal_file_number = len(glob.glob(config.NORMAL_DATA_DIR + "/*"))
    spam_file_number = len(glob.glob(config.SPAM_DATA_DIR + "/*"))
    print("spam number:{},normal number:{}".format(spam_file_number,
                                                   normal_file_number))

    "Loading model or calculating words frequency from training datasets"
    if os.path.exists(config.NORMAL_WORDS_FREQ_FILE):
        with open(config.NORMAL_WORDS_FREQ_FILE, "rb") as f:
            normal_words_freq = pickle.load(f)
    else:
        normal_words_freq = calc_dir_word_freq(config.NORMAL_DATA_DIR)
        with open(config.NORMAL_WORDS_FREQ_FILE, "wb") as f:
            pickle.dump(normal_words_freq, f)

    if os.path.exists(config.SPAM_WORDS_FREQ_FILE):
        with open(config.SPAM_WORDS_FREQ_FILE, "rb") as f:
            spam_words_freq = pickle.load(f)
    else:
        spam_words_freq = calc_dir_word_freq(config.SPAM_DATA_DIR)
        with open(config.SPAM_WORDS_FREQ_FILE, "wb") as f:
            pickle.dump(spam_words_freq, f)

    "Testing on test datasets"
    result_dict = {}
    for file in glob.glob(config.TEST_DATA_DIR + "/*"):
        words_count_dict = calc_file_word_freq(file)
        words_prob_dict = get_words_prob(list(words_count_dict.keys()),
                                         normal_words_freq,
                                         spam_words_freq,
                                         normal_file_number,
                                         spam_file_number,
                                         1000
                                         )

        result_dict[os.path.basename(file)] = int(is_spam(words_prob_dict, 0.5))

    print("Accuracy:", calc_accuracy(result_dict))
    # for i, ic in result_dict.items():
    #     print(i + "/" + str(ic))
