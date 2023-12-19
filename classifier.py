#############################################################
## DESCRIPTION: In this assignment, you will explore the
## text classification problem of identifying complex words.
## We have provided the following skeleton for your code,
## with several helper functions, and all the required
## functions you need to write.
#############################################################

from collections import defaultdict
import gzip
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import csv
from sklearn.metrics import precision_score



#### 1. Evaluation Metrics ####

## Input: y_pred, a list of length n with the predicted labels,
## y_true, a list of length n with the true labels

## Calculates the precision of the predicted labels


def get_precision(y_pred, y_true):

    # return precision_score(y_true, y_pred)

    t_pos = 0
    f_pos = 0

    for pred, true in zip(y_pred, y_true):
        if pred == true:
            t_pos+=1
        else:
            f_pos+=1

    precision = t_pos / ( t_pos + f_pos)
    return precision

## Calculates the recall of the predicted labels
def get_recall(y_pred, y_true):

    t_pos = 0
    f_neg = 0

    for pred, true in zip(y_pred, y_true):
        # if we predict simple but its complex
        if pred == 0 and true == 1:
            f_neg += 1

        if pred == true:
            t_pos+= 1

    recall = t_pos / (t_pos + f_neg)

    return recall

## Calculates the f-score of the predicted labels
def get_fscore(y_pred, y_true):
    # F = 2PR/(P+R)
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = (2*precision*recall) / ( recall + precision )

    return fscore

## Calls the top 3
def gather_performance(y_pred, y_true):
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)

    return precision, recall, fscore



#### 2. Complex Word Identification ####

## Loads in the words and labels of one of the datasets
def load_file(data_file):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels

def load_test_file( test_file ):
    words = []
    with open(test_file, 'rt', encoding="utf8" ) as f:
        i = 0
        for line in f:
            if i > 0:
                arr = line.split("\t")
                words.append(arr[0].lower())
            i += 1

    return words



def load_file_w_complexword_length(data_file, complex_word_length):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                # if the word lenght is 9 or more... complex
                if len(line_split[0]) < complex_word_length:
                    labels.append(0)
                else:
                    labels.append(1)
            i += 1
    return words, labels


def load_file_w_complexword_occ(data_file, counts, avg):
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                word = line_split[0].lower()
                words.append(word)

                if "-" in word:
                    arr = word.split("-")
                    word = arr[0]

                # if word is in dictionary
                if word in counts.keys():

                    # if the word is used less than avg, its complex
                    if counts[word] < avg:
                        labels.append(1)
                    # else its not complex, ie used a lot
                    else:
                        labels.append(0)
                # if word is not present in counys or not found easily, we say its simple for now
                else:
                    labels.append(0)

            i += 1
    return words, labels

### 2.1: A very simple baseline

## Makes feature matrix for all complex
def all_complex_feature(words):
    pass

## Labels every word complex
def all_complex(data_file):

    words, labels = load_file(data_file)

    all_complex = [1 for i in range(0, len(labels))]

    precision = get_precision(all_complex, labels)
    recall = get_recall(all_complex, labels)
    fscore = get_fscore(all_complex, labels)

    performance = [precision, recall, fscore]
    return performance

## finds avg complex word size
def find_avg_cmplx_word_length(test_file, dev_file):
    t_words, t_labels = load_file(test_file)
    d_words, d_labels = load_file(dev_file)


    count = 0
    avg = 0

    for word, label in zip(t_words,t_labels):
        # if the word is complex
        if label == 1:
            avg += len(word)
            count+=1

    # incorporates the dev file for teh average
    for word, label in zip(d_words,d_labels):
        # if the word is complex
        if label == 1:
            avg += len(word)
            count+=1

    avg = avg / count

    return avg


## finds avg unigram count of complex words
def find_avg_cmplx_occ(test_file, dev_file, counts):
    t_words, t_labels = load_file(test_file)
    d_words, d_labels = load_file(dev_file)

    count = 0
    avg = 0

    for word, label in zip(t_words, t_labels):
        # if the word is complex
        if label == 1:
            # if the word is in the dict
            if "-" in word:
                arr = word.split("-")
                word = arr[0]

            if word.lower() in counts:
                num_occ = counts[word.lower()]
                avg+= num_occ
                count+=1
    # LAZily adding dev set to average
    for word, label in zip(d_words, d_labels):
        if "-" in word:
            arr = word.split("-")
            word = arr[0]
        # if the word is complex
        if label == 1:
            # if the word is in the dict
            if word.lower() in counts:
                num_occ = counts[word.lower()]
                avg+= num_occ
                count+=1

    avg = avg / count

    return avg



### 2.2: Word length thresholding

## Makes feature matrix for word_length_threshold
def length_threshold_feature(words, threshold):

    pass

## Finds the best length threshold by f-score, and uses this threshold to
## classify the training and development set
def word_length_threshold(training_file, development_file, complex_word_length):
    ## YOUR CODE HERE

    t_words1, t_labels_pred = load_file_w_complexword_length(training_file, complex_word_length)
    d_words, d_labels_pred = load_file_w_complexword_length(development_file, complex_word_length)

    t_words2, t_labels_true = load_file(training_file)
    d_words2, d_labels_true = load_file(development_file)

    tprecision, trecall, tfscore = gather_performance(t_labels_pred, t_labels_true)
    dprecision, drecall, dfscore = gather_performance(d_labels_pred, d_labels_true)






    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.3: Word frequency thresholding

## Loads Google NGram counts
def load_ngram_counts(ngram_counts_file):
   counts = defaultdict(int)
   with gzip.open(ngram_counts_file, 'rt') as f:
       for line in f:
           token, count = line.strip().split('\t')
           if token[0].islower():
               counts[token] = int(count)
   return counts

# Finds the best frequency threshold by f-score, and uses this threshold to
## classify the training and development set

## Make feature matrix for word_frequency_threshold
# def frequency_threshold_feature(words, threshold, counts):
#     pass

def word_frequency_threshold(training_file, development_file, counts, avg):
    ## YOUR CODE HERE
    t_words1, t_labels_pred = load_file_w_complexword_occ(training_file, counts, avg)
    d_words, d_labels_pred = load_file_w_complexword_occ(development_file, counts, avg)



    t_words2, t_labels_true = load_file(training_file)
    d_words2, d_labels_true = load_file(development_file)

    if ( len(t_labels_pred) != len(t_labels_true )):
        print("not equal")
        print( len(t_labels_pred))
        print( len(t_labels_true ))


    tprecision, trecall, tfscore = gather_performance(t_labels_pred, t_labels_true)
    dprecision, drecall, dfscore = gather_performance(d_labels_pred, d_labels_true)


    training_performance = [tprecision, trecall, tfscore]
    development_performance = [dprecision, drecall, dfscore]
    return training_performance, development_performance

### 2.4: Naive Bayes

## Trains a Naive Bayes classifier using length and frequency features
def naive_bayes(training_file, development_file, counts):

    t_words, t_labels_true = load_file(training_file)
    d_words, d_labels_true = load_file(development_file)

    word_lengths = []
    word_occs = []
    for w in t_words:
        w = w.lower()
        word_lengths.append( len(w) )

        if "-" in w:
            arr = w.split("-")
            w = arr[0]
        if w in counts.keys():

            word_occs.append( counts[w] )


    avg_word_len = sum(word_lengths) / len( t_words )
    sd_word_len = np.std(word_lengths)

    avg_word_occ = sum(word_occs) / len(word_occs)
    sd_word_occ = np.std(word_occs)



    # creating x_train
    x_train = []
    for word in t_words:

        #normalizing features
        normalized_len = (len(word) - avg_word_len) / sd_word_len
        normalized_occ = ( counts[word]  - avg_word_occ ) / sd_word_occ
        x_train.append([normalized_len, normalized_occ])



    clf = GaussianNB()
    xtrain = np.array(x_train)
    y = np.array(t_labels_true)

    clf.fit(xtrain, y)


    # ok now we'll make the numpy array to make predictions on development data
    x_test_dev_data = []
    for w in d_words:
        normalized_len = (len(w) - avg_word_len) / sd_word_len
        normalized_occ = ( counts[w]  - avg_word_occ ) / sd_word_occ
        x_test_dev_data.append([normalized_len, normalized_occ])


    Y_pred_dev = clf.predict(x_test_dev_data)
    Y_pred_test = clf.predict(xtrain)

    development_performance = gather_performance(Y_pred_dev, d_labels_true )
    training_performance = gather_performance (Y_pred_test, t_labels_true )

    return training_performance, development_performance








    ## YOUR CODE HERE
    # training_performance = (tprecision, trecall, tfscore)
    # development_performance = (dprecision, drecall, dfscore)
    # return development_performance

### 2.5: Logistic Regression

## Trains a Naive Bayes classifier using length and frequency features
def logistic_regression(training_file, development_file, counts):

    t_words, t_labels_true = load_file(training_file)
    d_words, d_labels_true = load_file(development_file)

    word_lengths = []
    word_occs = []
    for w in t_words:
        w = w.lower()
        word_lengths.append( len(w) )

        if "-" in w:
            arr = w.split("-")
            w = arr[0]
        if w in counts.keys():

            word_occs.append( counts[w] )


    avg_word_len = sum(word_lengths) / len( t_words )
    sd_word_len = np.std(word_lengths)

    avg_word_occ = sum(word_occs) / len(word_occs)
    sd_word_occ = np.std(word_occs)



    # creating x_train
    x_train = []
    for word in t_words:

        #normalizing features
        normalized_len = (len(word) - avg_word_len) / sd_word_len
        normalized_occ = ( counts[word]  - avg_word_occ ) / sd_word_occ
        x_train.append([normalized_len, normalized_occ])



    clf = LogisticRegression()
    xtrain = np.array(x_train)
    y = np.array(t_labels_true)

    clf.fit(xtrain, y)


    # ok now we'll make the numpy array to make predictions on development data
    x_test_dev_data = []
    for w in d_words:
        normalized_len = (len(w) - avg_word_len) / sd_word_len
        normalized_occ = ( counts[w]  - avg_word_occ ) / sd_word_occ
        x_test_dev_data.append([normalized_len, normalized_occ])


    Y_pred_dev = clf.predict(x_test_dev_data)
    Y_pred_test = clf.predict(xtrain)

    development_performance = gather_performance(Y_pred_dev, d_labels_true )
    training_performance = gather_performance (Y_pred_test, t_labels_true )

    return training_performance, development_performance




def report_all_complex(training_file):

    performance = all_complex(training_file)
    print("Printing scores after assuming all words to be complex")
    print(f" Precision: {performance[0]}\n Recall: {performance[1]}\n Fscore: {performance[2]}")
    print("#" * 50)
    print()

def report_cmplx_word_length_thrshold(training_file, development_file):
    print("Using word length as a metric...")
    # finds avg complex word size from test data
    avg = math.ceil(find_avg_cmplx_word_length(training_file, development_file ) )
    # complex_word_length = int(input("Enter threshold for cmplx word length: "))

    range_of_lengths = [avg-3, avg-2, avg-1, avg, avg+1, avg+2, avg+3]

    for value in range_of_lengths:
        print(f"Score for development and training sets, with complex word length = {value}\n")

        t_res_words, d_res_words = word_length_threshold(training_file, development_file, value)
        printResults(t_res_words, d_res_words)


def report_cmplx_word_occ_thrshold(training_file, development_file, counts):
    print("Using average occurence as threshold")
    # finds avg complex word size from test data
    avg = math.ceil(find_avg_cmplx_occ(training_file, development_file, counts ) )
    # complex_word_length = int(input("Enter threshold for cmplx word length: "))

    range_of_lengths = [avg-3000000, avg-2000000, avg-1000000, avg, avg+1000000, avg+2000000, avg+3000000]

    for value in range_of_lengths:
        print(f"Score for development and training sets, with avg occ = {value}\n")
        t_res_occ, d_res_occ = word_frequency_threshold(training_file, development_file, counts, value)
        printResults(t_res_occ, d_res_occ)


def plot_cmplx_word_length(training_file, development_file):

    x_test = []
    y_test = []

    x_dev = []
    y_dev = []

    avg = math.ceil(find_avg_cmplx_word_length(training_file, development_file) )
    range_of_lengths = [avg-3, avg-2, avg-1, avg, avg+1, avg+2, avg+3]

    for value in range_of_lengths:
        # precision ( y ), recall ( x ),  fscore
        t_res_words, d_res_words = word_length_threshold(training_file, development_file, value)
        y_test.append( t_res_words[0] )
        y_dev.append( d_res_words [0] )

        x_test.append(t_res_words[1])
        x_dev.append(d_res_words[1])



    # Create the plot
    plt.plot(x_test, y_test, label='Training Data')
    plt.plot(x_dev, y_dev, label='Development Data')

    # Add labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Complex Word Length')

    # Add a legend
    plt.legend()
    plt.show()

def plot_cmplx_word_occ(training_file, development_file, counts):
    x_test = []
    y_test = []

    x_dev = []
    y_dev = []

    avg = math.ceil(find_avg_cmplx_occ(training_file, development_file, counts) )

    range_of_occ = [avg-3000000, avg-2000000, avg-1000000, avg, avg+1000000, avg+2000000, avg+3000000]

    for value in range_of_occ:
        # precision ( y ), recall ( x ),  fscore
        t_res_words, d_res_words = word_frequency_threshold(training_file, development_file, counts, value)
        y_test.append( t_res_words[0] )
        y_dev.append( d_res_words [0] )

        x_test.append(t_res_words[1])
        x_dev.append(d_res_words[1])



        # Create the plot
    plt.plot(x_test, y_test, label='Training Data')
    plt.plot(x_dev, y_dev, label='Development Data')

    # Add labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Complex Word Occurence')

    # Add a legend
    plt.legend()
    plt.show()

def plot_both_classifiers_dev(training_file, development_file, counts):
    x_length = []
    y_length = []

    x_occ = []
    y_occ = []

    avg_length = math.ceil(find_avg_cmplx_word_length(training_file, development_file) )
    range_of_lengths = [avg_length-3, avg_length-2, avg_length-1, avg_length, avg_length+1, avg_length+2, avg_length+3]

    avg_occ = math.ceil(find_avg_cmplx_occ(training_file, development_file, counts) )
    range_of_occ = [avg_occ-3000000, avg_occ-2000000, avg_occ-1000000, avg_occ, avg_occ+1000000, avg_occ+2000000, avg_occ+3000000]


    for value in range_of_occ:
        # precision ( y ), recall ( x ),  fscore

        # ignoring training results
        t_res_words, d_res_words = word_frequency_threshold(training_file, development_file, counts, value)
        y_occ.append( d_res_words [0] )
        x_occ.append(d_res_words[1])

    for value in range_of_lengths:
        # precision ( y ), recall ( x ),  fscore

        # ignoring training results
        t_res_words, d_res_words = word_length_threshold(training_file, development_file, value)
        y_length.append( d_res_words [0] )
        x_length.append(d_res_words[1])


    plt.plot(x_length, y_length, label='Length Classifier')
    plt.plot(x_occ, y_occ, label='Occurence Classifier')

    # Add labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve comparing Classifiers')

    # Add a legend
    plt.legend()
    plt.show()


# trains a logistic regression model on test and dev data
def run_test_data(training_file, development_file, test_file, counts ):


    t_words, t_labels_true = load_file(training_file)
    d_words, d_labels_true = load_file(development_file)

    word_lengths = []
    word_occs = []
    for w in t_words:
        w = w.lower()
        word_lengths.append( len(w) )

        if "-" in w:
            arr = w.split("-")
            w = arr[0]
        if w in counts.keys():

            word_occs.append( counts[w] )


    avg_word_len = sum(word_lengths) / len( t_words )
    sd_word_len = np.std(word_lengths)

    avg_word_occ = sum(word_occs) / len(word_occs)
    sd_word_occ = np.std(word_occs)



    # creating x_train
    x_train = []
    for word in t_words:

        #normalizing features
        normalized_len = (len(word) - avg_word_len) / sd_word_len
        normalized_occ = ( counts[word]  - avg_word_occ ) / sd_word_occ
        x_train.append([normalized_len, normalized_occ])

    x_dev = []
    for w in d_words:
        normalized_len = (len(w) - avg_word_len) / sd_word_len
        normalized_occ = ( counts[w]  - avg_word_occ ) / sd_word_occ
        x_dev.append([normalized_len, normalized_occ])

    clf = LogisticRegression()


    x_dev_np = np.array(x_dev)
    x_train_np = np.array(x_train)

    y_train_np = np.array(t_labels_true)
    y_dev_np = np.array( d_labels_true )

    x_combined = np.concatenate((x_dev_np, x_train_np))
    y_combined = np.concatenate((y_dev_np, y_train_np))

    # model trained on test and dev data...
    clf.fit(x_combined, y_combined)


    test_words = load_test_file(test_file)

    norm_test_stats = []
    for w in test_words:
        normalized_len = (len(w) - avg_word_len) / sd_word_len
        normalized_occ = ( counts[w]  - avg_word_occ ) / sd_word_occ
        norm_test_stats.append( [normalized_len, normalized_occ])


    pred_on_test = clf.predict(norm_test_stats)

    csv_file = 'predict_gal62.csv'

    # Combine the lists into a list of tuples
    data = list(zip(test_words, pred_on_test))

    # Write the data to the CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['WORD', 'PREDICTION'])  # Write header
        writer.writerows(data)  # Write the data


















def printResults( train, dev ):
    print("For test set")
    print(f" Precision: {train[0]}\n Recall: {train[1]}\n Fscore: {train[2]}\n")
    print("For development set")
    print(f" Precision: {dev[0]}\n Recall: {dev[1]}\n Fscore: {dev[2]}\n")
    print("#" * 50 )





if __name__ == "__main__":
    training_file = "data/complex_words_training.txt"
    development_file = "data/complex_words_development.txt"
    test_file = "data/complex_words_test_unlabeled.txt"
    ngram_counts_file = "ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)



    ### part 1 - returning performance after making everything complex
    # report_all_complex(training_file)
    #
    #
    # # word_length_baseline(training_file)
    # # eventually prompt user input and graph the stuff
    #
    # # finds avg length and prints the scores for 2 up and 2 down
    # report_cmplx_word_length_thrshold(training_file, development_file)
    #
    # plot_cmplx_word_length(training_file, development_file)
    # plot_cmplx_word_occ(training_file, development_file, counts)
    # plot_both_classifiers_dev(training_file, development_file, counts)
    #
    # report_cmplx_word_occ_thrshold(training_file, development_file, counts)





    ## Finding avg occurence of complex word
    # print("Finding avg comlex word occurence")
    # avg_occ = find_avg_cmplx_occ(training_file, development_file, counts)

    # print("Using average occurence as threshold")
    # t_res_occ, d_res_occ = word_frequency_threshold(training_file, development_file, counts, avg_occ)
    # printResults(t_res_occ, d_res_occ)

    print("Training a Naive Bayes Classifier...\n")

    train_performance, dev_performance = naive_bayes(training_file, development_file, counts)
    printResults(train_performance, dev_performance)

    print("Training a Logistic Regression...\n")
    log_train_performance, log_dev_performance = logistic_regression(training_file, development_file, counts)
    printResults(log_train_performance, log_dev_performance)


    run_test_data(training_file, development_file, test_file, counts)
