# Text Helper Functions
#---------------------------------------
#
# We pull out text helper functions to reduce redundant code

import string
import os
import urllib.request
import io
import tarfile
import collections
import numpy as np


# Normalize text
def normalize_text(texts, stops):
    # Lower case
    texts = [x.lower() for x in texts]

    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts] #use stemmer here?

    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    
    return(texts)


# Build dictionary of words
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', -1]]
    
    # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    
    # Now create the dictionary
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it
    # the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    
    return(word_dict)
    

# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict):
    # Initialize the returned data
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence.split():
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return(data)
    

# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    #print(sentences)
    # Fill up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # select random sentence to start
        rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
        rand_sentence = sentences[rand_sentence_ix]
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix-window_size),0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]

        # Pull out center word of interest for each window and create a tuple for each window
        try: 
            if method=='skip_gram':
              batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x,y in zip(window_sequences, label_indices)]
              # Make it in to a big list of tuples (target word, surrounding word)
              tuple_data = [(x, y_) for x,y in batch_and_labels for y_ in y]
              #print("tuple_data: " + str(tuple_data))
              batch, labels = [list(x) for x in zip(*tuple_data)]
            else:
              raise ValueError('Method {} not implmented yet.'.format(method))
        
                
            # extract batch and labels
            batch_data.extend(batch[:batch_size])
            label_data.extend(labels[:batch_size])
        except:
            pass
    # Trim batch and label at the end
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    
    # Convert to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))
    
    return(batch_data, label_data)


# Load the movie review data
# Check if data was downloaded, otherwise download it and save for future use
def load_ml1m_data(data_folder_name, name= 'slantour_data.txt'):
    dt_file = os.path.join(data_folder_name, name)
    pos_data = []
    with open(dt_file, 'r', encoding="UTF-8") as temp_pos_file:
        for row in temp_pos_file:
            pos_data.append(row)

    return pos_data


