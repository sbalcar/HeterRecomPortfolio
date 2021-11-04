# Word2Vec: Skipgram Model
#---------------------------------------
#
# In this example, we will download and preprocess the movie
#  review data.
#
# From this data set we will compute/fit the skipgram model of
#  the Word2Vec Algorithm
#
# Skipgram: based on predicting the surrounding words from the
#  Ex sentence "the cat in the hat"
#  context word:  ["hat"]
#  target words: ["the", "cat", "in", "the"]
#  context-target pairs:
#    ("hat", "the"), ("hat", "cat"), ("hat", "in"), ("hat", "the")

def word2vecRun(learningRate:float, window_size = 3, embedding_size = 64, generations = 100000, texts = []):
    import tensorflow.compat.v1 as tf
    #import tensorflow as tf
    tf.disable_v2_behavior()
    import numpy as np
    import os
    #import text_helpers
    from recommender.w2v.text_helpers import build_dictionary #function
    from recommender.w2v.text_helpers import text_to_numbers #function
    from recommender.w2v.text_helpers import generate_batch_data #function
    from tensorflow.python.framework import ops
    from configuration.configuration import Configuration #class

    ops.reset_default_graph()

    # Start a graph session
    sess = tf.Session()

    # Declare model parameters
    batch_size:int = 32
    vocabulary_size:int = 10000
    #generations:int = 200000
    #generations:int = 400000
    #generations:int = 450000

    #embedding_size = 64   # Word embedding size
    #doc_embedding_size = 64   # Document embedding size
    #concatenated_size = embedding_size + doc_embedding_size

    num_sampled = int(batch_size/2)    # Number of negative examples to sample.
    #window_size = 3       # How many words to consider to the left.
    # Add checkpoints to training
    save_embeddings_every = 50000
    print_valid_every:int = 50000
    print_loss_every:int = 10000

    # Declare stop words
    #stops = stopwords.words('english')
    stops = []

    # Load the movie review data
    #print('Loading Data')
    #texts = text_helpers.load_ml1m_data(data_folder_name, dataName)

    # Texts must contain at least 3 words
    #target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
    #texts = [x for x in texts if len(x.split()) > window_size]
    #assert(len(target)==len(texts))

    # Build our data set and dictionaries
    print('Creating Dictionary')
    word_dictionary = build_dictionary(texts, vocabulary_size)
    word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
    text_data = text_to_numbers(texts, word_dictionary)

    vocabulary_size = len(word_dictionary)
    print("Actual vocabulary size:"+str(vocabulary_size))

    # Get validation word keys
    valid_words = [word_dictionary_rev[1],word_dictionary_rev[10],word_dictionary_rev[100],word_dictionary_rev[1000]]
    valid_examples = [word_dictionary[x] for x in valid_words]

    # Define Embeddings:
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    # NCE loss parameters
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                   stddev=1.0 / np.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Create data/target placeholders
    x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Lookup the word embedding:
    embed = tf.nn.embedding_lookup(embeddings, x_inputs)

    loss = tf.reduce_mean(tf.nn.nce_loss(
            weights = nce_weights,
            biases = nce_biases,
            inputs = embed,
            labels = y_target,
            num_sampled = num_sampled,
            num_classes = vocabulary_size))

    # Create optimizer
    optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

    # Cosine similarity between words
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    #Add variable initializer.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Run the skip gram model.
    loss_vec = []
    loss_x_vec = []
    for i in range(generations):
        batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
        feed_dict = {x_inputs : batch_inputs, y_target : batch_labels}
        #print("x_inputs: " + str(x_inputs))

        #print("optimizer: " + str(optimizer))
        #print("")
        #print("feed_dict: " + str(feed_dict))

        # Run the train step
        sess.run(optimizer, feed_dict=feed_dict)

        # Return the loss
        if (i+1) % print_loss_every == 0:
            loss_val = sess.run(loss, feed_dict=feed_dict)
            loss_vec.append(loss_val)
            loss_x_vec.append(i+1)
            print("Loss at step {} : {}".format(i+1, loss_val))

        # Validation: Print some random words and top 5 related words
        if (i+1) % print_valid_every == 0:
            sim = sess.run(similarity, feed_dict=feed_dict)
            for j in range(len(valid_words)):
                valid_word = word_dictionary_rev[valid_examples[j]]
                top_k = 5 # number of nearest neighbors
                nearest = (-sim[j, :]).argsort()[1:top_k+1]
                log_str = "Nearest to {}:".format(valid_word)
                for k in range(top_k):
                    close_word = word_dictionary_rev[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

        #if (i+1) % save_embeddings_every == 0:
        #     final_embeddings = sess.run(embeddings)
        #     caller.saveModel((i+1), final_embeddings, word_dictionary_rev, word_dictionary)


    final_embeddings = sess.run(embeddings)

    #embeddingsFname = Configuration.modelDirectory + os.sep +"embed_word2vec_"+str(window_size)+"_"+str(embedding_size)+".csv"
    #np.savetxt(embeddingsFname, final_embeddings, fmt="%.6e")
    return(final_embeddings, word_dictionary_rev, word_dictionary)