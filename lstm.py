import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import collections
import random
from scipy import spatial
import gensim
import logging
import csv
import numpy as np
import pandas as pd
import time

import gensim
import logging
import csv
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
start = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
"""
with gzip.open ("quora_duplicate_questions.tsv", 'rb') as f:
        for i,line in enumerate (f):
            print(line)
            break
"""
"""
counter = 0
maximum = 0
df = pd.read_csv(r"processedNegative_p.csv", sep = ',')
data = np.array(df)
data_1 = data[:1117]
y_1 = []
for i in range(0, data_1.shape[0]):
    y_1.append(1)

df_1 = pd.read_csv(r"processedNeutral_p.csv", sep = ',')
data = np.array(df_1)
print(data.shape)
data_2 = data[:1570]
y_2 = []
for i in range(0, data_2.shape[0]):
    y_2.append(2)
df_2 = pd.read_csv(r"processedPositive_p.csv", sep = ',')
data = np.array(df_2)
print(data.shape)
data_3 = data[:1186]
y_3 = []
for i in range(0, data_3.shape[0]):
    y_3.append(3)

data_4 = np.concatenate((data_1,data_2))
final_data = np.concatenate((data_4,data_3) )

y_4 = np.concatenate((y_1,y_2))
final_y = np.concatenate((y_4,y_3) )

X_train, X_test, y_train, y_test = train_test_split(final_data, final_y, test_size=0.33, random_state=42)
"""
X_train = np.load('X_train.npy',allow_pickle=True)

y_train = np.load('y_train.npy',allow_pickle=True)

X_test = np.load('X_test.npy',allow_pickle=True)
y_test = np.load('y_test.npy',allow_pickle=True)

counter = 0

data_y = []
data_x = []

counter = 0

s1 = []
s2 = []
labels = []
labels_one = []
X_train = np.array(X_train)
y_train = np.array(y_train)
for i in range(0 , X_train.shape[0]):

    label = float(y_train[i])
    sent = X_train[i]
    sent_1 = gensim.utils.simple_preprocess(sent[0])
    #sent_2 = gensim.utils.simple_preprocess(train_set[i,4])

    flag = False
    if(len(sent_1) <= 25 ) :
        #q11 = np.append( q11 ,what_array_1 )
        #q22 = np.append( q22 ,what_array_2 )
        flag =True
        s1.append(sent_1)
        counter += 1
        #maximum = q11.shape[0]
    #s = ''.join(c for c in s if c.isnumeric())

    if (flag == True):
        my_labels = []
        first = float(label)
        if(first == 1):
            my_labels = [1,0,0]
        if(first == 2):
            my_labels = [0,1,0]
        if(first == 3):
            my_labels = [0,0,1]
        labels_one.append(first)
        labels.append(my_labels)

    if ( counter % 100 == 0 ):
       # print("maximum number is: {0} ".format(maximum))
        print(counter)



print("done")

"""
x1 = np.array(x1)
x2 = np.array(x2)
"""
print("First part Finished...")
print(counter)
labels = np.array(labels)
print(labels.shape)

print("#############################")
#model = gensim.models.Word2Vec.load("msrp_w2vec")

#np.save("my_labels_msrp_train.txt", labels)
y_train = labels
y_train = np.array(y_train)

y_train = y_train.reshape(y_train.shape[0],3)


#np.save("test_labels_twitter.npy", labels)
#np.save("test_labels_1d_twitter.npy", labels_one)
####asdasd
model = gensim.models.Word2Vec.load("model_twitter")

#model = gensim.models.KeyedVectors.load_word2vec_format("model_twitter")

#similarity_matrix = np.zeros(shape=(x1[10].shape[0],x2[10].shape[0]))
similarity_matrix = np.zeros(shape=(25,25))
print(similarity_matrix.shape)
i = -1
j = -1
similarity_data = []

#markers = np.fromfile("markers.txt")
data_num = 0
#(y.shape[0]+1)
for k in range(0,len(s1)):
    i = 0
    #similarity_matrix = np.zeros(shape=(x1[k].shape[0],x2[k].shape[0]))#np.zeros(shape=(50,50))
    similarity_matrix = []

    for word_q1 in s1[k]:
        if word_q1 in model.wv.vocab :
            similarity_matrix.append(model[word_q1])
    # final embedding array corresponds to dictionary of words in the document
    #embedding = np.asarray(embeddings_tmp)

        #similarity_matrix[i] = model.wv[word_q1]#model.similarity(word_q1, word_q2)
        i =+ 1
    zero = np.zeros(shape=(25))
    while(len(similarity_matrix)<25):
        similarity_matrix.append(zero)
    similarity_matrix = np.asarray(similarity_matrix)
    """
    if(data_num == 30 ):
        print(x1[k])
        print(x2[k])
        print(similarity_matrix)
        break
    """
    similarity_data.append(similarity_matrix)

    data_num +=1
    if ( data_num % 100 == 0 ):
             print("data number {0} is writing to the file...".format(data_num))
np.save("twitter_test.npy", similarity_data)
similarity_data = np.array(similarity_data)
my_train_embedding = similarity_data
my_train_embedding = my_train_embedding.reshape(my_train_embedding.shape[0], 25 , 25)


###################################################################################
data_y = []
data_x = []

counter = 0

s1 = []
s2 = []
labels = []
labels_one = []
X_test = np.array(X_test)
y_test = np.array(y_test)
for i in range(0 , X_test.shape[0]):

    label = float(y_test[i])
    sent = X_test[i]
    sent_1 = gensim.utils.simple_preprocess(sent[0])
    #sent_2 = gensim.utils.simple_preprocess(train_set[i,4])

    flag = False
    if(len(sent_1) <= 25 ) :
        #q11 = np.append( q11 ,what_array_1 )
        #q22 = np.append( q22 ,what_array_2 )
        flag =True
        s1.append(sent_1)
        counter += 1
        #maximum = q11.shape[0]
    #s = ''.join(c for c in s if c.isnumeric())

    if (flag == True):
        my_labels = []
        first = float(label)
        if(first == 1):
            my_labels = [1,0,0]
        if(first == 2):
            my_labels = [0,1,0]
        if(first == 3):
            my_labels = [0,0,1]
        labels_one.append(first)
        labels.append(my_labels)

    if ( counter % 100 == 0 ):
       # print("maximum number is: {0} ".format(maximum))
        print(counter)



print("done")

"""
x1 = np.array(x1)
x2 = np.array(x2)
"""
print("First part Finished...")
print(counter)
labels = np.array(labels)
print(labels.shape)

print("#############################")
#model = gensim.models.Word2Vec.load("msrp_w2vec")

#np.save("my_labels_msrp_train.txt", labels)
y_test = labels
y_test = np.array(y_test)


y_test = y_test.reshape(y_test.shape[0],3)


model = gensim.models.Word2Vec.load("model_twitter")


similarity_matrix = np.zeros(shape=(25,25))
print(similarity_matrix.shape)
i = -1
j = -1
similarity_data = []

#markers = np.fromfile("markers.txt")
data_num = 0
#(y.shape[0]+1)
for k in range(0,len(s1)):
    i = 0
    #similarity_matrix = np.zeros(shape=(x1[k].shape[0],x2[k].shape[0]))#np.zeros(shape=(50,50))
    similarity_matrix = []

    for word_q1 in s1[k]:
        if word_q1 in model.wv.vocab :
            similarity_matrix.append(model[word_q1])
    # final embedding array corresponds to dictionary of words in the document
    #embedding = np.asarray(embeddings_tmp)

        #similarity_matrix[i] = model.wv[word_q1]#model.similarity(word_q1, word_q2)
        i =+ 1
    zero = np.zeros(shape=(25))
    while(len(similarity_matrix)<25):
        similarity_matrix.append(zero)
    similarity_matrix = np.asarray(similarity_matrix)
    """
    if(data_num == 30 ):
        print(x1[k])
        print(x2[k])
        print(similarity_matrix)
        break
    """
    similarity_data.append(similarity_matrix)

    data_num +=1
    if ( data_num % 100 == 0 ):
             print("data number {0} is writing to the file...".format(data_num))
np.save("twitter_test.npy", similarity_data)
similarity_data = np.array(similarity_data)
my_test_embedding = similarity_data
my_test_embedding = my_test_embedding.reshape(my_test_embedding.shape[0], 25 , 25)

###################################
learning_rate = 0.01
training_steps = 3000
batch_size = 50
display_step = 200
train_res = []
test_res = []

num_input = 25
timesteps = 25
num_hidden = 256
num_classes = 3

X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])


weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):

    x = tf.unstack(x, timesteps, 1)


    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)


    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)


    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()


with tf.Session() as sess:


    sess.run(init)
    batch = 0
    #counter = 0
    for step in range(1, training_steps+1):
        if((y_train.shape[0] - (batch*batch_size)) < batch_size ):
            batch = 0
        batch_x = my_train_embedding[batch*batch_size: (batch+1)*batch_size]


        batch_y = y_train[batch*batch_size: (batch+1)*batch_size]
        #print(batch_x.shape)
        #print(batch_y.shape)
        batch += 1

        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        #print(type(batch_x))
        #print(type(batch_y))

        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)

        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        #counter =+ 1
        #print(step)
        #print(batch*batch_size)
        #print("----------------")
        if step % display_step == 0 or step == 1:
            print("----------------")
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            test_len = y_test.shape[0]
            test_data = my_test_embedding.reshape((-1, timesteps, num_input))
            test_label = y_test[:test_len]
            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    print("Optimization Finished!")


    test_len = y_test.shape[0]
    test_data = my_test_embedding.reshape((-1, timesteps, num_input))
    test_label = y_test[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
