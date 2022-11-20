#import gzip
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
counter = 0
for i in range(0,X_train.shape[0]):
    sen = X_train[i]
    sen = sen[0]
    sent_1 = gensim.utils.simple_preprocess(sen)

    res =len(sent_1)
    if(res > 25):
        counter += 1

print(counter)

data_y = []
data_x = []
"""
for i in range(0,asa.shape[0]):
    data = X_train[i]
    X_train.append(data[0])
    label = y_train[i]
    if(label == 'tech'):
        data_y.append(1)
    if(label == 'entertainment'):
        data_y.append(2)
    if(label == 'sport'):
        data_y.append(3)
"""
counter = 0
"""
data_x = np.array(data_x)
data_y = np.array(data_y)
print(data_x.shape)
print(data_y.shape)
"""
"""
data_x2 = []
data_y2 = []
for i in range(0,X_train.shape[0]):
    sent = X_train[i]
    sent_1 = gensim.utils.simple_preprocess(sent[i])

    res =len(sent_1)
    #res = len(test_string.split())
    #print(res)
    if(res <= 25):
        counter += 1
        data_x2.append(data_x[i])
        data_y2.append(data_y[i])

X_train, X_test, y_train, y_test = train_test_split(data_x2, data_y2, test_size=0.25, random_state=42)

"""
s1 = []
s2 = []
labels = []
labels_one = []
X_train = np.array(X_test)
y_train = np.array(y_test)
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
np.save("test_labels_twitter.npy", labels)
np.save("test_labels_1d_twitter.npy", labels_one)
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
print("Shape :--------------")
print(similarity_data.shape)
#print(similarity_matrix)
#print(similarity_matrix.shape)
print("First Matrix :--------------")
print(similarity_data[0])

end = time.time()
print(end - start)
"""
import numpy
mat = numpy.matrix("1 2 3; 4 5 6; 7 8 9")
mat.dump("my_matrix.dat")
mat2 = numpy.load("my_matrix.dat")
"""
