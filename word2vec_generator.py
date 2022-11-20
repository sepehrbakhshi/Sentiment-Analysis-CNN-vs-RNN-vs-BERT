
import gzip
import gensim
import logging
import csv
import numpy as np
import pandas as pd
###Train word2vec###########

"""
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
"""
"""
with gzip.open ("quora_duplicate_questions.tsv", 'rb') as f:
        for i,line in enumerate (f):
            print(line)
            break
"""

df = pd.read_csv(r"processedNegative_p.csv", sep = ',')
data = np.array(df)
data_1 = data[:1117]
df_1 = pd.read_csv(r"processedNeutral_p.csv", sep = ',')
data = np.array(df_1)
print(data.shape)
data_2 = data[:1570]
df_2 = pd.read_csv(r"processedPositive_p.csv", sep = ',')
data = np.array(df_2)
print(data.shape)
data_3 = data[:1186]
data_4 = np.concatenate((data_1,data_2))
final_data = np.concatenate((data_4,data_3) )
print(final_data.shape)
"""
with open("WikiQA.tsv") as fin:
    filereader = csv.reader(fin, delimiter='\t')
    x1, x2, y = [], [], []
    for _, q, _, _, _, answer,label in filereader:
        q11 = gensim.utils.simple_preprocess(q)
        q22 = gensim.utils.simple_preprocess(answer)
        x1.append(q11)
        x1.append(q22)
        #s = ''.join(c for c in s if c.isnumeric())
        if label !="Label":
            y.append(float(label))
"""
x1 = []
for i in range(0, final_data.shape[0]):
    q = final_data[i]
    q11 = gensim.utils.simple_preprocess(q[0])

    x1.append(q11)
"""
We can change the dimensions of the vector by changing the size element
in the model
This code generates 100 sized vector
"""
model = gensim.models.Word2Vec(
        x1,
        size=25,
        window=10,
        min_count=1,
        workers=10)
model.train(x1, total_examples=len(x1), epochs=20)
#model.wv.save_word2vec_format('model.bin', binary=True)
model.save("model_twitter")
model.wv.save_word2vec_format('model_twitter.txt', binary=False)
###load
#model = gensim.models.Word2Vec.load("model2.txt")

###find similarity###########
"""
here is an example of finding the similairty between words
"""
model = gensim.models.Word2Vec.load("model_twitter")
print(model.similarity('me', 'you'))
