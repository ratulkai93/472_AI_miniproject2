"""
Goal: NER + Sentiment Analysis

Input: SpaCy, sciKit Learn, AP text snippet, aFinn sentiment lexicon, AP text to analyze: APonTrump
(Trump suit against Clinton part of longtime legal strategy. By JILL COLVIN, ERIC TUCKER and BERNARD
CONDON, 30.3.2022)

Output: NER annotation, selected dependency graphs, sentiment labeling of named entities, analysis

Description: Working from Lab10, Question 4 and Lab 11, Question 3, run the text snippet through SpaCy
for preprocessing and obtain the NER and the dependency graphs for each sentence. You now have certain token
(sequences) labelled as a named entity and typed as a person or an institution. You also have the dependency
graph, that connects each named entity with words in the graph (each token has exactly one governor. Some
nodes have one or more dependents, some have none.
Perform sentiment analysis with aFinn (https://github.com/fnielsen/afinn) on all sentences. You now have
a prediction whether each sentence is neutral, positive or negative. Through lookup in the aFinn lexicon, you
also know the sentiment value of each word (if it is not present, mark the word as neutral, i.e. not carrying
sentiment).~

Put all tokens and all available features (NE?, NEtype, Governor, ListofDependants, SentimentValueofToken, SentimentValueofSentence) into a table T1 for classification. 
For a second experiment, put only named entities in table T2 (no other tokens) and reduce the set of features to (NEtype, Governor, ListofDependants,SentimentValueofToken, SentimentValueofSentence).

Perform a k-means clustering on both input tables. Experiment to find a good value for k.
Discuss the output. Note that A2 will ask you for more in-depth analysis, here, a brief discussion of the
quality of the obtained clusters and a justification of your choice of k will suffice.

Deliverables: Submit your well documented code and Project Report in Moodle.
For demonstrating your work in your submission, use the text snippet S1 (when asked): U.S. intelligence
agencies concluded in January 2017 that Russia mounted a far-ranging influence campaign aimed at helping
Trump beat Clinton. And the bipartisan Senate Intelligence Committee, after three years of investigation, affirmed
those conclusions, saying intelligence officials had specific information that Russia preferred Trump and that
Russian President Vladimir Putin had “approved and directed aspects” of the Kremlin’s influence campaign
"""
from cProfile import label
from filecmp import cmp
import spacy
import nltk
from nltk import word_tokenize
from spacy import displacy
from afinn import Afinn
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#nltk.download('punkt')
nlp= spacy.load("en_core_web_sm")
#text= "U.S. intelligence agencies concluded in January 2017 that Russia mounted a far-ranging influence campaign aimed at helping Trump beat Clinton. And the bipartisan Senate Intelligence Committee, after three years of investigation, affirmed those conclusions, saying intelligence officials had specific information that Russia preferred Trump and that Russian President Vladimir Putin had “approved and directed aspects” of the Kremlin’s influence campaign."
text= open("APonTrump.txt").read()

#--------NER + dependency graph------------------
doc = nlp(text)

#sentence split
# for sent in doc.sents: #for each sentences in doc
#   print(sent,"\n")

#Word/token split
# for sent in doc.sents: 
#   for token in sent: #for each token/words in the sentences
#     print(token.text,"---",token.tag_,"---", token.ent_type_, "---", token.dep_) #print the word, their entity type and dependency


# #for visualizing sentence dependency breakdown
# sentence_spans = list(doc.sents)
# displacy.serve(sentence_spans, style="dep")
# displacy.serve(sentence_spans, style="ent")

#---------------Sentiment analysis/kmeans---------------------
af= Afinn()
ne=0
#i=1
t=[]
tt=[]
def checkScore(score):
    if(score>0):
        return "positive"
    elif score==0:
        return "neutral"
    elif score<0:
        return "negative"

for sent in doc.sents:
    for token in sent:
        tokenArray=[]
        ttArray=[]
        sentence_score=af.score(sent.text)
        token_score=af.score(token.text)
        if(token.ent_type_):
            ne=1
        else:
            ne=0
        #print(f"token# {i}: {token.text} --> NE?: {ne} --> entity type:{token.ent_type_} --> governor: {token.head} --> dependency: {token.dep_} --> SentimentValueOfToken: {token_score} --> SentimentValueOfSentence: {sentence_score}")
        #i+=1
        
        #for T1--------------------
        tokenArray.append(token.text)
        tokenArray.append(ne)
        tokenArray.append(token.ent_type_)
        tokenArray.append(token.head)
        tokenArray.append(checkScore(token_score))
        tokenArray.append(checkScore(sentence_score))
        #print(tokenArray)
        t.append(tokenArray)
        #forT2------------------------
        if(ne==1):
          ttArray.append(token.ent_type_)
          ttArray.append(token.head)
          ttArray.append(checkScore(token_score))
          ttArray.append(checkScore(sentence_score))
          tt.append(ttArray)
            

t1=np.asarray(t)
#print(t1)
t2=np.asarray(tt)
#print(t2)

le = preprocessing.LabelEncoder()

X=t1[: , 0:6]
X[:, 0] = le.fit_transform(X[:, 0])#transforms the column 0 and so on
X[:, 2] = le.fit_transform(X[:, 2])
X[:, 3] = le.fit_transform(X[:, 3])
X[:, 4] = le.fit_transform(X[:, 4])
X[:, 5] = le.fit_transform(X[:, 5])

Y=t2[: , 0:4]
Y[:, 0] = le.fit_transform(Y[:, 0])
Y[:, 1] = le.fit_transform(Y[:, 1])
Y[:, 2] = le.fit_transform(Y[:, 2])
Y[:, 3] = le.fit_transform(Y[:, 3])

#print(X)
print(Y)

#for T1
kmeansT1 = KMeans(init='random', n_clusters=3, n_init=10) 
kmeansT1.fit(X)
t1Labels=kmeansT1.labels_
t1center=kmeansT1.cluster_centers_
#print(kmeansT1.labels_)
# print(kmeansT1.cluster_centers_)

#elbow method analysis
# sse=[]
# list_k=list(range(1,10))
# for k in list_k:
#     km = KMeans(n_clusters=k)
#     km.fit(X) #choose X or Y
#     sse.append(km.inertia_)
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse, '-o')
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance');

#for T2
kmeansT2 = KMeans(init='random', n_clusters=2, n_init=10) 
kmeansT2.fit(Y)
t2Labels=kmeansT2.labels_
t2center=kmeansT2.cluster_centers_
#print(kmeansT2.labels_)
# print(kmeansT2.cluster_centers_)

#for T1 graph
# plt.scatter(x=X[:,0], y=X[:,1])
# plt.scatter(x=t1center[:, 0],y=t1center[:, 1], c='red',s=200, alpha=0.5,label='centroid')

#for T2 graph
plt.scatter(x=Y[:,0], y=Y[:,1])
plt.scatter(t2center[:, 0], t2center[:, 1], c='red',s=200, alpha=0.5,label='centroid')

plt.grid()
plt.show()