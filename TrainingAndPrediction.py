# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 15:00:00 2021

@author: Prateek Prabhakar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle
import docx

is_model_trained = False
wordVectorizer = None
classifier = None
le_name_mapping = None

class TrainingAndPrediction:
    def train_model_and_predict(self, csv_path, filePath, session):
        global is_model_trained 
        print("is_model_trained:", is_model_trained)
        if is_model_trained == False:
            resumeDataSet = pd.read_csv(csv_path ,encoding='utf-8')
            resumeDataSet['cleaned_resume'] = ''
            resumeDataSet.head()
            resumeDataSet.info()
            
            # to check how many distinct categories are present in the dataset
            #print ("Displaying the distinct categories of resume:\n\n ")
            #print (resumeDataSet['Category'].unique())
            
            # to check distinct categories with individual record count are present in the dataset
            #print ("Displaying the distinct categories of resume and the number of records belonging to each category:\n\n")
            #print (resumeDataSet['Category'].value_counts())
            
            # to plot the categories with record count
            '''
            print("Plot of category-wise record count")
            plt.figure(figsize=(20,5))
            plt.xticks(rotation=90)
            ax=sns.countplot(x="Category", palette=['#432371',"#FAAE7B"], data=resumeDataSet)
            for p in ax.patches:
                ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
            plt.grid()
            '''
            
            
            # to plot percentage wise category details
            '''
            targetCounts = resumeDataSet['Category'].value_counts()
            targetLabels  = resumeDataSet['Category'].unique()
            # Make square figures and axes
            plt.figure(1, figsize=(22,22))
            the_grid = GridSpec(2, 2)
            
            cmap = plt.get_cmap('coolwarm')
            plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY-WISE PERCENTAGE DISTRIBUTION')
        
            source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True)
            plt.show()
            '''
            
            resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: self.clean_resume(x))
            
            # check few records of dataframe
            #print("Cleaned Resume: ")
            #print(resumeDataSet[['Category','cleaned_resume']].head())
            
            
            #create a copy of dataframe
            resumeDataSet_d = resumeDataSet.copy()
            #cleanedSentences = check_most_common_words(resumeDataSet)
            #plot_word_cloud(cleanedSentences)
            
            
            var_mod = ['Category']
            le = LabelEncoder()
            for i in var_mod:
                resumeDataSet[i] = le.fit_transform(resumeDataSet[i])
            global le_name_mapping
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            key_list = list(le_name_mapping.keys())
            val_list = list(le_name_mapping.values())
 

            position = val_list.index(0)
            print(key_list[position])
                
            #check few records of dataframe
            #print(resumeDataSet.head())
            #print("Label Encoded Category and record count")
            #print(resumeDataSet.Category.value_counts())
            
            #matching with the original dataframe copy
            #print("Category and record count")
            #print(resumeDataSet_d.Category.value_counts())
            #remove copy dataframe
            del resumeDataSet_d
            global wordVectorizer
            global classifier
            
            wordVectorizer, classifier = self.vectorize_and_train(resumeDataSet)
            is_model_trained = True
            pred = self.predict_resume_class(filePath)
            
            key_list = list(le_name_mapping.keys())
            val_list = list(le_name_mapping.values())
 

            position = val_list.index(pred[0])
            prediction = key_list[position]
        else:
            pred = self.predict_resume_class(filePath)
            key_list = list(le_name_mapping.keys())
            val_list = list(le_name_mapping.values())
 

            position = val_list.index(pred[0])
            prediction = key_list[position]
            
        return prediction
       
        
    def clean_resume(self, resumeText):
        resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
        resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
        resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
        resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
        resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
        resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
        resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
        return resumeText
    
    
    def check_most_common_words(self, resumeDataSet):
        oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
        totalWords =[]
        Sentences = resumeDataSet['Resume'].values
        cleanedSentences = ""
        for i in range(0,160):
            cleanedText = self.clean_resume(Sentences[i])
            cleanedSentences += cleanedText
            requiredWords = nltk.word_tokenize(cleanedText)
            for word in requiredWords:
                if word not in oneSetOfStopWords and word not in string.punctuation:
                    totalWords.append(word)
        
        wordfreqdist = nltk.FreqDist(totalWords)
        mostcommon = wordfreqdist.most_common(50)
        print("Displaying Top 50 most common words: ")
        print(mostcommon)
        return cleanedSentences
    
        
    def plot_word_cloud(cleanedSentences):
        wc = WordCloud(background_color='white').generate(cleanedSentences)
        plt.figure(figsize=(10,10))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        
    def vectorize_and_train(self, resumeDataSet):
        requiredText = resumeDataSet['cleaned_resume'].values
        requiredTarget = resumeDataSet['Category'].values
        
        #print("Required Text shape: ", requiredText.shape)
        #print(type(requiredText))
    
        #print ("Generate Word Feature started .....")
        
        word_vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                stop_words='english')
        word_vectorizer.fit(requiredText)
        WordFeatures = word_vectorizer.transform(requiredText)
        #print('Word Features Shape: ', WordFeatures.shape)
        #print("Word feature shape",WordFeatures.shape)
    
        #print ("Generate Word Feature completed .....")
    
        X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=42, test_size=0.2,
                                                     shuffle=True, stratify=requiredTarget)
        #print("X train shape: ", X_train.shape)
        #print("X test shape", X_test.shape)
        #print("X test: ", X_test)
        
        classifier = OneVsRestClassifier(KNeighborsClassifier())
        classifier.fit(X_train, y_train)
        #pickle.dump(clf, open(r'G:/ResumeScreening/resumeScreeningModel.sav', 'wb'))
        prediction = classifier.predict(X_test)
        #print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
        #print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
        #print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
        return word_vectorizer, classifier
        
        
    
    def extract_text_from_doc(self, doc_path):
        doc = docx.Document(doc_path)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        return '\n'.join(fullText)
    
    def predict_resume_class(self, filePath):
        extracted_text = self.extract_text_from_doc(filePath)
        #extracted_text = 
        #df = pd.DataFrame()
        Text = extracted_text #resumeDataSet['cleaned_resume'].values[1]
        
        vector = wordVectorizer.transform([Text])
        #print('Word Feature Shape: ', vector.shape)
        #print(WordFeature.shape)
        
        prediction = classifier.predict(vector)
        return prediction
        #print("Predicted for New Resume: ", prediction)
        

if __name__ == "__main__":
    pass