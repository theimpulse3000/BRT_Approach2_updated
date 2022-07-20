# This works on dataset given to it
import functionalities
import pandas as pd
import numpy as np
# spacy
import spacy
nlp = spacy.load("nl_core_news_sm")
from spacy import displacy 
# nltk
import re
import nltk
from nltk import tokenize
from nltk.tokenize import word_tokenize 
from operator import itemgetter
import math
#nltk.download('punkt')
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download(['stopwords','wordnet'])
# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import hstack
from sklearn import metrics

def read_resume_dataset(csv_path) :
    df = pd.read_csv(csv_path)
    df = df.reindex(np.random.permutation(df.index))
    data = df.copy().iloc[0:500]
    return data, df

'''Entity ruler helps us add additional rules to highlight various categories 
    within the text, such as skills and job description in our case.'''
def pipeline_newruler_adder(skill_patterns_path) :
    new_ruler = nlp.add_pipe("entity_ruler")
    new_ruler.from_disk(skill_patterns_path)
    return new_ruler

def skills_extract(resume_text) :
    # type of doc - <class 'spacy.tokens.doc.Doc'>
    doc = nlp(resume_text)
    my_set = []
    sub_set = []
    for ent in doc.ents :
        #print(ent.label_)
        #print(type(ent.label_))
        # tokens format -> SKILL|Python
        if ent.label_[0:5] == "SKILL" :
            sub_set.append(ent.text)
            #print(sub_set)
    my_set.append(sub_set)
    return sub_set

def unique_skills(skills) :
    #print(type(skills))
    return list(set(skills))

# cleaning of the text using nltk
def clean_resume_text(data) :
    clean_text = []
    for i in range(data.shape[0]) :
        # regex to remove hyperlinks, special characters, or punctuations.
        resume_text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"', " ", data["Resume_str"].iloc[i])
        # Lowering text
        resume_text = resume_text.lower()
        # Splitting text into array based on space
        resume_text = resume_text.split()
        # Lemmatizing text to its base form for normalizations
        lm = WordNetLemmatizer()
        # removing English stopwords
        resume_text = [lm.lemmatize(word) for word in resume_text if not word in set(stopwords.words("english"))]
        resume_text = " ".join(resume_text)
        # Appending the results into an array.
        clean_text.append(resume_text)
        #print(clean_text)
    return clean_text

#arg 1: data : DataFrame 
#arg 2: clean_text : List  
def modify_resume_csv(data, clean_text) :
    data["clean_resume"] = clean_text
    #print(type(data["clean_resume"].str))
    '''apply() method. This function acts as a map() function in Python. It takes a function as an 
       input and applies this function to an entire DataFrame.'''
    data["skills"] = data["clean_resume"].str.lower().apply(skills_extract)
    data["skills"] = data["skills"].apply(unique_skills)
    return data

def create_job_cat_var(data) :
    job_category = data["Category"].unique()
    job_category = np.append(job_category, "ALL")
    return job_category

def skills_distribution(data, job_cat) :
    total_skills = []
    if job_cat != "ALL" :
        fltr = data[data["Category"] == job_cat]["skills"]
        for i in fltr :
            for j in i :
                total_skills.append(j)
    else :
        fltr = data["skills"]
        for i in fltr :
            for j in i :
                total_skills.append(j)
    return total_skills
        
def most_used_word(data, job_cat) :
    text = ""
    for i in data[data["Category"] == job_cat]["clean_resume"].values :
        text = text + i + " "
    token = re.findall('\w+', text)
    words = []
    for word in token:
        words.append(word)
    nlp_words = nltk.FreqDist(words)
    return nlp_words

# helper function for most_used
def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

# helper function for most_used
def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result

def most_used(resume_text) :
    stop_words = set(stopwords.words('english'))
    doc = resume_text
    total_words = doc.split()
    total_word_length = len(total_words)
    #print(total_word_length)
    total_sentences = tokenize.sent_tokenize(doc)
    total_sent_len = len(total_sentences)
    #print(total_sent_len)
    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1
    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
    #print(tf_score)
    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1
    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())
    #print(idf_score)
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    #print(tf_idf_score)
    return get_top_n(tf_idf_score, 5)

def use_entity_recg(data) :
    resume_text = nlp(data["Resume_str"].iloc[0])
    # this will output in markup
    return displacy.render(resume_text, style = "ent")

def use_entity_recg_for_resume(resume_text) :
    resume_text = nlp(resume_text)
    # this will output in markup
    return displacy.render(resume_text, style = "ent")

def matching_score(input_skills, resume_text) : 
    #print('~~~~~~~~~~~~Starting def matching_score')
    req_skills = input_skills.lower().split(",")
    #print(req_skills)
    #print(resume_text.lower())
    resume_skills = unique_skills(skills_extract(resume_text.lower()))
    #print(resume_skills)
    score = 0
    for i in req_skills:
        if i in resume_skills:
            score = score + 1
    req_skills_len = len(req_skills)
    match = round(score / req_skills_len * 100, 1)
    return match

def custom_NER(data, df, new_ruler) :
    patterns = df.Category.unique()
    for a in patterns :
        new_ruler.add_patterns([{"label" : "Job-Category", "pattern" : a}])
    # options=[{"ents": "Job-Category", "colors": "#ff3232"},{"ents": "SKILL", "colors": "#56c426"}]
    colors = {
        "Job-Category": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
        "SKILL": "linear-gradient(90deg, #9BE15D, #00E3AE)",
        "ORG": "#ffd966",
        "PERSON": "#e06666",
        "GPE": "#9fc5e8",
        "DATE": "#c27ba0",
        "ORDINAL": "#674ea7",
        "PRODUCT": "#f9cb9c",
    }
    options = {
        "ents": [
            "Job-Category",
            "SKILL",
            "ORG",
            "PERSON",
            "GPE",
            "DATE",
            "ORDINAL",
            "PRODUCT",
        ],
        "colors": colors,
    }
    sent = nlp(data["Resume_str"].iloc[5])
    return displacy.render(sent, style="ent", options=options)

'''LabelEncoder can be used to normalize labels. 
   It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.'''
'''we will encode the ‘Category’ column using LabelEncoding. Even though the ‘Category’ column is ‘Nominal’ data we are using 
   LabelEncong because the ‘Category’ column is our ‘target’ column. By performing LabelEncoding each category will become a class 
   and we will be building a multiclass classification model'''
def label_encoding(data) :
    label_encoder = LabelEncoder()
    data["Category"] = label_encoder.fit_transform(data["Category"])
    return data

'''Vectorization or word embedding is the process of converting text data to numerical vectors'''
def word_vectorizer(data) :
    required_text = data["clean_resume"].values
    required_target = data["Category"].values
    # Transforms text to feature vectors that can be used as input to estimator
    # Sublinear tf-scaling is modification of term frequency, which calculates weight
    wordVectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
    '''maps each single token to a position in the output matrix. Fitting on the training set and transforming on the training and test set assures that, given a word, 
    the word is correctly always mapped on the same column, both in the training and test set. '''
    wordVectorizer.fit(required_text)
    #transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text
    wordFeatures = wordVectorizer.transform(required_text)
    # print("Feature completed ")
    # divide in train dataset and test dataset
    X_train,X_test,y_train,y_test = train_test_split(wordFeatures,required_target,random_state=0, test_size=0.2, shuffle=True, stratify=required_target)
    return X_train, X_test, y_train, y_test

def predict_classifier(X_train, X_test, y_train, y_test) :
    # when we want to do multiclass or multilabel classification and it's strategy consists of fitting one classifier per class. For each classifier, the class is fitted against all the other classes.
    # The K in the name of this classifier represents the k nearest neighbors, where k is an integer value specified by the user. Generally it is 5.
    classifier = OneVsRestClassifier(KNeighborsClassifier())
    # train the model
    classifier.fit(X_train, y_train)
    # prediction for each test instance
    predict = classifier.predict(X_test)
    training_accuracy = classifier.score(X_train, y_train)
    testing_accuracy = classifier.score(X_test, y_test)
    # precision    recall  f1-score   support
    report_classifier = metrics.classification_report(y_test, predict)
    return training_accuracy, testing_accuracy, report_classifier