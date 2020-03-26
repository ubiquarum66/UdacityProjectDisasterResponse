import sys

#==========================================================
# basic imports
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
# natural language imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
# sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib
#==========================================================
# globals
databasename = '../data/DisasterResponse.db'
startupdatabase = 'sqlite:///' + databasename
tablename = 'disastertweets'
modelname = '../model/DisasterResponse.pkl'
optmodelname = '../model/DisasterResponseOpt.pkl'
#==========================================================


def load_data(database_filepath):
    startupdatabase = 'sqlite:///' + database_filepath
    df = pd.read_sql_table(tablename, startupdatabase)
    df.head(3)
    
    X = df.message.values
    y = df.loc[:,'related':]
    category_names = df.columns['related':]
    
    print (df.shape)
    print (X.shape)
    print (y.shape)
    
    return X,y, category_names


def tokenize(text):
    #remove punctuation signs , just leave letters and numbers
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    #remove english stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    #now lemmatize by nltk
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # create a pipeline of vectorizer, tfdif trafo, and a randomforest - based multi-o-clasifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline;
    

def optimize_model(model, X_train, y_train)
    model.get_params().keys()
    # create a pipeline grid ... grid only 1x1 as computer is so slow ...:-(
    parameters = {
#            'clf__estimator__n_estimators': [50, 100, 200],
#            'clf__estimator__min_samples_split': [2, 3, 4],
            'clf__estimator__n_estimators': [80],
            'clf__estimator__min_samples_split': [2],
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train, y_train)    
    
    return cv;
    



def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    y_pred = pipeline.predict(X_test)
    #debug: show the columns ....
    print (y_test.loc[:,'related'].values)
    print (y_pred[:,0])
    for s in category_names:
        u = classification_report(y_test.loc[:,s].values,y_pred[:,0])
        print(s, u)



def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    print("#----------------------- pickle export done", model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        #introduced second name, as learning optimizing is so slow..
        # ....thus use first model for flask    
        opt_model_filepath = model_filepath + '_opt.pkl'
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Not-Optimized Trained model saved!')

        print('Optimize Model...')
        opt_model = optimize_model(model, X_train, Y_train)
        
        print('Evaluating opt model...')
        evaluate_model(opt_model, X_test, Y_test, category_names)

        print('Saving opt. model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, opt_model_filepath)
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()