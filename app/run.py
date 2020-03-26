import json
import plotly
import numpy as np
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# ---------------  global setup data:-------------------
# here are all global environment setups for this app...
# relatie to the directory of this file....
# could be put in a global class or package to be imported
# or transfered to shell environment ... cheap workaround :-)
#-------------------------------------------------------
my_databasename = '../data/DisasterResponse.db'
my_startupdatabase = 'sqlite:///' + my_databasename
my_tablename = 'disastertweets'
my_modelname = '../model/DisasterResponse.pkl'
my_Debug = True
#-------------------------------------------------------

#~ def tokenize(text):
    #~ tokens = word_tokenize(text)
    #~ lemmatizer = WordNetLemmatizer()

    #~ clean_tokens = []
    #~ for tok in tokens:
        #~ clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        #~ clean_tokens.append(clean_tok)

    #~ return clean_tokens

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



#engine = create_engine('sqlite:///../data/YourDatabaseName.db')
engine = create_engine(my_startupdatabase)
df = pd.read_sql_table(my_tablename, engine)

# load model
# model = joblib.load("../models/your_model_name.pkl")
print("shall I load the job?")
if (my_Debug == False):
    print("yes, no lazy debug...")
    model = joblib.load(my_modelname )


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    print("requested index page")
    print(genre_counts)
    genre_names = list(genre_counts.index)
    print(genre_names)
    genre_counts = df.groupby('genre').count()['message']
    
    cname=[]
    cvals=[]

    colnames = df.columns[4:]
        
    for col in colnames:
        cname.append(col)
        cvals.append(df[col].sum())
    print(cname)
    print(cvals)
    

    cnamed=[]
    cvalsd=[]
    for col in colnames:
        cnamed.append(col)
        cvalsd.append(df[df['genre']=='direct'][col].sum())
    print(cnamed)
    print(cvalsd)


    cnamen=[]
    cvalsn=[]
    for col in colnames:
        cnamen.append(col)
        cvalsn.append(df[df['genre']=='news'][col].sum())
    print(cnamen)
    print(cvalsn)


    cnames=[]
    cvalss=[]
    for col in colnames:
        cnames.append(col)
        cvalss.append(df[df['genre']=='social'][col].sum())
    print(cnames)
    print(cvalss)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts.values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cname,
                    y=cvals
                )
            ],
            'layout': {
                'title': 'Number of labels triggered',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label/Tag"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cnamed,
                    y=cvalsd
                )
            ],
            'layout': {
                'title': 'Number of labels triggered if direct contact',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label/Tag"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cnamen,
                    y=cvalsn
                )
            ],
            'layout': {
                'title': 'Number of labels triggered if news',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label/Tag"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cnames,
                    y=cvalss
                )
            ],
            'layout': {
                'title': 'Number of labels triggered if social',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Label/Tag"
                }
            }
        },

 


        {
            'data': [
                Scatter(
                    x=cvals,
                    y=cvalsd,
                    mode='markers',
                    name='direct'
                ),
                Scatter(
                    x=cvals,
                    y=cvalsn,
                    mode='markers',
                    name='news'
                ),
                Scatter(
                    x=cvals,
                    y=cvalss,
                    mode='markers',
                    name='social'
                )
            ],
            'layout': {
                'title': 'Scatter of NOfLabels total to input -direct-',
                'yaxis': {'title': "special Count"},
                'xaxis': {'title': "Total Count"}
            }
        }
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    print ("===========================")
    print(ids)
    print ("===========================")
    print(graphJSON)
    print ("===========================")
    
    # render web page with plotly graphs
    return render_template('master.html', 
                       ids=ids, 
                       graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    print("shall I call the model?")
    if (my_Debug == False):
        print("yes, no lazy init...")
        classification_labels = model.predict([query])[0]
    else:
        print("no, take numpy array...")
        classification_labels = np.array([1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0])
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()