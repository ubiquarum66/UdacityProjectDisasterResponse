# Udacity Disaster Response Pipeline Project


## Homework and Report Repository for Udacity Data Scientist Nanodegree

Table of Contents:
+ Report and Homework Project 2

** Report and Homework Project 2 **

### 1. Installations

The jupyter notebooks for project preparation are stored in the notebooks folder
of gitHub [project repository]( https://github.com/ubiquarum66/UdacityProjectDisasterResponse)
for Project 2.

The project is set up as described in the readme of Udacity:


It is tested to run with Python 3.7 and Anaconda with installed packages Pandas, Numpy, plotly, flask, scikit-learn and nltk.
In addition, easy sql is made possible by sqlAlchemy.

Some png Pictures as screenshots from the flask page are shown here:

+ Overview with 5 fact summary plots: 
![main page: overview diagrams of the measurement contants and chapters](overviewpage.png)

+ Query page using the learned model to classify queries
![query page: check model](querypage.png)

### 2. Project Motivation

The applidcation here represents the Project 2 of Data Science Nanodegree Course. 

#### Conceptual Context 

The idea is to bundle and classify 
queries in case of one or multiple disaster, coming on different input channels as short text notes.

The resulting classification might help to direct officials or NGO helpers of different capabilit 
to scenes where they are really needed.


A little bit output and interpretation will (hopefully) be found soon at [my GitHub Blog Page Project 2](https://ubiquarum66.github.io/).

#### Structural Concept: 

ETL:

+ load_data(messages_filepath, categories_filepath) get the csv files as pandas Dataframes
+ clean_data(df) merge the two table by a join with common ID, check for duplicates, remove duplicates
+ spread the one string representation of 36 category tags into 36 binary columns.
+ save_data(df, database_filepath) put it allm into a sqlite3 database file in sql format 

ML:

+ load_data(database_filepath)  gets database sql --> Dataframe, creates test and train data
+ provide an tokenizer that will remove puntuation, split into tokens and lemmatize tokens (using re and nltk)
+ build_model() creates  pipeline of vectorizer and MultiOutputClassifier and RandomForestClassifier
+ model is trained.... and evaluation results to stdout....
+ model is saved !!! Due to slow computer optimizing took to long... use flask with non opt model ...!!!
+ optimize_model(model, X_train, Y_train) via GridSearch , two RndomForest parameters are spnned to find an optimuum. 
    + This was set as an extra call, as it took to long at my site.
+ and evaluation goes to stdout
+ opt. model is saved !!! Due to slow computer, I never reached this in test, sorry!

FLASK:

FLASK side (backend) - run.py 
+ provide the same tokenizer, as the model will call it
+ load databaseand model 

plotly side (frontend) (templates...html), 
+ create basic layout, with 
+ form to submit and 
+ carry the ginger and flask induced variables and scripts to provide the plotly functions with the necessary backend information.


### 3. File Descriptions

+ Subdirs
    + app: Flask app for visual web server functionality
        + app\run.py   main starter and debugging-server for web application
        + app\templates plotly subdir templates, instrumented via ginger and flask
            + app\templates\go.html
            + app\templates\master.html

    + data: raw (csv) data and transformed sql datbase, as well as python transformer module
        + data\disaster_categories.csv  raw data - tagging labels provided by figure eight in a text coded encoding
        + data\disaster_messages.csv  raw data of tweets , as tagged by figure eight, messages by ID, genre of channel
        + data\process_data.py python module (ETL pipeline) to transfer and unif the two csv to a sql database
            + data\ExampleDisasterResponse.db example for resulting sqlite3 sql database
            + data\testresultsdatabase.txt sqlite3 shell cli extracted schema of created database
            

    + models, pickle of trained model, as well as python trainer module
        + models\train_classifier.py (ML Pipeline) ....

    + notebooks intermediate notebbbooks to prepare for transformer and trainer 
        + (not up to date , as jump to py took place before ipynb was o.k.)


### 4. How to Interact with your project

~~~~
    # Disaster Response Pipeline Project

    ### Instructions:

    1. Run the following commands in the project's root directory to set up your database and model.

        - To run ETL pipeline that cleans data and stores in database
            `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        - To run ML pipeline that trains classifier and saves
            `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

    2. Run the following command in the app's directory to run your web app.
        `python run.py`

    3. Go to http://0.0.0.0:3001/
~~~~
    
** There is one inconsistency here in the tasks formulation, as the name of the model is to be freely given in the pipeline, but hardcoded in FLASK! **

### 5. Licensing, Authors, Acknowledgements, etc.

For establishing the Github Blog Post, I have to thank Barry Clark for his Jekyll Now explanations and templates.
Google and Stackoverflow -- as usual -- helped enormeosly to find -- for me -- tricky ML and Python hacks.
As this work is part of Udacity Nanodegree , big parts of this work where provided as part of the task.


