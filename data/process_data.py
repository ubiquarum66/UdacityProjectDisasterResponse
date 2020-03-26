import sys
# import libraries
import re
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load the csv file containing all the tweets , in english, original, and with ID given ,
    # also there is a column genre, given the way the tweets were received
    # load messages dataset
    print("#-----------try loading data")
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    print(messages.head(2))
    # load the csv file containing all labesls assigned to the tweets externally (expert knowledge)
    # identified by id and coded by a concat string of ever the same sequence, with bin info added as suffix to the topic name
    # e.g. blue-0 = not blue,  blue-1 = is blue
    # load categories dataset
    categories =  pd.read_csv(categories_filepath, encoding='latin-1')
    print(categories.head(2))
    #merge data according to ID .....
    df = messages.merge(categories, on='id')
    print(df.head(2))
    print("#------------data loaded successfully")
    return df
    
# here is a small helper function to separate the catagorie names from the value suffix -0,-1
def namestring(s):
    return s[0:-2]


def clean_data(df):
    print("#------------data cleansing start")
    # cleaning here is multipass:
    print ("shape of DataFrame before drop:")
    print(df.shape)
    
# ===1.) create category columns: blue-1;gree-0;red-1 shall be ['blue'] = 1,. ['green'] = 0, ['red]=1 for each row.

    # thus, entries have to be split, and each part creates a new column
    # assumption: all rows contain string in same seuence, all members are set each row.
    # test: later, in sql datbase, check for null entries.
    
    # create a dataframe of the 36 individual category columns (but keep the content in each string - like:
    #debug helper:
    collist = df.categories.str.split(';',expand=True)
    print (collist.shape, df.shape)
    #doit:
    categories = df.categories.str.split(';',expand=True)
    print(categories.head(2))
    
    # to rename the columns, one example string content of the columns is analyzed, and 
    # the name of topic has to be freed from the content markers -0,-1
    # therefor, select the first row of the categories dataframe
    row = categories.iloc[0,:]
    print(row)
    # use this row to extract a list of new column names for categories - I apply a lambda function of the helper above to each member of row Series
    category_colnames = row.apply(lambda x: namestring(x))
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()

# ===2.)  Convert category values to just numbers 0 or 1.
    # - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    #    For example, `related-0` becomes `0`, `related-1` becomes `1`.

    for column in categories:
        #print(categories.head(20))
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    categories.head()


# ===3.) . Replace `categories` column in `df` with new category columns.
    # - Up to now, no filtering, row size of df is row size of categories, thus we concatenate columnwise to enter the information
    # -and drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    df = pd.concat([df , categories], axis=1)
    print(df.head(2))


# ===4.)  Remove duplicates.
    # - Check how many duplicates are in this dataset.
    # - Drop the duplicates.
    # - Confirm duplicates were removed.
    # debug: to not have to run all above, if failing ....
    #    df.to_csv("mycsv.csv", sep='\t')
    #    df = pd.read_csv("mycsv.csv", sep='\t')
    # assumptions: duplicate of id key is sufficient to trace duplicates....
    # check number of duplicates (here optimizing by duplicate().sum() possible, not done yet....
    udf = df.drop_duplicates(['id'])
    print ("number of duplicates before drop:")
    print(df.shape[0]- udf.shape[0])
    #
    df = df.drop_duplicates(['id'])
    #
    print (df.shape)
    udf = df.drop_duplicates(['id'])
    print ("number of duplicates after drop:")
    print(df.shape[0]- udf.shape[0])
    print ("cleansing finished... shape of result:")
    print (df.shape)
    print("#------------data cleansed successfully")
    return df
    
def save_data(df, database_filename):

# ===1.)   Save the clean dataset into an sqlite database.
    # the schema of result has to fit the assumptions in the flask app, and in the ml pipeline and model py: 
    # flask: genre column is actively searched - placing no problem
    # flask: tweet categories are indexed with 4:, thus they should start at column nr. 4, counting start with 0
    # schema of result is: (checked with sqlite shell cli:)
    
    startupdatabase = 'sqlite:///' + database_filename
    engine = create_engine(startupdatabase)
    df.to_sql('disastertweets', engine, index=False)

    #---------------result as of sqlite cli: ---------------------------------------------
    #~ count(*)
    #~ --------
    #~ 26180
    #
    #~ CREATE TABLE disastertweets (
    #~ id BIGINT, 
    #~ message TEXT, 
    #~ original TEXT, 
    #~ genre TEXT, 
    #~ related BIGINT, 
    #~ ....
    #~ direct_report BIGINT
    #~ );
    #
    #~ Statistics Genre (Group By):

    #~ genre|count(*)
    #~ --------------
    #~ direct|10747
    #~ news|13039
    #~ social|2394
    
    # onle line (shortened ....): 
    #
    #~ 202|?? port au prince ?? and ....| ....gouvenman an ak d entenasyonal. Mesi, BonDye beni Ayiti. |direct|     
    #~                                            1|1|0|1|0|0|0|0|0|0|0|1|0|0|0|0|0|0|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0
    #------------------------------------------------------------

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()