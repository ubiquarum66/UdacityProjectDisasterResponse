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
    messages = pd.read_csv(messages_filepath, categories_filepath,encoding='latin-1')
    print(messages.head(2))
    # load the csv file containing all labesls assigned to the tweets externally (expert knowledge)
    # identified by id and coded by a concat string of ever the same sequence, with bin info added as suffix to the toüpic name
    # e.g. blue-0 = not blue,  blue-1 = is blue
    # load categories dataset
    categories =  pd.read_csv(messages_filepath, categories_filepath, encoding='latin-1')
    print(categories.head(2))
    #merge data according to ID .....
    df = messages.merge(categories, on='id')
    print(df.head(2))
    print("#------------data loaded successfully")
    return df
    

def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


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