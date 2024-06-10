import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ load messages and categories dataset
    Inputs:
      - messages_filepath = str
      - categories_filepath = str
    Returns:
      - df = merge of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    """ Split categories, transform data into useful dataframe
    Inputs:
        - df = pandas dataframe
    Returns:
        - cleaned df
    """
    categories = df.categories.str.split(';', expand = True) # split strings
    row = categories.iloc[0,:] # get first row
    category_colnames = row.apply(lambda x: x[:len(x) - 2]) # get names without following numeric
    categories.columns = category_colnames # set column names

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    df = df[['id','message','original','genre']] # restructure dataset

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    df = df.drop_duplicates() # drop duplicates


def save_data(df, database_filename):
    engine = create_engine(r'sqlite:///'+database_filename)
    df.to_sql('DisasterMessages', engine, index=False)


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