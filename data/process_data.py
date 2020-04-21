import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads and merges messages and categories dataframes.

    Arguments:
    messages_filepath -- path of messages dataframe
    categories_filepath -- path of categories dataframe

    Returns:
    DataFrame object
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='inner')

    return df

def clean_data(df):
    """Cleans merged dataframe.

    Arguments:
    df -- dataframe to clean

    Returns:
    DataFrame object
    """

    categories = df['categories'].str.split(';', expand=True)

    # create a dataframe of the 36 individual category columns
    row = categories.iloc[:1,:].values[0]
    category_colnames = [cat[:-2] for cat in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories],axis=1)

    # drop column with missing data
    df.dropna(axis=1, inplace=True)

    # drop rows where category dummy > 1
    category_names = df.iloc[:, 3:].columns
    df_clean = df[~(df[category_names]>1).any(axis=1)].copy()

    df_clean.drop_duplicates(inplace=True)

    return df_clean

def save_data(df, database_filename):
    """Stores a DataFrame object as a SQLite file.

    Arguments:
    df -- dataframe to clean
    database_filename -- path of SQLite file
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_categorized', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)

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
