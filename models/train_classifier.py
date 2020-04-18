import sys

import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """Loads from a database file, cleans dataset
    and returns X/Y arrays and label names.

    Arguments:
    database_filepath -- path of database

    Returns:
    X -- array of messages
    Y -- array of categories
    category_names -- list of category names
    """

    # load data
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categorized', engine)

    category_names = df.iloc[:, 3:].columns
    X = df['message'].values
    Y = df.iloc[:,3:].values

    return X, Y, category_names

def tokenize(text):
    """Returns list of word tokens from clean_text

    Arguments:
    text -- string of text

    Returns:
    clean_tokens -- list of word tokens
    """

    clean_text = re.sub('[^A-Za-z]+', ' ', text)

    tokens = word_tokenize(clean_text)
    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words('english')
    tokens = [t for t in tokens if t not in stop_words]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))]
    )

    # block-commented grid earch section to shorten execution time
    # grid searched parameters are set before model training

    # parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
    #             'vect__max_df': (0.5, 0.75, 1.0),
    #             #'vect__max_features': (None, 5000, 10000),
    #             'tfidf__use_idf': (True, False),
    #             #'clf__estimator__n_estimators': [50, 100, 200],
    #             #'clf__estimator__max_features': ['auto', 'sqrt'],
    #             'clf__estimator__min_samples_split': [2, 5, 10],
    #             #'clf__estimator__bootstrap': [True, False],
    #         }
    #
    # cv = GridSearchCV(pipeline, param_grid=parameters)
    #
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print( "Label: {}".format(category_names[i]), "\n", classification_report( Y_test[:,i], Y_pred[:,i] ) )

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        # set grid searched model parameters
        model.set_params(clf__estimator__min_samples_split = 2,
                         clf__estimator__n_jobs = -1,
                         clf__estimator__verbose = 10,
                         tfidf__use_idf = False,
                         vect__max_df = 0.5,
                         vect__ngram_range = (1, 2))

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
