import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """ load data from database
    Inputs:
    - database_filepath = str
    Return:
    - X = model inputs
    - y = output to predict 
    """
    table_name = 'DisasterResponseClean'
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql(table_name, engine)
    X = df.message.values
    y = df.drop(columns = ['id','genre', 'message', 'original'])
    category_names = y.columns.tolist()
    return X, y, category_names


def tokenize(text):
    """ process text into valuable tokens for ML process
    IMPUTS: text: A text object in string format
    OUTPUTS: 'clean_tokens' string
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text) 
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "")

    # tokenize text
    tokens = word_tokenize(text) 
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(grid_search = False):
    """ build pipeline """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]) 

    if grid_search == True:
        print('Searching for best parameters...')
        parameters = {
            'clf__estimator__n_estimators': [5, 10]
            , 'clf__estimator__min_samples_split': [2, 3]
        }
        pipeline = GridSearchCV(pipeline, param_grid = parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """ test model and evaluate parameters to determine accurate predictions
    Inputs:
        - model: trained model
        - X_test: subset of data
        - Y_test: subset of categories data
        - category_names: paramters of interest
    Output:
        None, a report is printed
    """
    y_pred = model.predict(X_test)
    report = classification_report(Y_test, y_pred, target_names=category_names)
    print(report)


def save_model(model, model_filepath):
    """ save model and export as pickle
    Input:
        - model to save (model object)
        - filepath to save file (str)
    Output:
        - none, model is saved
    """
    with open(str(model_filepath), 'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()