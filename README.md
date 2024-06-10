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

### Dependencies:
pandas == 2.2.2

numpy == 1.26

sqlalchemy == 2.0.30

nltk == 3.8.1

sklearn == 1.5.0

pickle == 3.12.4

### Acknowledgements:

Udacity Lectures (udacity.com)

Thanks to Appen for supplying the data.

License: MIT License