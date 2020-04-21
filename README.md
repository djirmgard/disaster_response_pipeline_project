# Disaster Response Pipeline Project

### Project Summary:
In this project I created a machine learning model that classifies messages sent after an disaster to categories such as medical aid, food request, etc.
The data set contains real messages that were sent during disaster events. The model can be accessed via a Flask webapp that visualizes the results and interesting facts about the data itself.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3001/
