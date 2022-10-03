from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

# 1. Crea un endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'



# 2. Crea un endpoint que reentrene de nuevo el modelo con los datos disponibles en la carpeta data, que guarde ese modelo reentrenado, devolviendo en la respuesta la media del MAE de un cross validation con el nuevo modelo
@app.route('/retrain', methods=['PUT'])
def retrain():
    connection = sqlite3.connect('data/advertising.db')
    cursor = connection.cursor()
    select_books = "SELECT * FROM table_1"
    result = cursor.execute(select_books).fetchall()
    names = [description[0] for description in cursor.description]
    connection.close()
    
    df = pd.DataFrame(result, columns=names)
    
    X = df.drop(columns=['sales'])
    y = df['sales']

    model = pickle.load(open('data/advertising_model','rb'))
    model.fit(X,y)
    pickle.dump(model, open('data/advertising_model_v1','wb'))

    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')

    return "New model retrained and saved as advertising_model_v1. The results of MAE with cross validation of 10 folds is: " + str(abs(round(scores.mean(),2)))

@app.route('/ingest_data', methods=["POST"])
def new_data():
    tv = int(request.args.get('tv', None))
    radio = int(request.args.get('radio', None))
    newspaper = int(request.args.get('newspaper', None))
    sales = int(request.args.get('sales', None))

    connection = sqlite3.connect('data/advertising.db')
    cursor = connection.cursor()
    query = "INSERT INTO table_1 VALUES (?, ? , ?, ?) "
    cursor.execute(query, (tv,radio,newspaper,sales)).fetchall()
    query = "SELECT * FROM table_1"
    result = cursor.execute(query).fetchall()
    connection.commit()
    connection.close()
    return jsonify(result)



app.run()