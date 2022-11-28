from flask import Flask , jsonify, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
print("model is loaded")

app = Flask (__name__)


@app.route('/',methods = ['GET'])
def index():

    diagnostic_result = str(request.args['diagnostic_result'])
    radius = int(request.args['radius'])
    texture = int(request.args['texture'])
    perimeter = int(request.args['perimeter'])
    area = int(request.args['area'])
    smoothness = int(request.args['smoothness'])
    compactness = int(request.args['compactness'])
    symmetry = int(request.args['symmetry'])
    fractal_dimension = int(request.args['fractal_dimension'])

    pred = model.predict(np.array([diagnostic_result,radius,texture,perimeter,area,smoothness,compactness,symmetry,fractal_dimension]).reshape(1,-1))

    return jsonify(prediction = str(pred))


if __name__ =="__main__":
    app.run(debug=True)