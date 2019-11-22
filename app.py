from flask import Flask, redirect, request, url_for
from prometheus_client import make_wsgi_app, Summary, Counter, Histogram
from werkzeug.wsgi import DispatcherMiddleware

import base64
from pickle import load as cPload
from pickle import loads as cPloads
import numpy
import cloudpickle
import sys
import os
import pandas as pd

app = Flask(__name__)

METRICS_PREFIX = os.getenv("S2I_APP_METRICS_PREFIX", "nachlass")

PREDICTION_TIME = Summary('%s_processing_seconds' % METRICS_PREFIX, 'Time spent processing predictions')

app.model = None

@app.route('/')
def index():
  return "Make a prediction by POSTing to /predict"

@app.route('/predict', methods=['POST'])
@PREDICTION_TIME.time()
def predict():
    import json
    if 'json_args' in request.form:
      args = pd.read_json(request.form['json_args'])
      if len(args.columns) == 1 and len(args.values) > 1:
          # convert to series
          args = args.squeeze()
      else:
          args = [args.squeeze()]
    else:
      args = cPloads(base64.b64decode(request.form['args']))
    try:
        predictions = app.model.predict(args)
        for v in predictions:
            # FIXME:  this assumes a classifier!  we want to be more flexible
            app.observe_prediction(v)
        return json.dumps(predictions.tolist())
    except ValueError as ve:
        return str(ve)
    except Exception as e:
        return str(e)

def classifier_prediction_recorder(p):
    def record(v):
        p.labels(v).inc()
    return record

def regressor_prediction_recorder(p):
    def record(v):
        p.observe(v)
    return record


try:
    import json
    from sklearn.pipeline import Pipeline
    app.model = Pipeline([(k, cPload(open(v, "rb"))) for k, v in json.load(open("stages.json", "r"))])
    if app.model.steps[-1][1]._estimator_type == 'classifier':
        pm = Counter('%s_predictions_total' % METRICS_PREFIX, 'Total predictions for a given label', ['value'])
        app.observe_prediction = classifier_prediction_recorder(pm)
    elif app.model.steps[-1][1]._estimator_type == 'regressor':
        pm = Histogram("%s_predictions" % METRICS_PREFIX, "Prediction values for this pipeline")
        app.observe_prediction = regressor_prediction_recorder(pm)
        
      
except Exception as e:
    print(str(e))
    sys.exit()

app_dispatch = DispatcherMiddleware(app, {
    '/metrics': make_wsgi_app()
})

if __name__ == "__main__":
    app.logger.setLevel(0)
    app.run(host='0.0.0.0', port=8080)

