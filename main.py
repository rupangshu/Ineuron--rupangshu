from flask import Flask, request
from flask import Response
from test1 import IneuronPredict
import os
app = Flask(__name__)


@app.route("/predict", methods=['GET'])
def predictRoute():
	message = request.args.get('messageText')
	input = str(message)
	print(input)

	predict = IneuronPredict()
	result = predict.getprediction(input)
	return Response(result)


port = int(os.getenv("PORT"))
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=port)
	#app.run(port=8080)
