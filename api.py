from flask import Flask,request,jsonify
from flask_cors import CORS

from bert import QA

app = Flask(__name__)
CORS(app)

model = QA("model")

@app.route("/predict",methods=['POST'])
def predict():
    doc = request.json["documents"]
    q = request.json["question"]

    try:
        # print(doc)
        out = []
        # print(doc)
        for i in doc:
            ans = model.predict(i,q)
            out.append({ans['answer'],ans['confidence']})
        print(out)
        return jsonify({"result":out})
    except Exception as e:
        print(e)
        return jsonify({"result":"Model Failed"})

if __name__ == "__main__":
    app.run('0.0.0.0',port=8000)
