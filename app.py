from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import redirect
from flask import url_for
from preprocess_data import *
import pickle
app = Flask(__name__)


XGB_from_pickle = pickle.load(open('models//model1.pkl', 'rb'))
@app.route('/')
def index():
    return render_template("index.html")


@app.route("/api", methods=["GET", "POST"])
def api():
    if request.method == "POST":
        text = request.form["text"]
        new_text = text
        response = ''
        mapper = {0: 'Etat', 1: 'Inland', 2: 'International', 3: 'Kultur', 4: 'Panorama', 5: 'Sport', 6: 'Web',
                  7: 'Wirtschaft', 8: 'Wissenschaft'}
        new_text = preprocessing(new_text)
        vector = vectorize(new_text)
        padded = padd_sequence(vector)
        predictions = XGB_from_pickle.predict(padded)
        print('1111111111',predictions)
        mapped_pred = [mapper[k] for k in predictions]
        print(mapped_pred)
        if (mapped_pred):
            response = response + str(mapped_pred[0])
            msg = "This text is classified as {}".format(response)
            return render_template("index.html", msg=msg)
        else:
            print("sorry")
            msg = "sorry your text cannot be treated by our model"
            return render_template("index.html", msg=msg)
    else:
        return redirect(url_for("html"))


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
