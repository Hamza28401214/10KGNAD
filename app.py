from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from preprocess_data import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/api", methods=["GET", "POST"])
def api():
    if request.method == "POST":
        SVC = pickle.load(open('models//final_model1.pkl', 'rb'))
        text = request.form["text"]
        new_text = text
        response = ''
        # mapper = {0: 'Etat', 1: 'Inland', 2: 'International', 3: 'Kultur', 4: 'Panorama', 5: 'Sport', 6: 'Web',
        #           7: 'Wirtschaft', 8: 'Wissenschaft'}
        new_text = preprocessing(new_text)
        ## vocabulary fitting###################################"
        cd = pd.read_csv('df1.csv')
        X_train, X_test, y_train, y_test = train_test_split(cd['text'], cd['class'], random_state=0)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        clf = LinearSVC().fit(X_train_tfidf, y_train)
        ####
        ################ Predict using our loaded model ##############
        mapped_pred = SVC.predict(count_vect.transform([new_text]))
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
