from flask import Flask,request,render_template
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Load the sentiment analysis model and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

app = Flask(__name__)

stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')

def preprocessing(text):
    text = re.sub('<[^>]*>', ' ', text)
    emojis = emoji_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        comments = request.form.get('comment')

        # Preprocess the comment
        preprocessed_comments = preprocessing(comments)

        # Transform the preprocessed comment into a feature vector
        comment_vector = tfidf.transform([preprocessed_comments])

        # Predict the sentiment
        prediction = clf.predict(comment_vector)[0]

        return render_template('index.html', sentiment = prediction)

    return render_template('index.html')

if __name__ =="__main__":
    app.run(debug=True)
