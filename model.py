import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def model_prediction():
	df= pd.read_csv("nlp_model_sentiment_heroku_test.csv", encoding="latin-1")
	# df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['sentiment'] = df['sentiment'].map({'negative': 0, 'positive': 1})
	X = df['text']
	y = df['sentiment']

	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	pickle.dump(cv, open('text_tranform.pkl', 'wb'))


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	#Naive Bayes Classifier

	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	filename = 'nlp_model.pkl'
	pickle.dump(clf, open(filename, 'wb'))

	# # Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

if __name__ == "__main__":
	model_prediction()