import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import joblib
import pickle
import re


#prepare the dataset
df = pd.read_csv('sent_dataset.csv', encoding='latin-1')
df = df[[df.columns[-1],df.columns[0]]]
df.columns = ['text', 'sentiment']
df = df.dropna()
df.loc[df['sentiment'] == 0, 'sentiment'] = -1
df.loc[df['sentiment'] == 2, 'sentiment'] = 0
df.loc[df['sentiment'] == 4, 'sentiment'] = 1
#print(df)

#clean text in dataset


#Lowercasing all the letters
df['text'] = df['text'].str.lower()
#Removing hashtags and mentions and links
df['text'] = df['text'].replace(to_replace='#[A-Za-z0-9_]+', regex=True, value='')
df['text'] = df['text'].replace(to_replace='@[A-Za-z0-9_]+', regex=True, value='')
df['text'] = df['text'].replace(to_replace=r'http\S+', regex=True, value='')
     
#print(df)

#splitting the dataset into a training and test set
text = df['text'].values
sent = df['sentiment'].values
reviews_train, reviews_test, y_train, y_test = train_test_split(text, sent, test_size=0.2, random_state=1000)

#vectorize the data    
vectorizer = CountVectorizer()
vectorizer.fit(reviews_train)

X_train = vectorizer.transform(reviews_train)
X_test = vectorizer.transform(reviews_test)

#train
classifier = LogisticRegression(solver='lbfgs', max_iter=2000)
classifier.fit(X_train, y_train)

#score
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

y_pred = classifier.predict(X_test)
#Confusion matrix
#cm = confusion_matrix(y_test, y_pred)
#tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#print(tn, fp, fn, tp)

#precision=tp/(tp+fp)
#recall = tp/(tp+fn)
#f1 = (2*tp)/(2*tp+fp+fn)
#print(precision,recall,f1)
'''
#save the model
#create the folder and store it there
joblib.dump(classifier, 'sentiment_model.pkl', compress=9)

# Save the vectorizer
vec_file_en = 'sentiment_vectorizer.pickle'
pickle.dump(vectorizer, open(vec_file_en, 'wb'))


#load the model

classifier = joblib.load('sentiment_model.pkl')
vectorizer = pickle.load(open("sentiment_vectorizer.pickle", 'rb'))
'''
#test on data
text_for_analysis=['today was a good day']

X_new = vectorizer.transform(text_for_analysis)
results = classifier.predict(X_new)
print(results)
results = results.tolist()

results_sum = sum(results)
sentiment_score=""
sntscr = results_sum/len(results)

if ((sntscr > 0.5) and (sntscr <0.75)):
    print("The average sentiment score of the text is: ","%.2f" %sntscr, ", Slightly Positive")
    sentiment_score = "Slightly Positive"
elif (sntscr > 0.75):
    print("The average sentiment score of the text is: ","%.2f" %sntscr, ", Positive")
    sentiment_score = "Positive"
elif ((sntscr > 0.25) and (sntscr < 0.5)):
    print("The average sentiment score of the text is: ","%.2f" %sntscr, ", Slightly Negative")
    sentiment_score = "Slightly Negative"   
elif (sntscr == 0.5):
    print("The average sentiment score of the text is: ","%.2f" %sntscr, ", Neutral")
    sentiment_score = "Neutral"
else:
    print("The average sentiment score of the text is: ","%.2f" %sntscr, ", Negative")
    sentiment_score = "Negative"
