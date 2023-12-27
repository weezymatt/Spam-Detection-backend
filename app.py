# Dependencies (FastAPI)
import uvicorn
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Dependencies 2 (Processing Message)
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

app = FastAPI(title="Ham or Spam API", description="API to predict if SMS is spam")

model = joblib.load("model/finalized_model.sav")

vectorizer = joblib.load("model/vectorizer.sav")

class request_body(BaseModel):
	message   : str # A free service for you ONLY!! Please click on the link now! // String value

def process_msg(msg):
    """
    Replace email address with 'email'
    Replace URLS with 'http'
    Replace currency symbols with 'moneysymb'
    Replace phone numbers with 'phonenumb'
    Replace numbers with 'numb'
    """
    ps = PorterStemmer()
    clean = []
    cleaned_msg = msg
    cleaned_msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'email', cleaned_msg)
    cleaned_msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'https', cleaned_msg)
    cleaned_msg = re.sub('£|\$', 'moneysymb', cleaned_msg)
    cleaned_msg = re.sub('\b(?:\+?(\d{1,3})\s?)?[\-(.]?\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumb', cleaned_msg)
    cleaned_msg = re.sub('\d+(\.\d+)?','numb', cleaned_msg)
    cleaned_msg = re.sub('[^\w\d\s]', ' ', cleaned_msg)
    cleaned_msg = cleaned_msg.lower()

    tokenized_msg = cleaned_msg.split()
    stemmed_msg = [ps.stem(word) for word in tokenized_msg if word not in set(stopwords.words('english'))]
    final_msg = ' '.join(stemmed_msg)
   
    clean.append(final_msg)  
    clean_input = vectorizer.transform(clean)

    return clean_input

@app.get('/')
def Welcome():
	return{'message': 'Welcome to the Spam classifier API!'}

@app.post('/api_predict')
def classify_msg(msg : request_body):
	
	# Check if the message exists
	if (not (msg.message)):
		raise HTTPException(status_code=400, detail="Please provide a valid message")

	# Process the message to fit with the model
	dense = process_msg(msg.message)

	# classification results
	label = model.predict(dense)[0]
	# proba = model.predict_proba(dense) // check again after test

	# extract the corresponding information
	if label == 0:
		return {'Answer': "This is a Ham email!"}
	else:
		return {'Answer': "This is a Spam email!"}