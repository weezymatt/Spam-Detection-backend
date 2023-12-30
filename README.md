# Spam-Detection-with-FastAPI

> **Note** Access to nlp documentation [here](https://github.com/weezymatt/text-scrapbook)

Last updated December 28th 2023.

> **Note** The true scope of this project involves the full implementation and integration of the classifier into a website. This repository details the backend side of machine learning.

This machine learning (ML) project was created for the purpose of deploying a ML-based API model by using the cloud provider Render to set up the environment and host the docker container. The swagger documentation of the API is available [here](https://spam-detection-e9se.onrender.com/docs) for viewing. 

## Table of Contents
- [Objective](#objective)
- [MLOps](#mlops)
- [Project Setup](#project-setup)
- [Instructions](#instructions)
- [Deployment](#deployment)
- [See More](#see-more)

## Objective
The objective of this repository is to use a prediction model—that will accurately classify texts as spam—using the FastAPI framework and serve as an educational experience. We will not go over the details of our model as there is much documentation involving the spam dataset. However, it is necessary to understand the full picture of machine learning and here we are. Happy coding!

> The original dataset can be found [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

![hannes-johnson-mRgffV3Hc6c-unsplash](https://github.com/weezymatt/Spam-Detection-backend/assets/85853890/acd2af65-3153-42a3-b857-5ae63e7f7a16)
Photo by <a href="https://unsplash.com/@hannes?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Hannes Johnson</a> on <a href="https://unsplash.com/photos/blue-and-brown-cardboard-boxes-mRgffV3Hc6c?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

<a name="mlops"></a>
## MLOps — *Where does my model go from here?*
The "Spam Detection backend" project aims to provide a brief outline of Machine Learning Operations (MLOps) by focusing on certain steps necessary to deploy your ML model. Below I have provided helpful documentation that allowed me to complete this project.

### Helpful Resources 
1. [Machine Learning Mastery: Save and Load ML Models in Python](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)
    
   This article you will discover how to save and load your model with pickle and joblib. You will then be able to reuse your saved file to make predictions at this stage.

2. [Integrating ML classifier with FastAPI](https://www.fastapitutorial.com/blog/ml-model-prediction-with-fastapi/)

   Explanation of the overall backend pipeline for ML models. Granted, I did not follow this article much but the ```main.py``` file provides a simple overview of how integrating your saved model with FastAPI.

3. [Building a Machine Learning API in 15 Minutes](https://www.youtube.com/watch?v=C82lT9cWQiA&t=300s)

   Very useful video! The format details, in digestible parts, how an API project may be deployed. Additionally, you can the application with ```uvicorn app:app --reload``` at this stage.

4. [FastAPI in Containers - Docker](https://fastapi.tiangolo.com/deployment/docker/)

   This is another very helpful tutorial that clearly demonstrates the purpose of Docker in machine learning. Crucially, there is a piece on creating a Dockerfile that helped greatly. You can build the Docker image and start the container at this stage.

5. [Share the Application - Docker](https://docs.docker.com/get-started/04_sharing_app/)
  
   After building your Docker image, you can share it with Docker Hub. The purpose of sharing the image allows for easy integration into a cloud environment (i.e. Render) and demonstrates the portablility of Docker containers. You can run your application on a hosted site at this stage.

## Project Setup
> Note You may ignore this section if you only interested in deploying the model.

To set up the project environment locally, follow these steps:

1. Cloning the Repository
```bash
git clone https://github.com/weezymatt/Spam-Detection-backend.git
cd spam-detection-backend
```
2. Setting up Virtual Environment
- Windows
  ```bash
  python -m venv <virtual-environment-name>
  venv\Scripts\activate
  ```
- Linux and MacOS
  ```bash
  python3 -m venv <virtual-environment-name>
  source env/bin/activate
  ```
3. Install the Required Dependencies

   *The virtual environment will make use of its own pip, so you don't need to use pip3.*
   ```bash
   pip install -r requirements.txt
   ```
The commands above can be copied and run in your terminal to easily simulate my project environment.

<a name="instructions"></a>
## Instructions to build FastAPI app
> Note You may skip this section and go to Deployment if you're knowledgeable about APIs.

There are a few steps required to build your FastAPI app and capture the essence of your model. Here we briefly discuss how to write the ```app.py``` code and the ```Dockerfile```.

### Create the API
Following the initialization of your virtual environment, we will write the ```app.py``` file and initialize an instance of FastAPI.

```python
app = FastAPI(title="Ham or Spam API", description="API to predict SMS spam")
```

Load the saved models with joblib. A vectorizer is loaded to follow the same processing steps in the Jupyter Notebook.

```python
model = joblib.load("model/finalized_model.sav")

vectorizer = joblib.load("model/vectorizer.sav")
```

Define the data format for incoming input.

```python
class request_body(BaseModel):
	message   : str # A free service for you ONLY!! Please click on the link now! // String value
```

Process the input sent by the user.

```python
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
```

Define the ```GET``` method.

```python
@app.get('/')
def Welcome():
	return{'message': 'Welcome to the Spam classifier API!'}
```

Create the ```POST``` method (this is the meat of your API).
```python
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
```
> Note The last section in the Jupyter Notebook demonstrates how I used my Naïve Bayes model to make predictions before making the app.py file.

### Write the requirements.txt 
> Realistically you have a virtual environment ready and install your dependencies throughout the project then freeze them into a text file.

The ```requirements.txt``` file enables us to recreate all the modules necessary for our application. This is crucial when we write our ```Dockerfile``` later on. 

```python
pip freeze > requirements.txt
```
## Deployment
Here we develop the deployment in stages until we reach the container step where we are able to display the webpage.

### FastAPI Deployment

1. Open the terminal and navigate to the directory where your ```app.py``` file is located.

2. Run the FastAPI application by using the uvicorn command, specifying the application name. The "--reload" feature is useful for changes to be automatically reflected.

   ```python
   uvicorn <application-file>:app --reload
   ```

3. After running the uvicorn command, the FastAPI application is up and running on the specific address (i.e. localhost:8000) listed. This address represents the API endpoint where we can access our application. We will see the importance of this endpoint during the front-end part of the project.

4. You may open your browser to interact with your deployed FastAPI application. The endpoint acts as an intermediary between requests and responses (Press CTRL+C to quit).

### API Documentation
The FastAPI Documentation details the available endpoints, JSON request and response formats, and information specified in your ```app.py``` file. You can access this documentation by adding the /docs to the server (http://localhost:8000/docs).

### Containerized Deployment
hello
## See More

For the second part of this project involving the front-end piece, please click [here](pending).
