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
The objective of this repository is to use a prediction model—that will accurately classify texts as spam—using the FastAPI framework and serve as an educational experience. Happy coding!

> The original dataset can be found [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

![hannes-johnson-mRgffV3Hc6c-unsplash](https://github.com/weezymatt/Spam-Detection-backend/assets/85853890/acd2af65-3153-42a3-b857-5ae63e7f7a16)
Photo by <a href="https://unsplash.com/@hannes?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Hannes Johnson</a> on <a href="https://unsplash.com/photos/blue-and-brown-cardboard-boxes-mRgffV3Hc6c?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

<a name="mlops"></a>
## MLOps — *Where does my model go from here?*
The "Spam Detection backend" project aims to provide a brief outline of Machine Learning Operations (MLOps) by focusing on certain steps necessary to deploy your ML model. Below I have provided helpful documentation that allowed me to complete this project.

### Helpful Resources 
1. [Machine Learning Mastery: Save and Load ML Models in Python](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)
    
   This article you will discover how to save and load your model with pickle or joblib. You will then be able to reuse your saved file to make predictions at this stage.

2. [Integrating ML classifier with FastAPI](https://www.fastapitutorial.com/blog/ml-model-prediction-with-fastapi/)

   Explanation of the overall backend pipeline for ML models. Granted, I did not follow this article much but the ```main.py``` file provides a simple overview of how integrating your saved model with FastAPI looks like.

3. [Building a Machine Learning API in 15 Minutes](https://www.youtube.com/watch?v=C82lT9cWQiA&t=300s)

   Very useful video on how an API project may be deployed! Additionally, you can run the application with ```uvicorn app:app --reload``` at this stage.

4. [FastAPI in Containers - Docker](https://fastapi.tiangolo.com/deployment/docker/)

   Another helpful tutorial that demonstrates the purpose of Docker and details how to create a Dockerfile. You can build the Docker image and start the container at this stage.

5. [Share the Application - Docker](https://docs.docker.com/get-started/04_sharing_app/)
  
   After building your Docker image, you can share it with Docker Hub. The purpose of sharing allows for easy integration into a cloud environment and demonstrates the portablility of containers. You can run your application on a hosted site at this stage.

## Project Setup
> Note: You may ignore this section if you only interested in deploying the model. The commands below can be copied and run in your terminal to easily simulate my project environment.

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
   
<a name="instructions"></a>
## Instructions to build FastAPI app
> Note: You may ignore this section and go to Deployment if you're knowledgeable about APIs. The Jupyter Notebook is useful for using your model to test predictions before makig the app.py file.

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
    ... 

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

### Write the requirements.txt 
> Realistically you have a virtual environment ready and install your dependencies throughout the project then freeze them into a text file.

The ```requirements.txt``` file enables us to recreate all the modules necessary for our application. This is crucial when we write our ```Dockerfile``` later on. 

```python
pip freeze > requirements.txt
```

Deactivate your virtual environment.

```bash
deactivate
```

## Deployment
Here we develop the deployment in stages until we reach the container step where we are able to display the webpage.

### FastAPI Deployment

1. Open the terminal and navigate to the directory where your ```app.py``` file is located.

2. Run the FastAPI application by using the uvicorn command, specifying the application name. The ```--reload``` feature is useful for changes to be automatically reflected.

   ```python
   uvicorn <application-file>:app --reload
   ```

3. After running the uvicorn command, the FastAPI application is up and running on the specific address (i.e. http://localhost:8000) listed. This address represents the API endpoint where we can access our application. We will see the importance of this endpoint during the front-end part of the project.

4. You may open your browser to interact with your deployed FastAPI application. The endpoint acts as an intermediary between requests and responses (Press CTRL+C to quit).

### API Documentation
The FastAPI Documentation details the available endpoints, JSON request and response formats, and information specified in your ```app.py``` file. You can access this documentation by adding the /docs to the server (http://localhost:8000/docs).

<img width="1364" alt="Spam-endpoint" src="https://github.com/weezymatt/Spam-Detection-backend/assets/85853890/9a562132-7b2d-4b43-8038-332cfc1bf4a2">

<img width="1334" alt="api-predict" src="https://github.com/weezymatt/Spam-Detection-backend/assets/85853890/6e5f300c-1068-4ccd-a6cd-5f547058b42e">

### Containerized Deployment

When deploying an API a common approach is to build a container image. We will be needing to write a Dockerfile for the application.

#### Dockerfile
> Dependency Issue: For the Docker container to properly run, an additional file initializing the NLTK stopwords was incorporated into the workflow. This may not be necessary in your process.

```Dockerfile
FROM python:3.11.5

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./initialize.py /code/initialize.py
RUN python3 /code/initialize.py

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
```
#### Docker Image
> If you are using an ARM-based Mac with Apple Silicon, you will need to rebuild the image to be compatible and push the new image to your repository on Docker Hub. Otherwise the process is rather straightforward. Click [here](https://stackoverflow.com/questions/71000707/docker-get-started-warning-the-requested-images-platform-linux-arm64-v8-doe) for a solution on Stack Overflow.

Let's build the container image. Use the ```docker tag``` command to give the image a new name.

```bash
docker tag <image-title> YOUR-USERNAME/<dockerhub-repo>
```

Switch to a new driver before your build (we will be following the process for an M1+).

```bash
docker buildx create --use
```

Launch the following command to build your Docker image.

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t <tag> .
```

Push the image to Docker Hub.

```bash
docker buildx build --push --platform linux/amd64,linux/arm64 -t docker.io/YOUR-USERNAME/<dockerhub-repo>:latest .
```

You may run your image to verify it is working and visit the server (http://localhost:8000/docs).

```bash
docker run -d --name mycontainer -p 8000:80 <image-title>
```

## See More
<img width="1346" alt="welcome-goodbye" src="https://github.com/weezymatt/Spam-Detection-backend/assets/85853890/1a228934-469f-4f64-9997-d9b76bb3d3a1">

There you have it! You can use your saved image on Docker Hub with your cloud environment of choice and start the next step of your application. For the second part of this project involving the front-end piece, please click [here](pending).
