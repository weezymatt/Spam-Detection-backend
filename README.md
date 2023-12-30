# Spam-Detection-with-FastAPI

> **Note** Access to nlp documentation [here](https://github.com/weezymatt/text-scrapbook)

Last updated December 28th 2023.

> **Note** The true scope of this project involves the full implementation and integration of the classifier into a website. This repository details the backend side of machine learning.

This machine learning (ML) project was created for the purpose of deploying a ML-based API model by using the cloud provider Render to set up the environment and host the docker container. The swagger documentation of the API is available [here](https://spam-detection-e9se.onrender.com/docs) for viewing. 

## Table of Contents
- [Objective](#objective)
- [MLOps](#MLOps)
- [Project Setup](#project-setup)
- [Deployment](#deployment)
- [See More](#see-more)

## Objective
The objective of this repository is to build a prediction model that will accurately classify which texts are spam using the FastAPI framework and serve as an educational experience. As there is many documentation involving the spam dataset, we will not be going over the details of our model here. However, I do find it important to understand the full picture of machine learning and thus here we are. Happy coding! 
> The original dataset can be found [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

![hannes-johnson-mRgffV3Hc6c-unsplash](https://github.com/weezymatt/Spam-Detection-backend/assets/85853890/acd2af65-3153-42a3-b857-5ae63e7f7a16)
Photo by <a href="https://unsplash.com/@hannes?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Hannes Johnson</a> on <a href="https://unsplash.com/photos/blue-and-brown-cardboard-boxes-mRgffV3Hc6c?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>

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


## Deployment
### FastAPI Deployment
hello

## See More
  
