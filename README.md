# FastAPI implementation of an SKLearn pipeline

This is a FastAPI based API template for putting sklearn pipeline-based predictors into production.

## Installation

The provided notebook /analysis/generate_model.ipynb can be used to generate the resource paths needed as well as a dummy classifier for the API.

If installing in conda on Windows:

```bash
conda create -n clusterer python=3.9
pip install -r requirements.txt
uvicorn app.main:app
```

If using Docker, use the provided Dockerfile:

```
docker build -t epg_api . 
docker run -d --name mycontainer -p 8000:8000 epg_api --mount type=bind, source=./resources, target=/code/resources

```

Or just use the docker compose file:

```
docker-compose build 
docker-compose run

```
## Connectivity

The predefined port in this repository is 8000. To change it when running locally do it by adding in the call to uvicron --port PORTNUMBER. If running it as a container change it on docker run sentence or in the docker-compose.yaml if using it.

## Docs

The swagger docs are located at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## TODO

 * Encapsulate model creation
 * Change Swagger parameters
 * Check adaptability for prediction/classification
 * Generation of model from docker start-up/environment variable
 * Add endpoints to change the hyperparameters of the model.
 * Definition of model on environment variables.