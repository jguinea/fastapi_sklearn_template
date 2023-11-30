# Video-MOS Clusterer

This is a FastAPI based API template for putting sklearn pipeline based predictors into production.

## Installation

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

 * Find memory hoarding problem and solve it.
 * Check prediction of individual data points for issues. Probably a bug somewhere.
 * Installable and encrypted standalone module for Ubuntu.
 * Add model generation on start-up.
 * Add endpoints to change the hyperparameters of the model.
 * Use sklearn pipeline for full model.
 * Change UMAP online.
 * Remove -1 ids from return values of API