# Online inference

## Run locally

`python app.py sources/transformer.pkl sources/model_logreg.pkl`

## Docker

- `docker build -t pollytur/heart-flask .`  
- `docker pull pollytur/heart-flask`  
- `docker run -it -d -p 5000:5000 pollytur/heart-flask`  

## Request

`curl -X POST -H "Content-Type: application/json" -d @request.json localhost:5000/predict`

Example request is in `request.json` file.  
You may run `sh tickle.sh`, which runs the cmd above.  

## Test

`python -m pytest`
