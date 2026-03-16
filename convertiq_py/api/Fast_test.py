from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

# wb pour ecrire en binary - a utiliser avec pickle.dump [pckl] - challenge Kaggle a revoir 
# load and open rb method