from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}
