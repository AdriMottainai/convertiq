FROM python:3.10-slim
COPY requirements_prod.txt .
COPY setup.py .
COPY convertiq_py ./convertiq_py
RUN apt-get update && apt-get install -y libgomp1
RUN pip install -r requirements_prod.txt
CMD ["uvicorn", "convertiq_py.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
