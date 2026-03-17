FROM python:3.10-slim
COPY requirements_prod.txt .
COPY setup.py .
COPY convertiq_py ./convertiq_py
COPY save_models ./model_lgbm_baseline.pkl
RUN pip install -r requirements_prod.txt && \
    pip install -e .
CMD ["uvicorn", "convertiq_py.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
