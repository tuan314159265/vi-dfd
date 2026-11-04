# Scaffold only: /recognize returns 501 until ONNX inference is implemented.
# Pin the image digest and serving dependencies when validating the final demo.
FROM python:3.12-slim
WORKDIR /app
COPY serve/requirements.in /app/requirements.in
RUN pip install --no-cache-dir -r /app/requirements.in
COPY serve /app/serve
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
