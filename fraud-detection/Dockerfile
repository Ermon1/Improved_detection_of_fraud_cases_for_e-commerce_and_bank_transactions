FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install dependencies
# We need to make sure we install the packages required for the app and model
# including scikit-learn, xgboost, fastpi, uvicorn etc.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "5000"]
