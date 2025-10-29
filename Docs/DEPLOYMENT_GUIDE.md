# ClearRoads: Easiest Deployment Guide

## Option 1: Save Model & Create Simple API (EASIEST)

### Step 1: Save the Trained Model

Add this to the end of `traffic_analysis_v2.py`:

```python
import joblib

# After training XGBoost model
best_pipeline = results['XGBoost']['pipeline']
joblib.dump(best_pipeline, 'clearroads_xgboost_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(feature_cols, 'feature_columns.pkl')

print("✓ Model saved as 'clearroads_xgboost_model.pkl'")
```

### Step 2: Create Simple Prediction Script

```python
# predict.py
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('clearroads_xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_columns.pkl')

def predict_congestion(input_data):
    """
    input_data: dict with keys matching feature_cols
    returns: congestion level (Very Low, Low, High, Very High)
    """
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure all features present
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Predict
    prediction = model.predict(df[feature_cols])
    congestion_level = label_encoder.inverse_transform(prediction)[0]
    
    # Get probability
    probabilities = model.predict_proba(df[feature_cols])[0]
    confidence = max(probabilities) * 100
    
    return {
        'congestion_level': congestion_level,
        'confidence': f'{confidence:.1f}%',
        'probabilities': {
            label: f'{prob*100:.1f}%' 
            for label, prob in zip(label_encoder.classes_, probabilities)
        }
    }

# Example usage
if __name__ == "__main__":
    sample_input = {
        'morning_rush': 15000,
        'evening_rush': 18000,
        'peak_hour_traffic': 8500,
        'distance_to_cbd_km': 25.5,
        'PM2_5': 12.3,
        'PM10': 25.6,
        'NO2': 45.2,
        'rainfall_mm': 0.0,
        'max_temp_c': 28.5,
        # ... other features
    }
    
    result = predict_congestion(sample_input)
    print(f"Predicted: {result['congestion_level']} ({result['confidence']} confidence)")
```

### Step 3: Create Flask API (5 minutes)

```python
# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model once at startup
model = joblib.load('clearroads_xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        
        # Ensure all features
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Predict
        prediction = model.predict(df[feature_cols])
        probabilities = model.predict_proba(df[feature_cols])[0]
        
        congestion_level = label_encoder.inverse_transform(prediction)[0]
        
        return jsonify({
            'success': True,
            'congestion_level': congestion_level,
            'confidence': float(max(probabilities)),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(label_encoder.classes_, probabilities)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'XGBoost v2.0'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**Install Flask:**
```bash
pip install flask
```

**Run API:**
```bash
python app.py
```

**Test API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"morning_rush": 15000, "evening_rush": 18000, "peak_hour_traffic": 8500}'
```

---

## Option 2: Streamlit Web App (EASIEST UI)

```python
# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd

st.title('🚗 ClearRoads: Traffic Congestion Predictor')
st.write('98.30% Accurate Traffic Prediction')

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('clearroads_xgboost_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    return model, label_encoder, feature_cols

model, label_encoder, feature_cols = load_model()

# Input form
st.header('Enter Traffic & Environmental Data')

col1, col2 = st.columns(2)

with col1:
    morning_rush = st.number_input('Morning Rush (6-9am)', value=15000)
    evening_rush = st.number_input('Evening Rush (4-7pm)', value=18000)
    peak_hour = st.number_input('Peak Hour Traffic', value=8500)
    distance_cbd = st.number_input('Distance to CBD (km)', value=25.0)

with col2:
    pm25 = st.number_input('PM2.5', value=12.0)
    pm10 = st.number_input('PM10', value=25.0)
    no2 = st.number_input('NO2', value=45.0)
    temp = st.number_input('Max Temperature (°C)', value=28.0)

if st.button('Predict Congestion'):
    # Create input
    input_data = {
        'morning_rush': morning_rush,
        'evening_rush': evening_rush,
        'peak_hour_traffic': peak_hour,
        'distance_to_cbd_km': distance_cbd,
        'PM2_5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'max_temp_c': temp,
    }
    
    df = pd.DataFrame([input_data])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Predict
    prediction = model.predict(df[feature_cols])
    probabilities = model.predict_proba(df[feature_cols])[0]
    
    congestion_level = label_encoder.inverse_transform(prediction)[0]
    confidence = max(probabilities) * 100
    
    # Display result
    st.success(f'Predicted Congestion: **{congestion_level}**')
    st.metric('Confidence', f'{confidence:.1f}%')
    
    # Show probabilities
    st.subheader('Probability Distribution')
    prob_df = pd.DataFrame({
        'Level': label_encoder.classes_,
        'Probability': probabilities * 100
    })
    st.bar_chart(prob_df.set_index('Level'))
```

**Install Streamlit:**
```bash
pip install streamlit
```

**Run App:**
```bash
streamlit run streamlit_app.py
```

---

## Option 3: Docker Container (PRODUCTION)

```dockerfile
# Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY clearroads_xgboost_model.pkl .
COPY label_encoder.pkl .
COPY feature_columns.pkl .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

**Build & Run:**
```bash
docker build -t clearroads-api .
docker run -p 5000:5000 clearroads-api
```

---

## Recommendation

**For Academic Demo:** Use **Streamlit** (Option 2) - Beautiful UI, zero web dev knowledge needed

**For Production:** Use **Flask API** (Option 1) + Docker (Option 3) - Industry standard

**Deployment Time:**
- Streamlit: 10 minutes
- Flask API: 15 minutes
- Docker: 20 minutes

All options are production-ready and can handle real-time predictions!
