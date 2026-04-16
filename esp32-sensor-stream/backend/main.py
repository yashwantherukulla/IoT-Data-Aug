from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from spectrogram_utils import load_config, generate_spectrogram_base64, get_window_samples, compute_window_metrics

app = FastAPI()
cfg = load_config()

class SensorBatch(BaseModel):
    device_id: str
    acc: List[List[float]]
    gyro: List[List[float]]

# Simple in-memory storage for samples (rolling buffer)
MAX_SAMPLES = 1000
current_data = {
    "acc_x": [], "acc_y": [], "acc_z": [],
    "gyro_x": [], "gyro_y": [], "gyro_z": []
}

# Top 3 recent spectrograms
recent_spectrograms = {
    "acc": [],
    "gyr": []
}

# Latest metrics per modality
recent_metrics = {
    "acc": None,
    "gyr": None,
}

# Tracking for spectrogram generation frequency
samples_received = 0
WINDOW_SIZE = get_window_samples(cfg) # Usually 250 (2.5 seconds at 100Hz)

@app.post("/api/sensor")
async def receive_sensor_data(batch: SensorBatch):
    global current_data, samples_received, recent_spectrograms, recent_metrics
    
    # Append the new batch data and trim to MAX_SAMPLES
    new_acc = batch.acc
    new_gyro = batch.gyro
    
    current_data["acc_x"].extend([reading[0] for reading in new_acc])
    current_data["acc_y"].extend([reading[1] for reading in new_acc])
    current_data["acc_z"].extend([reading[2] for reading in new_acc])
    
    current_data["gyro_x"].extend([reading[0] for reading in new_gyro])
    current_data["gyro_y"].extend([reading[1] for reading in new_gyro])
    current_data["gyro_z"].extend([reading[2] for reading in new_gyro])
    
    # Update sample count
    samples_received += len(new_acc)
    
    # Generate spectrogram if we have enough new samples or it's been a while
    # We use the last WINDOW_SIZE samples for the spectrogram
    if samples_received >= WINDOW_SIZE:
        samples_received = 0 # Reset
        
        # Acc Spectrogram
        acc_window = list(zip(current_data["acc_x"][-WINDOW_SIZE:], 
                             current_data["acc_y"][-WINDOW_SIZE:], 
                             current_data["acc_z"][-WINDOW_SIZE:]))
        if len(acc_window) == WINDOW_SIZE:
            spec_b64 = generate_spectrogram_base64(acc_window, cfg)
            if spec_b64:
                recent_spectrograms["acc"].insert(0, spec_b64)
                recent_spectrograms["acc"] = recent_spectrograms["acc"][:3]
            try:
                recent_metrics["acc"] = compute_window_metrics(acc_window, cfg)
            except Exception:
                pass
        
        # Gyro Spectrogram
        gyro_window = list(zip(current_data["gyro_x"][-WINDOW_SIZE:], 
                              current_data["gyro_y"][-WINDOW_SIZE:], 
                              current_data["gyro_z"][-WINDOW_SIZE:]))
        if len(gyro_window) == WINDOW_SIZE:
            spec_b64 = generate_spectrogram_base64(gyro_window, cfg)
            if spec_b64:
                recent_spectrograms["gyr"].insert(0, spec_b64)
                recent_spectrograms["gyr"] = recent_spectrograms["gyr"][:3]
            try:
                recent_metrics["gyr"] = compute_window_metrics(gyro_window, cfg)
            except Exception:
                pass

    # Keep only the last K samples
    for key in current_data:
        current_data[key] = current_data[key][-MAX_SAMPLES:]
    
    return {"status": "success"}

@app.get("/api/data")
async def get_latest_data():
    return current_data

@app.get("/api/spectrograms")
async def get_spectrograms():
    return recent_spectrograms

@app.get("/api/metrics")
async def get_metrics():
    return recent_metrics