import numpy as np
import pandas as pd
import xarray as xr
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the NetCDF dataset
nc_file = '../data/raw/WWLLN_sd_td_2005.nc/WWLLN_sd_td_2005.nc'
df = xr.open_dataset(nc_file)

# Extract stroke density data
stroke_density = df['stroke_density'].values  # Shape: (lon, lat, 12)
lat_values = df['lat'].values
lon_values = df['lon'].values

print(f"Data shape: {stroke_density.shape}")
print(f"Lat size: {len(lat_values)}, Lon size: {len(lon_values)}")

# Reshape for grid-level training
# Flatten spatial dimensions: (num_grids, timesteps)
lon_size, lat_size, timesteps = stroke_density.shape
num_grids = lon_size * lat_size
data_reshaped = stroke_density.reshape(num_grids, timesteps)

# Normalize each grid independently and save scalers
print("\nNormalizing data...")
scalers = {}
data_scaled = np.zeros_like(data_reshaped, dtype=np.float32)

for grid_idx in range(num_grids):
    scaler = MinMaxScaler()
    data_scaled[grid_idx] = scaler.fit_transform(
        data_reshaped[grid_idx].reshape(-1, 1)
    ).flatten()
    scalers[grid_idx] = scaler

# Create sequences for LSTM training
# Predict next timestep from lookback window
lookback = 11
X, y = [], []

for grid_idx in range(num_grids):
    for i in range(len(data_scaled[grid_idx]) - lookback):
        X.append(data_scaled[grid_idx, i:i+lookback])
        y.append(data_scaled[grid_idx, i+lookback])

X = np.array(X, dtype=np.float32)  # Shape: (num_samples, 11)
y = np.array(y, dtype=np.float32)  # Shape: (num_samples,)

# Reshape X for LSTM: (samples, timesteps, features)
X = X.reshape(-1, lookback, 1)

print(f"Training data shapes - X: {X.shape}, y: {y.shape}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build LSTM model for grid-level prediction
model = keras.Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("\nModel architecture:")
model.summary()

# Train the model
print("\nTraining LSTM model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate on test set
print("\nEvaluating model...")
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

# Save model and scalers
model_dir = '../models/lstm_monthly'
os.makedirs(model_dir, exist_ok=True)

model.save(os.path.join(model_dir, 'lstm_grid_model.h5'))
print(f"\nModel saved to {model_dir}/lstm_grid_model.h5")

# Save scalers
with open(os.path.join(model_dir, 'scalers.pkl'), 'wb') as f:
    pickle.dump(scalers, f)
print(f"Scalers saved to {model_dir}/scalers.pkl")

# Save grid information for later use
grid_info = {
    'lon_size': lon_size,
    'lat_size': lat_size,
    'num_grids': num_grids,
    'lookback': lookback,
    'lat_values': lat_values,
    'lon_values': lon_values
}
with open(os.path.join(model_dir, 'grid_info.pkl'), 'wb') as f:
    pickle.dump(grid_info, f)
print(f"Grid info saved to {model_dir}/grid_info.pkl")

print("\nTraining complete!")