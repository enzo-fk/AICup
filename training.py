import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch.optim as optim
import glob


def load_location_data(location_number):
    """
    Load data for a specific location, handling both regular and _2 files
    """
    files = []

    dir_name = "data/"

    try:
        regular_file = f'{dir_name}/L{location_number}_Train.csv'
        df1 = pd.read_csv(regular_file)
        files.append(df1)
    except FileNotFoundError:
        pass

    try:
        extra_file = f'{dir_name}/L{location_number}_Train_2.csv'
        df2 = pd.read_csv(extra_file)
        files.append(df2)
    except FileNotFoundError:
        pass

    if not files:
        raise FileNotFoundError(f"No data files found for location {location_number}")

    # Combine all available data
    df = pd.concat(files, ignore_index=True)

    # Basic preprocessing
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime')

    return df


def load_all_data():
    """
    Load data from all locations and combine into one DataFrame
    """
    all_data = []

    for location in range(1, 18):
        try:
            df = load_location_data(location)
            all_data.append(df)
        except FileNotFoundError:
            print(f"Warning: No data found for location {location}")

    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


df = load_all_data()

def clean_and_prepare_data(df):
    # Remove obvious sensor errors
    df = df.copy()
    mask = (
            (df['Pressure(hpa)'] < 1030) & (df['Pressure(hpa)'] > 950) &
            (df['Temperature(°C)'] < 45) & (df['Temperature(°C)'] > 0) &
            (df['Humidity(%)'] <= 100) & (df['Humidity(%)'] > 10)
    )
    df = df[mask]

    # Create time features
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute
    df['month'] = df['DateTime'].dt.month
    df['day'] = df['DateTime'].dt.day
    df['dayofweek'] = df['DateTime'].dt.dayofweek

    # Create cyclic features
    # Time features like hours (0-23) and months (1-12) are cyclical - hour 23 is actually close to hour 0, and December (12) is close to January (1)
    # Using raw numbers would make the model think these values are far apart, when they're actually adjacent in time
    df['hour_sin'] = np.sin(2 * np.pi * (df['hour'] + df['minute'] / 60) / 24)
    df['hour_cos'] = np.cos(2 * np.pi * (df['hour'] + df['minute'] / 60) / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Handle sunlight saturation
    max_sunlight = 117758.2
    df['is_saturated'] = (df['Sunlight(Lux)'] >= max_sunlight).astype(int)
    return df

cleaned_df = clean_and_prepare_data(df)
df = cleaned_df
data = df


weather_df = pd.read_csv("data/weather.csv")
weather_df.rename(columns={"tmp": "Temperature", "sun": "Sunshine", "rad": "Radiation"}, inplace=True)

data["Year"] = pd.to_datetime(data["DateTime"]).dt.year
data["Month"] = pd.to_datetime(data["DateTime"]).dt.month
data["Day"] = pd.to_datetime(data["DateTime"]).dt.day
data["Hour"] = pd.to_datetime(data["DateTime"]).dt.hour
data["Minute"] = pd.to_datetime(data["DateTime"]).dt.minute

data = data.merge(weather_df, on=["Month", "Day", "Hour"], how="left")
data["Temperature"].fillna(data["Temperature"].mean(), inplace=True)
data["Sunshine"].fillna(data["Sunshine"].mean(), inplace=True)
data["Radiation"].fillna(data["Radiation"].mean(), inplace=True)

direction_mapping = {
    1: 181, 2: 175, 3: 180, 4: 161, 5: 208,
    6: 208, 7: 172, 8: 219, 9: 151, 10: 223,
    11: 131, 12: 298, 13: 249, 14: 197,
    15: 127, 16: 82, 17: np.nan
}

data["direction_sin"] = np.sin(np.radians(data["LocationCode"].map(direction_mapping).fillna(0)))
data["direction_cos"] = np.cos(np.radians(data["LocationCode"].map(direction_mapping).fillna(0)))

height_mapping = {
    1: 5, 2: 5, 3: 5, 4: 5, 5: 5,
    6: 5, 7: 5, 8: 3, 9: 3, 10: 1,
    11: 1, 12: 1, 13: 5, 14: 5,
    15: 1, 16: 1, 17: 3
}
data["building_height"] = data["LocationCode"].map(height_mapping)

def get_season(month):
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    else:
        return 4

data["Season"] = data["Month"].apply(get_season)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract relevant features
time_features = ["Year", "Month", "Day", "Hour", "Minute", "LocationCode",
                 "Season", "direction_sin", "direction_cos", "building_height",
                 "Temperature", "Sunshine", "Radiation"]

features = data[time_features]
target = data["Power(mW)"]

# Normalize data
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target.values, test_size=0.2, random_state=42
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class FNN(nn.Module):
    def __init__(self, input_dim):
        super(FNN, self).__init__()

        # Feature preprocessing with squeeze-and-excitation
        self.se_block = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

        # Multi-scale feature extraction
        self.path1 = nn.Sequential(  # Deep path
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.path2 = nn.Sequential(  # Medium path
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.path3 = nn.Sequential(  # Short path
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Main pyramid structure
        self.pyramid = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1792, 896),  # 1024 + 512 + 256 = 1792
                nn.LayerNorm(896),
                nn.GELU(),
                nn.Dropout(0.25)
            ),
            nn.Sequential(
                nn.Linear(896, 448),
                nn.LayerNorm(448),
                nn.GELU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(448, 224),
                nn.LayerNorm(224),
                nn.GELU(),
                nn.Dropout(0.15)
            )
        ])

        # Skip connections with gating mechanism
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1792, 896),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Linear(896, 448),
                nn.Sigmoid()
            ),
            nn.Sequential(
                nn.Linear(448, 224),
                nn.Sigmoid()
            )
        ])

        # Output layers with multi-head prediction
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(224, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(3)
        ])

        # Final aggregation
        self.output_weights = nn.Parameter(torch.ones(3) / 3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Squeeze and excitation
        se_weights = self.se_block(x)
        x = x * se_weights

        # Multi-path feature extraction
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)

        # Concatenate all paths
        features = torch.cat([p1, p2, p3], dim=1)

        # Pyramid with gated skip connections
        skip_features = features
        for pyramid_layer, gate in zip(self.pyramid, self.gates):
            # Main path
            features = pyramid_layer(features)
            # Gated skip connection
            gate_weights = gate(skip_features)
            skip_features = features * gate_weights

        # Multi-head prediction
        outputs = []
        for head in self.output_heads:
            outputs.append(head(features))

        # Weighted average of predictions
        weights = F.softmax(self.output_weights, dim=0)
        final_output = sum(w * o for w, o in zip(weights, outputs))

        return final_output


input_dim = len(time_features)
model = FNN(input_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=1,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
)


def train_model(model, train_loader, test_loader, epochs=100):
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                test_loss += criterion(outputs, y_batch).item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.6f}")
        print(f"Test Loss: {avg_test_loss:.6f}")
        print("-" * 50)

        # Early stopping check
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    return train_losses, test_losses


#train_losses, test_losses = train_model(model, train_loader, test_loader)

# Load best model for evaluation
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Model evaluation
model.eval()
with torch.no_grad():
    test_predictions = []
    test_actuals = []
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        predictions = model(X_batch).cpu().numpy()
        test_predictions.extend(predictions)
        test_actuals.extend(y_batch.numpy())

test_predictions = np.array(test_predictions)
test_actuals = np.array(test_actuals)

# Calculate metrics
mse = np.mean((test_predictions - test_actuals) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(test_predictions - test_actuals))
r2 = 1 - np.sum((test_actuals - test_predictions) ** 2) / np.sum((test_actuals - test_actuals.mean()) ** 2)

print("\nModel Performance Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R2 Score: {r2:.4f}")


# Parse datetime and location code
answer_data = pd.read_csv("data/upload.csv")
answer_data["Year"] = answer_data["序號"].astype(str).str[:4].astype(int)
answer_data["Month"] = answer_data["序號"].astype(str).str[4:6].astype(int)
answer_data["Day"] = answer_data["序號"].astype(str).str[6:8].astype(int)
answer_data["Hour"] = answer_data["序號"].astype(str).str[8:10].astype(int)
answer_data["Minute"] = answer_data["序號"].astype(str).str[10:12].astype(int)
answer_data["LocationCode"] = answer_data["序號"].astype(str).str[12:].astype(int)

# Feature engineering
# Add season
def get_season(month):
    if month in [12, 1, 2]:
        return 1  # Winter
    elif month in [3, 4, 5]:
        return 2  # Spring
    elif month in [6, 7, 8]:
        return 3  # Summer
    else:
        return 4  # Fall

answer_data["Season"] = answer_data["Month"].apply(get_season)

# Add direction and building height features
answer_data["direction_sin"] = np.sin(np.radians(answer_data["LocationCode"].map(direction_mapping).fillna(0)))
answer_data["direction_cos"] = np.cos(np.radians(answer_data["LocationCode"].map(direction_mapping).fillna(0)))
answer_data["building_height"] = answer_data["LocationCode"].map(height_mapping)

# Merge weather data
answer_data = answer_data.merge(weather_df, on=["Month", "Day", "Hour"], how="left")

# Handle missing values
for col in ["Temperature", "Sunshine", "Radiation"]:
    answer_data[col].fillna(answer_data[col].mean(), inplace=True)

# Prepare features
test_features = answer_data[time_features]
print(f"Feature columns used for prediction: {test_features.columns.tolist()}")

# Scale features
scaler = joblib.load("scaler.pkl")
test_features_scaled = scaler.transform(test_features)

# Convert to PyTorch tensor
test_tensor = torch.tensor(test_features_scaled, dtype=torch.float32).to(device)

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(test_tensor).cpu().numpy().flatten()

# Ensure no negative predictions
predictions = np.maximum(predictions, 0)

# Print prediction statistics
print("\nPrediction statistics:")
print(f"Number of predictions: {len(predictions)}")
print(f"Min prediction: {predictions.min():.2f}")
print(f"Max prediction: {predictions.max():.2f}")
print(f"Mean prediction: {predictions.mean():.2f}")

# Save results
answer_data["答案"] = predictions
answer_data[["序號", "答案"]].to_csv("predictions.csv", index=False)
print("Results saved successfully")