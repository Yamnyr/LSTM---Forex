import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import mutual_info_regression
import warnings

warnings.filterwarnings('ignore')

# Configuration optimisée
SEQUENCE_LENGTH = 45
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 96
NUM_LAYERS = 2
DROPOUT = 0.25
N_MODELS = 3  # Ensemble de 3 modèles
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🔧 Device: {DEVICE}")

# ==================== 1. CHARGEMENT DES DONNÉES ====================
print("\n📊 Chargement des données...")
df = pd.read_csv('../archive/daily_forex_rates.csv')

df = df[(df['base_currency'] == 'EUR') & (df['currency'] == 'USD')].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"✅ {len(df)} lignes chargées")
print(f"📅 Période: {df['date'].min()} à {df['date'].max()}")

# Split
train_df = df[df['date'].dt.year < 2023].copy()
val_df = df[df['date'].dt.year == 2023].copy()
test_df = df[df['date'].dt.year == 2024].copy()

print(f"📦 Train: {len(train_df)} lignes")
print(f"📦 Validation: {len(val_df)} lignes")
print(f"📦 Test: {len(test_df)} lignes")

# ==================== 2. FEATURES SÉLECTIONNÉES ====================
print("\n🔧 Création des features optimales...")


def add_features(df):
    df = df.copy()

    # Moyennes mobiles sélectionnées
    df['ma_7'] = df['exchange_rate'].rolling(window=7, min_periods=1).mean()
    df['ma_14'] = df['exchange_rate'].rolling(window=14, min_periods=1).mean()
    df['ma_30'] = df['exchange_rate'].rolling(window=30, min_periods=1).mean()
    df['ma_60'] = df['exchange_rate'].rolling(window=60, min_periods=1).mean()

    # EMA
    df['ema_12'] = df['exchange_rate'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['exchange_rate'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']

    # Bollinger Bands
    df['bb_middle'] = df['exchange_rate'].rolling(window=20, min_periods=1).mean()
    df['bb_std'] = df['exchange_rate'].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_position'] = (df['exchange_rate'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # Volatilité
    df['std_7'] = df['exchange_rate'].rolling(window=7, min_periods=1).std()
    df['std_14'] = df['exchange_rate'].rolling(window=14, min_periods=1).std()
    df['std_30'] = df['exchange_rate'].rolling(window=30, min_periods=1).std()

    # Returns
    df['return_1d'] = df['exchange_rate'].pct_change(1)
    df['return_5d'] = df['exchange_rate'].pct_change(5)
    df['return_10d'] = df['exchange_rate'].pct_change(10)

    # Momentum
    df['momentum_5'] = df['exchange_rate'] - df['exchange_rate'].shift(5)
    df['momentum_10'] = df['exchange_rate'] - df['exchange_rate'].shift(10)

    # RSI
    delta = df['exchange_rate'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Distance from MA
    df['dist_ma_30'] = df['exchange_rate'] - df['ma_30']

    # Temporal
    df['day_of_week'] = df['date'].dt.dayofweek / 6.0
    df['month'] = df['date'].dt.month / 12.0
    df['day_of_month'] = df['date'].dt.day / 31.0

    df = df.fillna(method='bfill').fillna(method='ffill')

    return df


train_df = add_features(train_df)
val_df = add_features(val_df)
test_df = add_features(test_df)

# Features sélectionnées (les plus importantes)
feature_cols = ['exchange_rate', 'ma_7', 'ma_14', 'ma_30', 'ma_60',
                'ema_12', 'ema_26', 'macd',
                'bb_position', 'std_7', 'std_14', 'std_30',
                'return_1d', 'return_5d', 'return_10d',
                'momentum_5', 'momentum_10', 'rsi', 'dist_ma_30',
                'day_of_week', 'month', 'day_of_month']

print(f"✅ {len(feature_cols)} features sélectionnées")

# Normalisation avec StandardScaler (meilleur que MinMax pour LSTM)
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[feature_cols].values)
val_scaled = scaler.transform(val_df[feature_cols].values)
test_scaled = scaler.transform(test_df[feature_cols].values)

# Scaler pour target
target_scaler = StandardScaler()
target_scaler.fit(train_df[['exchange_rate']].values)


# ==================== 3. DATASET ====================
class ForexDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length, 0]
        return torch.FloatTensor(X), torch.FloatTensor([y])


train_dataset = ForexDataset(train_scaled, SEQUENCE_LENGTH)
val_dataset = ForexDataset(val_scaled, SEQUENCE_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

print(f"✅ {len(train_dataset)} séquences d'entraînement")


# ==================== 4. MODÈLE LSTM OPTIMISÉ ====================
class OptimizedForexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=96, num_layers=2, dropout=0.25):
        super(OptimizedForexLSTM, self).__init__()

        # LSTM bidirectionnel
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # Attention simple mais efficace
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Fully connected
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # FC layers
        out = self.relu(self.fc1(context))
        out = self.dropout(out)

        out = self.relu(self.fc2(out))
        out = self.dropout(out)

        out = self.relu(self.fc3(out))
        out = self.fc_out(out)

        return out


# ==================== 5. ENSEMBLE DE MODÈLES ====================
print(f"\n🤖 Création d'un ensemble de {N_MODELS} modèles...")

models = []
optimizers = []
schedulers = []

for i in range(N_MODELS):
    model = OptimizedForexLSTM(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    models.append(model)
    optimizers.append(optimizer)
    schedulers.append(scheduler)

total_params = sum(p.numel() for p in models[0].parameters())
print(f"✅ Chaque modèle: {total_params:,} paramètres")

criterion = nn.MSELoss()

# ==================== 6. ENTRAÎNEMENT ====================
print("\n🚀 Entraînement de l'ensemble...")

all_train_losses = [[] for _ in range(N_MODELS)]
all_val_losses = [[] for _ in range(N_MODELS)]
best_val_losses = [float('inf')] * N_MODELS

for epoch in range(EPOCHS):
    # Train each model
    for model_idx, (model, optimizer) in enumerate(zip(models, optimizers)):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        all_train_losses[model_idx].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        all_val_losses[model_idx].append(val_loss)

        schedulers[model_idx].step(val_loss)

        # Save best
        if val_loss < best_val_losses[model_idx]:
            best_val_losses[model_idx] = val_loss
            torch.save(model.state_dict(), f'best_forex_model_{model_idx}.pth')

    # Print progress
    if (epoch + 1) % 10 == 0:
        avg_train = np.mean([losses[-1] for losses in all_train_losses])
        avg_val = np.mean([losses[-1] for losses in all_val_losses])
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Avg Train: {avg_train:.6f}, Avg Val: {avg_val:.6f}")

print(f"✅ Entraînement terminé!")

# Load best models
for i, model in enumerate(models):
    model.load_state_dict(torch.load(f'best_forex_model_{i}.pth'))
    model.eval()

# ==================== 7. PRÉDICTION AVEC ENSEMBLE ====================
print("\n📈 Prédictions avec ensemble de modèles...")

test_sequences = []
for i in range(len(test_scaled) - SEQUENCE_LENGTH):
    test_sequences.append(test_scaled[i:i + SEQUENCE_LENGTH])

# Predict with all models
all_predictions = []
for model in models:
    model_preds = []
    with torch.no_grad():
        for seq in test_sequences:
            X = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
            pred = model(X).detach().cpu().numpy()[0, 0]
            model_preds.append(pred)
    all_predictions.append(model_preds)

# Average predictions (ensemble)
predictions = np.mean(all_predictions, axis=0)

# Add padding
for i in range(SEQUENCE_LENGTH):
    ensemble_pred = []
    with torch.no_grad():
        for model in models:
            X = torch.FloatTensor(test_scaled[len(test_scaled) - SEQUENCE_LENGTH:]).unsqueeze(0).to(DEVICE)
            pred = model(X).detach().cpu().numpy()[0, 0]
            ensemble_pred.append(pred)
    predictions = np.append(predictions, np.mean(ensemble_pred))

# Denormalize
predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1))
true_values = test_df['exchange_rate'].values

# ==================== 8. MÉTRIQUES ====================
print("\n📊 Métriques de l'ensemble...")

mae = mean_absolute_error(true_values, predictions)
rmse = np.sqrt(mean_squared_error(true_values, predictions))
mape = np.mean(np.abs((true_values - predictions.flatten()) / true_values)) * 100

print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAPE: {mape:.2f}%")

# ==================== 9. VISUALISATION ====================
print("\n📊 Création des graphiques...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
dates = test_df['date'].values

# Plot 1
axes[0].plot(dates, true_values, label='Vraies valeurs', linewidth=2.5, color='#2E86AB', marker='o', markersize=2,
             alpha=0.9)
axes[0].plot(dates, predictions, label=f'Ensemble LSTM (×{N_MODELS})', linewidth=2.5, color='#D62828', marker='s',
             markersize=2, alpha=0.85)
axes[0].set_title(f'EUR/USD - Prédictions 2024 (Ensemble LSTM) - MAPE: {mape:.2f}%', fontsize=15, fontweight='bold')
axes[0].set_ylabel('Taux de change', fontsize=12)
axes[0].legend(fontsize=12, loc='best')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].tick_params(axis='x', rotation=45)

# Plot 2
zoom_days = min(120, len(dates))
axes[1].plot(dates[:zoom_days], true_values[:zoom_days], label='Vraies valeurs', linewidth=3, color='#2E86AB',
             marker='o', markersize=4)
axes[1].plot(dates[:zoom_days], predictions[:zoom_days], label=f'Ensemble LSTM', linewidth=3, color='#D62828',
             marker='s', markersize=4, alpha=0.85)
axes[1].set_title('EUR/USD - Zoom sur 120 premiers jours', fontsize=15, fontweight='bold')
axes[1].set_ylabel('Taux de change', fontsize=12)
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].tick_params(axis='x', rotation=45)

# Plot 3
errors = true_values - predictions.flatten()
axes[2].plot(dates, errors, linewidth=2, color='#F77F00', alpha=0.8)
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
axes[2].fill_between(dates, errors, 0, alpha=0.4, color='#F77F00')
axes[2].set_title(f'Erreur de Prédiction (MAE: {mae:.4f}, RMSE: {rmse:.4f})', fontsize=15, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=12)
axes[2].set_ylabel('Erreur', fontsize=12)
axes[2].grid(True, alpha=0.3, linestyle='--')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('v4.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: v4.png")
plt.show()

# Loss curves
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for i in range(N_MODELS):
    axes[0].plot(all_train_losses[i], label=f'Modèle {i + 1}', linewidth=2, alpha=0.7)
    axes[1].plot(all_val_losses[i], label=f'Modèle {i + 1}', linewidth=2, alpha=0.7)

axes[0].set_title('Train Loss par Modèle', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Loss par Modèle', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v4_loss.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✨ Terminé!")
print(f"🎯 Performance finale: MAPE = {mape:.2f}%, MAE = {mae:.4f}")