import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Configuration améliorée
SEQUENCE_LENGTH = 30  # Réduit pour éviter le surapprentissage
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0005
HIDDEN_SIZE = 64
NUM_LAYERS = 3
DROPOUT = 0.3
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

# Split train/validation/test
train_df = df[df['date'].dt.year < 2023].copy()
val_df = df[df['date'].dt.year == 2023].copy()
test_df = df[df['date'].dt.year == 2024].copy()

print(f"📦 Train: {len(train_df)} lignes (avant 2023)")
print(f"📦 Validation: {len(val_df)} lignes (2023)")
print(f"📦 Test: {len(test_df)} lignes (2024)")

# ==================== 2. FEATURES ENGINEERING ====================
print("\n🔧 Création des features...")


def add_features(df):
    df = df.copy()
    # Features temporelles
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day

    # Features techniques (moyennes mobiles)
    df['ma_7'] = df['exchange_rate'].rolling(window=7, min_periods=1).mean()
    df['ma_30'] = df['exchange_rate'].rolling(window=30, min_periods=1).mean()
    df['ma_90'] = df['exchange_rate'].rolling(window=90, min_periods=1).mean()

    # Volatilité
    df['std_7'] = df['exchange_rate'].rolling(window=7, min_periods=1).std()
    df['std_30'] = df['exchange_rate'].rolling(window=30, min_periods=1).std()

    # Returns
    df['return_1d'] = df['exchange_rate'].pct_change(1)
    df['return_7d'] = df['exchange_rate'].pct_change(7)

    # Momentum
    df['momentum_5'] = df['exchange_rate'] - df['exchange_rate'].shift(5)
    df['momentum_10'] = df['exchange_rate'] - df['exchange_rate'].shift(10)

    # Remplir les NaN
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df


train_df = add_features(train_df)
val_df = add_features(val_df)
test_df = add_features(test_df)

# Sélection des features
feature_cols = ['exchange_rate', 'ma_7', 'ma_30', 'ma_90', 'std_7', 'std_30',
                'return_1d', 'return_7d', 'momentum_5', 'momentum_10',
                'day_of_week', 'month']

# Normalisation
scaler = MinMaxScaler(feature_range=(-1, 1))
train_scaled = scaler.fit_transform(train_df[feature_cols].values)
val_scaled = scaler.transform(val_df[feature_cols].values)
test_scaled = scaler.transform(test_df[feature_cols].values)

# Scaler séparé pour la target
target_scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaler.fit(train_df[['exchange_rate']].values)

print(f"✅ {len(feature_cols)} features créées")


# ==================== 3. DATASET ====================
class ForexDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length, 0]  # Seulement exchange_rate
        return torch.FloatTensor(X), torch.FloatTensor([y])


train_dataset = ForexDataset(train_scaled, SEQUENCE_LENGTH)
val_dataset = ForexDataset(val_scaled, SEQUENCE_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ {len(train_dataset)} séquences d'entraînement")


# ==================== 4. MODÈLE AMÉLIORÉ ====================
class ImprovedForexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3, dropout=0.3):
        super(ImprovedForexLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True  # Bidirectionnel pour capturer plus de contexte
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Fully connected layers avec skip connection
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Fully connected avec batch norm
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc_out(out)

        return out


print("\n🤖 Création du modèle amélioré...")
model = ImprovedForexLSTM(
    input_size=len(feature_cols),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)

criterion = nn.HuberLoss()  # Plus robuste que MSE
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

print(f"✅ Modèle créé avec {sum(p.numel() for p in model.parameters()):,} paramètres")

# ==================== 5. ENTRAÎNEMENT ====================
print("\n🚀 Entraînement du modèle...")

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(EPOCHS):
    # Training
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

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_forex_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping à l'epoch {epoch + 1}")
            break

print(f"✅ Entraînement terminé! Meilleure val loss: {best_val_loss:.6f}")

# Charger le meilleur modèle
model.load_state_dict(torch.load('best_forex_model.pth'))

# ==================== 6. PRÉDICTION DIRECTE (pas auto-régressive) ====================
print("\n📈 Génération des prédictions pour 2024...")

model.eval()
predictions = []

# Créer toutes les séquences de test
test_sequences = []
for i in range(len(test_scaled) - SEQUENCE_LENGTH):
    test_sequences.append(test_scaled[i:i + SEQUENCE_LENGTH])

# Prédire en batch
with torch.no_grad():
    for seq in test_sequences:
        X = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
        pred = model(X).detach().cpu().numpy()[0, 0]
        predictions.append(pred)

# Ajouter les dernières prédictions pour avoir la même longueur
with torch.no_grad():
    for i in range(SEQUENCE_LENGTH):
        X = torch.FloatTensor(test_scaled[len(test_scaled) - SEQUENCE_LENGTH:]).unsqueeze(0).to(DEVICE)
        pred = model(X).detach().cpu().numpy()[0, 0]
        predictions.append(pred)

# Dénormaliser
predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
true_values = test_df['exchange_rate'].values

print(f"✅ {len(predictions)} prédictions générées")

# ==================== 7. MÉTRIQUES ====================
print("\n📊 Calcul des métriques...")

mae = mean_absolute_error(true_values, predictions)
rmse = np.sqrt(mean_squared_error(true_values, predictions))
mape = np.mean(np.abs((true_values - predictions.flatten()) / true_values)) * 100

print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAPE: {mape:.2f}%")

# ==================== 8. VISUALISATION ====================
print("\n📊 Création des graphiques...")

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

dates = test_df['date'].values

# Plot 1: Toute l'année
axes[0].plot(dates, true_values, label='Vraies valeurs', linewidth=2, color='#2E86AB', marker='o', markersize=2)
axes[0].plot(dates, predictions, label='Prédictions LSTM amélioré', linewidth=2, color='#A23B72', marker='s',
             markersize=2, alpha=0.8)
axes[0].set_title(f'EUR/USD - Prédictions 2024 (LSTM Amélioré) - MAPE: {mape:.2f}%', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Taux de change')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Zoom
zoom_days = min(90, len(dates))
axes[1].plot(dates[:zoom_days], true_values[:zoom_days], label='Vraies valeurs', linewidth=2.5, color='#2E86AB',
             marker='o', markersize=4)
axes[1].plot(dates[:zoom_days], predictions[:zoom_days], label='Prédictions LSTM amélioré', linewidth=2.5,
             color='#A23B72', marker='s', markersize=4, alpha=0.8)
axes[1].set_title('EUR/USD - Zoom sur les 90 premiers jours', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Taux de change')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Erreur
errors = true_values - predictions.flatten()
axes[2].plot(dates, errors, linewidth=1.5, color='#F18F01', alpha=0.7)
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
axes[2].fill_between(dates, errors, 0, alpha=0.3, color='#F18F01')
axes[2].set_title(f'Erreur de Prédiction (MAE: {mae:.4f})', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Erreur')
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('lstm_forex_predictions_2024_improved.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: lstm_forex_predictions_2024_improved.png")
plt.show()

# Loss curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2, color='#2E86AB')
plt.plot(val_losses, label='Validation Loss', linewidth=2, color='#A23B72')
plt.title('Évolution de la Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss_improved.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✨ Terminé!")