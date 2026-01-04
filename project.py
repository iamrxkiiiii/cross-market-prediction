from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import yfinance as yf  # type: ignore
import plotly.express as px  # type: ignore
from scipy.stats import skew, kurtosis  # type: ignore
from statsmodels.tsa.stattools import adfuller  # type: ignore
from statsmodels.tsa.api import VAR  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import torch  # type: ignore
from torch_geometric.utils import from_networkx  # type: ignore
import torch.nn.functional as F  # type: ignore
from torch_geometric.nn import GATConv, GCNConv  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
import torch.nn as nn  # type: ignore
import itertools

# =========================
# 1. DATA FETCHING & CLEAN
# =========================

# Define the stock tickers
tickers = ['^GSPC', '^GDAXI', '^FCHI', '^FTSE', '^NSEI', '^N225', '^KS11', '^HSI']

# Define the start and end dates
start_date = '2007-11-06'
end_date = '2022-06-03'


def fetch_and_fill_data(
    symbol: str,
    start: str,
    end: str
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Fetch historical data from yfinance, align to full business-day range,
    forward/backward fill, and also return count of filled cells.
    """
    try:
        data = yf.download(symbol, start=start, end=end)  # type: ignore

        # Handle None or empty DataFrame explicitly
        if data is None or data.empty:
            print(f"No data returned for {symbol}")
            return None, None

        # Full date range (business days)
        full_range = pd.date_range(start=start, end=end, freq='B')

        original_data = data.copy()
        data = data.reindex(full_range)

        data_filled = data.ffill().bfill()

        # Count how many values were filled (per column)
        filled_days_count = data_filled.notna().sum() - original_data.notna().sum()

        return data_filled, filled_days_count
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None, None


# Dictionary to store the processed data and filled days count
stock_data: Dict[str, pd.DataFrame] = {}
filled_days_counts: Dict[str, pd.Series] = {}

# Fetch and fill data for each ticker
for ticker in tickers:
    data, filled_days_count = fetch_and_fill_data(ticker, start_date, end_date)
    if data is not None and filled_days_count is not None:
        stock_data[ticker] = data
        filled_days_counts[ticker] = filled_days_count
        data.to_csv(f"{ticker}_stock_data.csv")

# Plot the 'Close' price of each ticker in separate graphs
for ticker, data in stock_data.items():
    plt.figure(figsize=(14, 7))  # type: ignore
    plt.plot(data.index, data['Close'], label=f'{ticker} Close Price')  # type: ignore
    plt.title(f'{ticker} Close Price Over Time')  # type: ignore
    plt.xlabel('Date')  # type: ignore
    plt.ylabel('Close Price')  # type: ignore
    plt.legend()  # type: ignore
    plt.show()  # type: ignore

# Print the head, total count of each processed DataFrame, and the count of filled values
for ticker, data in stock_data.items():
    print(f'Head of {ticker} data:')
    print(data.head(), '\n')
    print(f'Total count of trading days for {ticker}: {len(data)}\n')
    print(f'Count of forward-filled or backward-filled days for {ticker}:')
    print(filled_days_counts[ticker], '\n')

for ticker, data in stock_data.items():
    print(f"Ticker: {ticker}, Total number of days: {data.shape[0]}, Total number of fields: {data.shape[1]}")

# Ensure datetime index
for ticker, data in stock_data.items():
    data.index = pd.to_datetime(data.index)

for ticker, data in stock_data.items():
    print(f"Ticker: {ticker}")
    print(data.head())
    print("\n")

# =========================
# 2. CLOSE SERIES & PLOTS
# =========================

ticker_close_dict: Dict[str, pd.Series] = {}

for ticker, data in stock_data.items():
    ticker_close_series = data['Close']
    ticker_close_dict[ticker] = ticker_close_series

for ticker, series in ticker_close_dict.items():
    print(f"Ticker: {ticker}")
    print(series.head())
    print(f"Length: {len(series)}\n")

# Plot with Plotly
for ticker, series in ticker_close_dict.items():
    df = series.reset_index()
    df.columns = ['date', 'Close']

    fig = px.line(df, x='date', y='Close',  # type: ignore
                  labels={'date': 'Date', 'close': 'Close Stock'})
    fig.update_traces(marker_line_width=2, opacity=0.8)  # type: ignore
    fig.update_layout(  # type: ignore
        title_text=f'Stock Close Price Chart for {ticker}',
        plot_bgcolor='white',
        font_size=15,
        font_color='black'
    )
    fig.update_xaxes(showgrid=False)  # type: ignore
    fig.update_yaxes(showgrid=False)  # type: ignore
    fig.show()  # type: ignore

# =========================
# 3. REALIZED VOLATILITY
# =========================

def calculate_realized_volatility(data: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Calculate realized volatility for the given data using squared returns.
    """
    returns = data['Close'].pct_change()
    squared_returns = returns ** 2
    realized_variance = squared_returns.rolling(window=window).sum()
    realized_volatility = np.sqrt(realized_variance)
    return realized_volatility.dropna()  # type: ignore


realized_vol_dict: Dict[str, pd.Series] = {}
for ticker, data in stock_data.items():
    realized_volatility = calculate_realized_volatility(data)
    realized_vol_dict[ticker] = realized_volatility

for ticker, series in realized_vol_dict.items():
    print(f"Ticker: {ticker}")
    print(series.head())
    print(f"Length: {len(series)}\n")

# Plot realized volatility
for ticker, series in realized_vol_dict.items():
    df = series.reset_index()
    df.columns = ['date', 'realized_volatility']

    fig = px.line(df, x='date', y='realized_volatility',  # type: ignore
                  labels={'date': 'Date', 'realized_volatility': 'Realized Volatility'})
    fig.update_traces(marker_line_width=2, opacity=0.8)  # type: ignore
    fig.update_layout(  # type: ignore
        title_text=f'Realized Volatility Chart for {ticker}',
        plot_bgcolor='white',
        font_size=15,
        font_color='black'
    )
    fig.update_xaxes(showgrid=False)  # type: ignore
    fig.update_yaxes(showgrid=False)  # type: ignore
    fig.show()  # type: ignore

# =========================
# 4. DESCRIPTIVE STATS
# =========================

def calculate_descriptive_statistics(realized_volatility: pd.Series) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for the given realized volatility series.
    """
    mean_value = realized_volatility.mean()
    std_dev = realized_volatility.std()
    skewness_value = skew(realized_volatility)
    kurtosis_value = kurtosis(realized_volatility, fisher=False)
    adf_result = adfuller(realized_volatility.dropna())  # type: ignore
    adf_statistic = adf_result[0]  # type: ignore
    adf_p_value = adf_result[1]  # type: ignore

    return {
        'Mean': mean_value,
        'Standard Deviation': std_dev,
        'Skewness': skewness_value,
        'Kurtosis': kurtosis_value,
        'ADF Statistic': adf_statistic,
        'ADF p-value': adf_p_value
    }


descriptive_stats_dict: Dict[str, Dict[str, Any]] = {}
for ticker, series in realized_vol_dict.items():
    descriptive_stats = calculate_descriptive_statistics(series)
    descriptive_stats_dict[ticker] = descriptive_stats

for ticker, stats in descriptive_stats_dict.items():
    print(f"Descriptive Statistics for {ticker}:")
    for stat_name, value in stats.items():
        print(f"{stat_name}: {value}")
    print()

# =========================
# 5. TRAIN / VAL / TEST SPLITS
# =========================

data_splits: Dict[str, Dict[str, Any]] = {}

for ticker, realized_volatility in realized_vol_dict.items():
    df = realized_volatility.reset_index()
    df.columns = ['date', 'realized_volatility']

    train_data, temp_data = train_test_split(df, test_size=0.5, shuffle=False)  # type: ignore
    validation_data, test_data = train_test_split(temp_data, test_size=0.6, shuffle=False)  # type: ignore

    data_splits[ticker] = {
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    }

for ticker, splits in data_splits.items():
    print(f"Ticker: {ticker}")
    print(f"Training Set Size: {len(splits['train'])}")
    print(f"Validation Set Size: {len(splits['validation'])}")
    print(f"Test Set Size: {len(splits['test'])}\n")

# =========================
# 6. SPILLOVER (VAR-FEVD)
# =========================

def calculate_spillover_index(realized_vol_dict: Dict[str, pd.Series],
                              lag_order: int = 2,
                              forecast_horizon: int = 10) -> pd.DataFrame:
    """
    Calculate the volatility spillover index using the Diebold-Yilmaz methodology.
    """
    combined_data = pd.DataFrame(realized_vol_dict).dropna()  # type: ignore

    model = VAR(combined_data)  # type: ignore
    var_result = model.fit(lag_order)  # type: ignore
    fevd = var_result.fevd(forecast_horizon)

    n = len(realized_vol_dict)
    spillover_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                spillover_matrix[i, j] = fevd.decomp[j][:, i].sum() / fevd.decomp[j].sum()

    spillover_index = pd.DataFrame(
        spillover_matrix,
        index=list(realized_vol_dict.keys()),
        columns=list(realized_vol_dict.keys())
    )
    return spillover_index * 100


train_realized_vol_dict: Dict[str, pd.Series] = {
    ticker: splits['train']['realized_volatility']
    for ticker, splits in data_splits.items()
}

spillover_index_train = calculate_spillover_index(train_realized_vol_dict)

print("Spillover Index Matrix (Training Data):")
print(spillover_index_train)

n = len(train_realized_vol_dict)
total_spillover_index_train = spillover_index_train.to_numpy().sum() / (n**2 - n)
print(f"\nTotal Spillover Index (Training Data): {total_spillover_index_train:.2f}%")

plt.figure(figsize=(10, 8))  # type: ignore
plt.title("Volatility Spillover Index Heatmap (Training Data)")  # type: ignore
sns.heatmap(spillover_index_train, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)  # type: ignore
plt.show()  # type: ignore


def create_spillover_graph(spillover_matrix: pd.DataFrame) -> Any:  # type: ignore
    """
    Create a directed graph from the spillover matrix.
    """
    G = nx.DiGraph()  # type: ignore

    for node in spillover_matrix.columns:
        G.add_node(node)  # type: ignore

    for i, source in enumerate(spillover_matrix.columns):
        for j, target in enumerate(spillover_matrix.columns):
            if i != j:
                weight = spillover_matrix.iloc[i, j]
                if weight > 0:  # type: ignore
                    G.add_edge(source, target, weight=weight)  # type: ignore

    return G  # type: ignore


G = create_spillover_graph(spillover_index_train)

plt.figure(figsize=(12, 8))  # type: ignore
pos = nx.spring_layout(G)  # type: ignore
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000,  # type: ignore
        font_size=12, font_weight='bold', edge_color='gray', width=2)
edge_labels = nx.get_edge_attributes(G, 'weight')  # type: ignore
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)  # type: ignore
plt.title('Volatility Spillover Directed Graph (Training Data)')  # type: ignore
plt.show()  # type: ignore

# Test & Validation spillovers
test_realized_vol_dict: Dict[str, pd.Series] = {
    ticker: splits['test']['realized_volatility']
    for ticker, splits in data_splits.items()
}
validation_realized_vol_dict: Dict[str, pd.Series] = {
    ticker: splits['validation']['realized_volatility']
    for ticker, splits in data_splits.items()
}

spillover_index_test = calculate_spillover_index(test_realized_vol_dict)
print("Spillover Index Matrix (Test Data):")
print(spillover_index_test)

n_test = len(test_realized_vol_dict)
total_spillover_index_test = spillover_index_test.to_numpy().sum() / (n_test**2 - n_test)
print(f"\nTotal Spillover Index (Test Data): {total_spillover_index_test:.2f}%")

plt.figure(figsize=(10, 8))  # type: ignore
plt.title("Volatility Spillover Index Heatmap (Test Data)")  # type: ignore
sns.heatmap(spillover_index_test, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)  # type: ignore
plt.show()  # type: ignore

G_test = create_spillover_graph(spillover_index_test)
plt.figure(figsize=(12, 8))  # type: ignore
pos_test = nx.spring_layout(G_test)  # type: ignore
nx.draw(G_test, pos_test, with_labels=True, node_color='lightgreen', node_size=3000,  # type: ignore
        font_size=12, font_weight='bold', edge_color='gray', width=2)
edge_labels_test = nx.get_edge_attributes(G_test, 'weight')  # type: ignore
nx.draw_networkx_edge_labels(G_test, pos_test, edge_labels=edge_labels_test, font_size=10)  # type: ignore
plt.title('Volatility Spillover Directed Graph (Test Data)')  # type: ignore
plt.show()  # type: ignore

spillover_index_validation = calculate_spillover_index(validation_realized_vol_dict)
print("Spillover Index Matrix (Validation Data):")
print(spillover_index_validation)

n_val = len(validation_realized_vol_dict)
total_spillover_index_validation = spillover_index_validation.to_numpy().sum() / (n_val**2 - n_val)
print(f"\nTotal Spillover Index (Validation Data): {total_spillover_index_validation:.2f}%")

plt.figure(figsize=(10, 8))  # type: ignore
plt.title("Volatility Spillover Index Heatmap (Validation Data)")  # type: ignore
sns.heatmap(spillover_index_validation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)  # type: ignore
plt.show()  # type: ignore

G_validation = create_spillover_graph(spillover_index_validation)
plt.figure(figsize=(12, 8))  # type: ignore
pos_validation = nx.spring_layout(G_validation)  # type: ignore
nx.draw(G_validation, pos_validation, with_labels=True, node_color='lightcoral', node_size=3000,  # type: ignore
        font_size=12, font_weight='bold', edge_color='gray', width=2)
edge_labels_validation = nx.get_edge_attributes(G_validation, 'weight')  # type: ignore
nx.draw_networkx_edge_labels(G_validation, pos_validation, edge_labels=edge_labels_validation, font_size=10)  # type: ignore
plt.title('Volatility Spillover Directed Graph (Validation Data)')  # type: ignore
plt.show()  # type: ignore

# =========================
# 7. CONVERT TO PYTORCH GEOMETRIC DATA
# =========================

def networkx_to_pyg_data(G: Any, realized_vol_dict: Dict[str, pd.Series]) -> Any:  # type: ignore
    """
    Convert a NetworkX graph and realized volatility dictionary
    into PyTorch Geometric Data format.
    """
    data = from_networkx(G)  # type: ignore

    node_features = []
    for node in G.nodes():
        vol_series = realized_vol_dict[node].values
        node_features.append(torch.tensor(vol_series, dtype=torch.float).view(-1, 1))  # type: ignore

    data.x = torch.cat(node_features, dim=1)  # type: ignore
    return data  # type: ignore


train_data = networkx_to_pyg_data(G, train_realized_vol_dict)
validation_data = networkx_to_pyg_data(G_validation, validation_realized_vol_dict)
test_data = networkx_to_pyg_data(G_test, test_realized_vol_dict)

# =========================
# 8. GCN + GAT MODEL + GRID SEARCH
# =========================

class GCN_GAT_Model(torch.nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int = 2, dropout_p: float = 0.5) -> None:  # type: ignore
        super(GCN_GAT_Model, self).__init__()  # type: ignore
        self.num_layers = num_layers
        self.gcn_layers = torch.nn.ModuleList()
        self.gat_layers = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(p=dropout_p)

        self.gcn_layers.append(GCNConv(node_feature_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))

        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False))

        self.fc = torch.nn.Linear(hidden_dim, 8)

    def forward(self, data: Any) -> Any:  # type: ignore
        x, edge_index = data.x, data.edge_index  # type: ignore

        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        out = self.fc(x)
        return out


node_feature_dim: int = train_data.x.shape[1]  # type: ignore
hidden_dim_list = [32, 64]
num_heads_list = [2, 4]
num_layers_list = [2, 3]
learning_rates = [0.001, 0.0005]
dropout_rates = [0.1, 0.3]

param_combinations = list(itertools.product(
    hidden_dim_list, num_heads_list, num_layers_list, learning_rates, dropout_rates
))
best_val_loss = float('inf')
best_params = None

all_train_loss_values = []
all_validation_loss_values = []

criterion = torch.nn.MSELoss()

for params in param_combinations:
    hidden_dim, num_heads, num_layers, lr, dropout_rate = params

    model = GCN_GAT_Model(node_feature_dim, hidden_dim, num_heads, num_layers, dropout_p=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_values = []
    validation_loss_values = []
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data)
        loss = criterion(out[:-1], train_data.x[1:])
        loss.backward()
        optimizer.step()  # type: ignore
        train_loss_values.append(loss.item())  # type: ignore

        model.eval()
        with torch.no_grad():
            validation_out = model(validation_data)
            validation_loss = criterion(validation_out[:-1], validation_data.x[1:])
            validation_loss_values.append(validation_loss.item())  # type: ignore

        if epoch % 10 == 0:
            print(f"[GCN+GAT] Params {params}, Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {validation_loss.item():.6f}")

    all_train_loss_values.append(train_loss_values)  # type: ignore
    all_validation_loss_values.append(validation_loss_values)  # type: ignore

    final_val_loss = validation_loss_values[-1]  # type: ignore
    if final_val_loss < best_val_loss:  # type: ignore
        best_val_loss = final_val_loss  # type: ignore
        best_params = params  # type: ignore

best_index = param_combinations.index(best_params)  # type: ignore

plt.figure(figsize=(12, 6))  # type: ignore
plt.plot(range(num_epochs), all_train_loss_values[best_index], label='Training Loss', color='blue')  # type: ignore
plt.plot(range(num_epochs), all_validation_loss_values[best_index], label='Validation Loss', color='red')  # type: ignore
plt.title('Training and Validation Loss Over Epochs (Best GCN+GAT Config)')  # type: ignore
plt.xlabel('Epochs')  # type: ignore
plt.ylabel('Loss')  # type: ignore
plt.legend()  # type: ignore
plt.grid(True)  # type: ignore
plt.show()  # type: ignore

print(f"Best Hyperparameters (GCN+GAT): Hidden Dim: {best_params[0]}, Num Heads: {best_params[1]}, Num Layers: {best_params[2]}, Learning Rate: {best_params[3]}, Dropout Rate: {best_params[4]}")  # type: ignore

def mean_absolute_percentage_error(y_true: Any, y_pred: Any) -> float:  # type: ignore
    y_true, y_pred = np.array(y_true), np.array(y_pred)  # type: ignore
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100  # type: ignore

validation_targets = validation_data.x

def calculate_all_metrics(actuals: Any, predictions: Any, h: int) -> pd.DataFrame:  # type: ignore
    T = actuals.shape[0]  # type: ignore
    if predictions.shape[0] > T - h:  # type: ignore
        predictions = predictions[:T - h]  # type: ignore
    actuals = actuals[h:]  # type: ignore

    metrics_list = []

    for idx in range(actuals.shape[1]):  # type: ignore
        act = actuals[:, idx].detach().cpu().numpy()  # type: ignore
        pred = predictions[:, idx].detach().cpu().numpy()  # type: ignore

        mse = mean_squared_error(act, pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(act, pred)
        mafe = np.mean(np.abs(pred - act))  # type: ignore

        metrics_list.append({  # type: ignore
            'Index': f"Index {idx+1}",
            'MAFE': mafe,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df

# Re-train best GCN+GAT model on train_data
best_hidden, best_heads, best_layers, best_lr, best_dropout = best_params  # type: ignore
best_model = GCN_GAT_Model(node_feature_dim, best_hidden, best_heads, best_layers, dropout_p=best_dropout)  # type: ignore
best_optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)  # type: ignore

num_epochs = 50
for epoch in range(num_epochs):
    best_model.train()
    best_optimizer.zero_grad()
    out = best_model(train_data)
    loss = criterion(out[:-1], train_data.x[1:])
    loss.backward()
    best_optimizer.step()  # type: ignore

best_model.eval()
with torch.no_grad():
    validation_out = best_model(validation_data)

horizons = [1, 5, 10, 22]
metrics_results_per_horizon = {}

for h in horizons:
    metrics_df = calculate_all_metrics(validation_targets, validation_out, h)
    metrics_results_per_horizon[f"Metrics for horizon {h}"] = metrics_df

for horizon, metrics_df in metrics_results_per_horizon.items():  # type: ignore
    print(f"{horizon}:")
    print(metrics_df.to_string(index=False))  # type: ignore

# =========================
# 9. BASELINE MLP MODEL + GRID SEARCH
# =========================

class BaselineMLPModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float) -> None:  # type: ignore
        super(BaselineMLPModel, self).__init__()  # type: ignore
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # type: ignore
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # type: ignore
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)  # type: ignore
        self.fc_final = nn.Linear(hidden_dim, 1)  # type: ignore
        self.dropout = nn.Dropout(p=dropout_rate)  # type: ignore

    def forward(self, data: Any) -> Any:  # type: ignore
        x = data.x  # type: ignore
        x = F.relu(self.fc1(x)); x = self.dropout(x)
        x = F.relu(self.fc2(x)); x = self.dropout(x)
        x = F.relu(self.fc3(x)); x = self.dropout(x)
        out = self.fc_final(x)
        return out

hidden_dim_values = [32, 64, 128]
learning_rate_values = [0.0001, 0.001, 0.01]
dropout_rate_values = [0.3, 0.5, 0.7]

param_combinations = list(itertools.product(hidden_dim_values, learning_rate_values, dropout_rate_values))
best_val_loss = float('inf')
best_params = None
all_train_loss_values = []
all_validation_loss_values = []

for params in param_combinations:
    hidden_dim, lr, dropout_rate = params

    baseline_model = BaselineMLPModel(input_dim=train_data.x.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    train_loss_values = []
    validation_loss_values = []
    num_epochs = 50
    for epoch in range(num_epochs):
        baseline_model.train()
        optimizer.zero_grad()
        out = baseline_model(train_data)
        loss = criterion(out[:-1], train_data.x[1:])
        loss.backward()
        optimizer.step()  # type: ignore
        train_loss_values.append(loss.item())  # type: ignore

        baseline_model.eval()
        with torch.no_grad():
            validation_out = baseline_model(validation_data)
            validation_loss = criterion(validation_out[:-1], validation_data.x[1:])
            validation_loss_values.append(validation_loss.item())  # type: ignore

        if epoch % 10 == 0:
            print(f"[MLP] Params {params}, Epoch {epoch}, Training Loss: {loss.item():.6f}, Validation Loss: {validation_loss.item():.6f}")

    all_train_loss_values.append(train_loss_values)  # type: ignore
    all_validation_loss_values.append(validation_loss_values)  # type: ignore

    final_val_loss = validation_loss_values[-1]  # type: ignore
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss  # type: ignore
        best_params = params

best_index = param_combinations.index(best_params)  # type: ignore

plt.figure(figsize=(12, 6))  # type: ignore
plt.plot(range(num_epochs), all_train_loss_values[best_index], label='Training Loss', color='blue')  # type: ignore
plt.plot(range(num_epochs), all_validation_loss_values[best_index], label='Validation Loss', color='red')  # type: ignore
plt.title('Training and Validation Loss Over Epochs (Best MLP Config)')  # type: ignore
plt.xlabel('Epochs')  # type: ignore
plt.ylabel('Loss')  # type: ignore
plt.legend()  # type: ignore
plt.grid(True)  # type: ignore
plt.show()  # type: ignore

print(f"Best Hyperparameters (MLP): Hidden Dim: {best_params[0]}, Learning Rate: {best_params[1]}, Dropout Rate: {best_params[2]}")  # type: ignore

def calculate_metrics_for_all_indices(actuals: Any, predictions: Any, h: int) -> Any:  # type: ignore
    """
    Calculate evaluation metrics (MAFE, MSE, RMSE, MAPE) for each index at a given horizon h.
    """
    T = actuals.shape[0]  # type: ignore
    if predictions.shape[0] > T - h:  # type: ignore
        predictions = predictions[:T - h]  # type: ignore
    actuals = actuals[h:]  # type: ignore

    metrics_results: Any = {}  # type: ignore

    for idx in range(actuals.shape[1]):  # type: ignore
        actual = actuals[:, idx]  # type: ignore
        prediction = predictions[:, idx] if predictions.shape[1] > 1 else predictions.view(-1)  # type: ignore

        eps = 1e-8
        mafe = torch.mean(torch.abs(prediction - actual)).item()  # type: ignore
        mse = F.mse_loss(prediction, actual).item()  # type: ignore
        rmse = torch.sqrt(F.mse_loss(prediction, actual)).item()  # type: ignore
        mape = (torch.mean(torch.abs((actual - prediction) / (actual + eps))) * 100).item()  # type: ignore

        metrics_results[f"Index {idx+1}"] = {
            "MAFE": mafe,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape
        }

    return metrics_results

validation_targets2 = validation_data.x  # type: ignore

baseline_model.eval()  # type: ignore
with torch.no_grad():
    validation_out2 = baseline_model(validation_data)  # type: ignore

horizons = [1, 5, 10, 22]
metrics_results_per_horizon = {}

for h in horizons:
    metrics_results = calculate_metrics_for_all_indices(validation_targets2, validation_out2, h)  # type: ignore
    metrics_results_per_horizon[f"Metrics for horizon {h}"] = metrics_results  # type: ignore

for horizon, metrics_by_index in metrics_results_per_horizon.items():  # type: ignore
    print(f"Metrics for horizon {horizon}:")
    print(f"  {'Index':<8} {'MAFE':<10} {'MSE':<10} {'RMSE':<10} {'MAPE':<12}")
    for index, metrics in metrics_by_index.items():  # type: ignore
        print(f"{index:<8} {metrics['MAFE']:<10.6f} {metrics['MSE']:<10.6f} {metrics['RMSE']:<10.6f} {metrics['MAPE']:<12.6f} ")  # type: ignore
