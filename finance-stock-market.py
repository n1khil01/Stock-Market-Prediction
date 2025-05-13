import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import streamlit as st
from datetime import datetime, timedelta
import time

# Set page configuration
st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1E88E5;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
    }
    .stSubheader {
        color: #424242;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Set plot style
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Cache the data fetching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance with caching"""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data found for {symbol}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# Cache the model
@st.cache_resource
def create_model(input_size, hidden_size=50, num_layers=2):
    """Create and cache the LSTM model"""
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    return model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class StockPredictor:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        self.data = fetch_stock_data(self.symbol, self.start_date, self.end_date)
        return self.data
    
    def prepare_data(self, sequence_length=60):
        """Prepare data for model training"""
        if self.data is None or self.data.empty:
            return None, None
            
        # Calculate daily returns and volatility
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
        
        # Select features for prediction
        features = ['Close', 'Volume', 'Returns', 'Volatility']
        data = self.data[features].dropna()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences for training
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict closing price
            
        return torch.FloatTensor(X), torch.FloatTensor(y).view(-1, 1)
    
    def build_model(self, input_size):
        """Build and compile the LSTM model"""
        self.model = create_model(input_size=input_size)
        self.model.to(self.device)
        return self.model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        if X_train is None or y_train is None:
            return
            
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Move data to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
    
    def predict(self, X):
        """Make predictions"""
        if X is None:
            return None
            
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.model(X)
            predictions = predictions.cpu().numpy()
            return self.scaler.inverse_transform(np.concatenate([predictions, np.zeros((len(predictions), 3))], axis=1))[:, 0]
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        if X_test is None or y_test is None:
            return {'MSE': 0, 'RMSE': 0, 'R2': 0}
            
        predictions = self.predict(X_test)
        actual = self.scaler.inverse_transform(np.concatenate([y_test.cpu().numpy(), np.zeros((len(y_test), 3))], axis=1))[:, 0]
        
        mse = mean_squared_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        return {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R2': r2
        }

def create_dashboard(predictor, test_data, predictions, X_test, y_test):
    """Create an interactive dashboard using Streamlit"""
    # Title with emoji
    st.title('📈 Stock Market Prediction Dashboard')
    
    # Sidebar for stock selection
    with st.sidebar:
        st.header('Stock Selection')
        popular_stocks = {
            'Apple Inc.': 'AAPL',
            'Microsoft': 'MSFT',
            'Google': 'GOOGL',
            'Amazon': 'AMZN',
            'Tesla': 'TSLA',
            'Meta (Facebook)': 'META',
            'Netflix': 'NFLX',
            'NVIDIA': 'NVDA'
        }
        
        selected_stock = st.selectbox(
            'Select a Stock',
            options=list(popular_stocks.keys()),
            index=0  # Default to Apple
        )
        
        # Add date range selector
        st.header('Date Range')
        start_date = st.date_input(
            'Start Date',
            value=datetime(2018, 1, 1),
            max_value=datetime.now()
        )
        end_date = st.date_input(
            'End Date',
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        # Add model parameters
        st.header('Model Parameters')
        epochs = st.slider(
            'Training Epochs',
            min_value=10,
            max_value=100,
            value=50,
            help='Number of times the model goes through the training data. More epochs = more training time but potentially better predictions.'
        )
        sequence_length = st.slider(
            'Sequence Length',
            min_value=30,
            max_value=90,
            value=60,
            help='Number of historical days used to predict the next day. Longer sequences = more historical context but more complex calculations.'
        )
    
    # Get the stock symbol
    symbol = popular_stocks[selected_stock]
    
    # Update data if stock selection changes
    if symbol != predictor.symbol:
        with st.spinner(f'Analyzing {selected_stock} data...'):
            predictor.symbol = symbol
            data = predictor.fetch_data()
            if data is not None:
                X, y = predictor.prepare_data(sequence_length=sequence_length)
                
                if X is not None and y is not None:
                    # Split data into training and testing sets
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Build and train the model
                    predictor.build_model(input_size=X.shape[2])
                    predictor.train_model(X_train, y_train, epochs=epochs)
                    
                    # Make predictions
                    predictions = predictor.predict(X_test)
                    test_data = data.iloc[train_size:]
                else:
                    st.error("Error preparing data for the selected stock")
                    return
            else:
                st.error("Error fetching data for the selected stock")
                return
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stock information header
        st.markdown(f"### {selected_stock} ({symbol})")
        
        # Current price and change
        current_price = float(predictor.data['Close'].iloc[-1])
        price_change = float(predictor.data['Close'].pct_change().iloc[-1] * 100)
        price_color = "green" if price_change >= 0 else "red"
        
        st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>
                <h3 style='margin: 0; color: #424242;'>Current Price: ${current_price:.2f}</h3>
                <p style='color: {price_color}; margin: 0;'>Change: {price_change:+.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Model metrics in cards
        metrics = predictor.evaluate_model(X_test, y_test)
        st.markdown("### Model Performance")
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{metrics['RMSE']:.2f}</div>
                <div class='metric-label'>RMSE</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{metrics['R2']:.2f}</div>
                <div class='metric-label'>R² Score</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Historical prices plot
    st.subheader('Historical Stock Prices')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictor.data.index, predictor.data['Close'], label='Actual', linewidth=2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    
    # Predictions plot
    st.subheader('Model Predictions')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_data.index[-len(predictions):], test_data['Close'].iloc[-len(predictions):], 
            label='Actual', linewidth=2)
    ax.plot(test_data.index[-len(predictions):], predictions, label='Predicted', 
            linewidth=2, linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    
    # Volatility analysis
    st.subheader('Volatility Analysis')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictor.data.index, predictor.data['Volatility'], 
            label='Historical Volatility', linewidth=2, color='purple')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    
    # Additional statistics
    st.subheader('Additional Statistics')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Daily Volume", 
                 f"{float(predictor.data['Volume'].mean()):,.0f}")
    with col2:
        st.metric("Highest Price", 
                 f"${float(predictor.data['High'].max()):.2f}")
    with col3:
        st.metric("Lowest Price", 
                 f"${float(predictor.data['Low'].min()):.2f}")

def main():
    # Set parameters
    start_date = '2018-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Initialize with default stock (will be changed by user selection)
    predictor = StockPredictor('AAPL', start_date, end_date)
    
    # Only show initial loading spinner on first run
    if 'initial_load' not in st.session_state:
        with st.spinner('Loading initial data...'):
            data = predictor.fetch_data()
            if data is not None:
                # Prepare data
                X, y = predictor.prepare_data()
                
                if X is not None and y is not None:
                    # Split data into training and testing sets
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Build and train the model
                    predictor.build_model(input_size=X.shape[2])
                    predictor.train_model(X_train, y_train)
                    
                    # Make predictions
                    predictions = predictor.predict(X_test)
                    
                    # Create and display the dashboard
                    create_dashboard(predictor, data.iloc[train_size:], predictions, X_test, y_test)
                else:
                    st.error("Error preparing initial data")
            else:
                st.error("Error fetching initial data")
        st.session_state.initial_load = True
    else:
        # For subsequent runs, just create the dashboard
        data = predictor.fetch_data()
        if data is not None:
            X, y = predictor.prepare_data()
            if X is not None and y is not None:
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]
                predictor.build_model(input_size=X.shape[2])
                predictor.train_model(X_train, y_train)
                predictions = predictor.predict(X_test)
                create_dashboard(predictor, data.iloc[train_size:], predictions, X_test, y_test)
            else:
                st.error("Error preparing initial data")
        else:
            st.error("Error fetching initial data")

if __name__ == "__main__":
    main()
