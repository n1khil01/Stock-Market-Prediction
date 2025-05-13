# Stock Market Prediction Dashboard

A machine learning-powered dashboard for stock market analysis and prediction using LSTM (Long Short-Term Memory) neural networks.

## Features

- Real-time stock data fetching from Yahoo Finance
- Interactive dashboard with Streamlit
- LSTM-based price prediction
- Historical price visualization
- Volatility analysis
- Model performance metrics
- Customizable model parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-market-prediction.git
cd stock-market-prediction
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run finance-stock-market.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Model Parameters

- **Training Epochs**: Number of times the model goes through the training data
- **Sequence Length**: Number of historical days used to predict the next day

## Supported Stocks

- Apple Inc. (AAPL)
- Microsoft (MSFT)
- Google (GOOGL)
- Amazon (AMZN)
- Tesla (TSLA)
- Meta (Facebook) (META)
- Netflix (NFLX)
- NVIDIA (NVDA)

## Technologies Used

- Python
- Streamlit
- PyTorch
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## License

This project is licensed under the MIT License - see the LICENSE file for details. 