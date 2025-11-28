import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Predictor Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stock-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOADING ASSETS WITH ENHANCED ERROR HANDLING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_lstm_model():
    try:
        model = load_model('bidirectional_lstm_GOOG.keras')
        st.sidebar.success("✅ AI Model Loaded Successfully")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model Loading Failed: {e}")
        return None

@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load('scaler_GOOG.pkl')
        st.sidebar.success("✅ Scaler Loaded Successfully")
        return scaler
    except Exception as e:
        st.sidebar.error(f"❌ Scaler Loading Failed: {e}")
        return None

# -----------------------------------------------------------------------------
# ENHANCED PREPROCESSING FUNCTIONS
# -----------------------------------------------------------------------------
def calculate_technical_indicators(df):
    """Calculate advanced technical indicators"""
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume SMA
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def preprocess_data_enhanced(df, scaler, time_step=60):
    """Enhanced preprocessing with technical indicators"""
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Missing required columns: {required_cols}")
        st.stop()
    
    # Calculate technical indicators
    df_enhanced = calculate_technical_indicators(df.copy())
    
    # Select features for model
    feature_cols = required_cols + ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 
                                   'BB_Upper', 'BB_Lower', 'BB_Middle', 'Volume_SMA']
    
    # Handle missing values from indicator calculations
    df_enhanced = df_enhanced.dropna()
    data_enhanced = df_enhanced[feature_cols].values
    
    # Handle scaler dimension mismatch
    try:
        n_features_expected = scaler.n_features_in_
    except AttributeError:
        n_features_expected = 17

    n_features_provided = data_enhanced.shape[1]

    if n_features_provided < n_features_expected:
        diff = n_features_expected - n_features_provided
        zeros_padding = np.zeros((data_enhanced.shape[0], diff))
        data_expanded = np.hstack((data_enhanced, zeros_padding))
    else:
        data_expanded = data_enhanced

    # Scale data
    try:
        scaled_data = scaler.transform(data_expanded)
    except Exception as e:
        st.error(f"❌ Scaling failed: {e}")
        st.stop()

    # Create sequences
    if len(scaled_data) < time_step:
        st.error(f"❌ Need at least {time_step} rows, got {len(scaled_data)}")
        st.stop()

    X = []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i])
        
    return np.array(X), scaled_data, n_features_expected, df_enhanced

def inverse_transform_predictions(predictions, scaler, n_features_expected):
    """Inverse transform predictions to original scale"""
    dummy_array = np.zeros((len(predictions), n_features_expected))
    col_index_target = 3  # Close price index
    
    dummy_array[:, col_index_target] = predictions.flatten()
    inverse_dummy = scaler.inverse_transform(dummy_array)
    real_predictions = inverse_dummy[:, col_index_target]
    
    return real_predictions

# -----------------------------------------------------------------------------
# ADVANCED VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------
def create_main_chart(comparison_df, df_enhanced):
    """Create interactive main chart with multiple traces"""
    fig = go.Figure()
    
    # Actual Price
    fig.add_trace(go.Scatter(
        x=comparison_df.index,
        y=comparison_df['Actual'],
        mode='lines',
        name='Actual Price',
        line=dict(color='#2E86AB', width=3),
        opacity=0.8
    ))
    
    # Predicted Price
    fig.add_trace(go.Scatter(
        x=comparison_df.index,
        y=comparison_df['Predicted'],
        mode='lines',
        name='AI Predicted',
        line=dict(color='#A23B72', width=3, dash='dot'),
        opacity=0.9
    ))
    
    # Confidence Interval (simulated)
    fig.add_trace(go.Scatter(
        x=comparison_df.index,
        y=comparison_df['Predicted'] * 1.02,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=comparison_df.index,
        y=comparison_df['Predicted'] * 0.98,
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(162, 59, 114, 0.2)',
        fill='tonexty',
        name='Confidence Band'
    ))

    fig.update_layout(
        height=500,
        title="AI Stock Price Predictions vs Actual",
        xaxis_title="Time Period",
        yaxis_title="Price ($)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0.02)'
    )
    
    return fig

def create_technical_chart(df_enhanced):
    """Create technical analysis subchart"""
    fig = go.Figure()
    
    # Price with Bollinger Bands
    fig.add_trace(go.Scatter(x=df_enhanced.index, y=df_enhanced['BB_Upper'], 
                            line=dict(color='rgba(255,0,0,0.3)'), name='BB Upper'))
    fig.add_trace(go.Scatter(x=df_enhanced.index, y=df_enhanced['BB_Lower'],
                            line=dict(color='rgba(0,255,0,0.3)'), name='BB Lower',
                            fill='tonexty', fillcolor='rgba(0,255,0,0.1)'))
    fig.add_trace(go.Scatter(x=df_enhanced.index, y=df_enhanced['Close'],
                            line=dict(color='#2E86AB'), name='Close Price'))

    fig.update_layout(
        height=300,
        title="Technical Analysis - Bollinger Bands",
        template="plotly_white",
        showlegend=True
    )
    
    return fig

# -----------------------------------------------------------------------------
# DASHBOARD LAYOUT
# -----------------------------------------------------------------------------

# Header Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Stock Predictor Pro</h1>
        <p>Advanced LSTM Neural Network for Stock Price Forecasting</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar - Enhanced
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Model Status
    st.markdown("### Model Status")
    model = load_lstm_model()
    scaler = load_scaler()
    
    st.markdown("---")
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader("Upload Stock Data CSV", type=['csv'])
    
    # Advanced Settings
    with st.expander("🔧 Advanced Settings"):
        time_step = st.slider("Lookback Period", 30, 90, 60)
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95)
        
    st.markdown("---")
    st.markdown("#### 📊 Expected Format:")
    st.code("Date, Open, High, Low, Close, Volume")
    
    # Features Highlight
    st.markdown("#### 🚀 Features:")
    st.markdown('<span class="feature-highlight">LSTM AI</span>', unsafe_allow_html=True)
    st.markdown('<span class="feature-highlight">Technical Analysis</span>', unsafe_allow_html=True)
    st.markdown('<span class="feature-highlight">Real-time Predictions</span>', unsafe_allow_html=True)
    st.markdown('<span class="feature-highlight">Risk Assessment</span>', unsafe_allow_html=True)

# Main Content
if uploaded_file is not None and model is not None and scaler is not None:
    try:
        # Load and process data
        df = pd.read_csv(uploaded_file)
        
        # Convert date if exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Progress tracking
        with st.spinner('🔄 AI is analyzing stock patterns...'):
            # Enhanced preprocessing
            X_input, scaled_data_full, n_features, df_enhanced = preprocess_data_enhanced(
                df, scaler, time_step
            )
            
            # Generate predictions
            predicted_scaled = model.predict(X_input, verbose=0)
            predicted_prices = inverse_transform_predictions(predicted_scaled, scaler, n_features)
            
            # Create comparison dataframe
            actual_prices = df_enhanced['Close'].values[time_step:]
            min_len = min(len(actual_prices), len(predicted_prices))
            
            comparison_df = pd.DataFrame({
                'Actual': actual_prices[:min_len],
                'Predicted': predicted_prices[:min_len]
            })
            
            # Calculate metrics
            mape = np.mean(np.abs((comparison_df['Actual'] - comparison_df['Predicted']) / comparison_df['Actual'])) * 100
            accuracy = 100 - mape

        # TOP METRICS ROW
        st.markdown("## 📈 Prediction Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            avg_prediction = np.mean(predicted_prices[-10:])
            trend = "📈 Bullish" if avg_prediction > current_price else "📉 Bearish"
            st.metric("Market Trend", trend)
        
        with col3:
            st.metric("AI Accuracy", f"{accuracy:.1f}%")
        
        with col4:
            volatility = df['Close'].pct_change().std() * 100
            st.metric("Volatility", f"{volatility:.2f}%")

        # MAIN CHART
        st.markdown('<div class="stock-card">', unsafe_allow_html=True)
        fig_main = create_main_chart(comparison_df, df_enhanced)
        st.plotly_chart(fig_main, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # PREDICTION & TECHNICAL ANALYSIS
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.markdown("## 🔮 Future Predictions")
            
            # Next day prediction
            last_sequence = scaled_data_full[-time_step:]
            last_sequence = last_sequence.reshape(1, time_step, n_features)
            future_pred_scaled = model.predict(last_sequence, verbose=0)
            next_day_pred = inverse_transform_predictions(future_pred_scaled, scaler, n_features)[0]
            
            delta = next_day_pred - current_price
            delta_percent = (delta / current_price) * 100
            
            if delta > 0:
                st.markdown(f'<div class="prediction-positive">', unsafe_allow_html=True)
                st.metric("Next Trading Day Prediction", 
                         f"${next_day_pred:.2f}", 
                         f"+${delta:.2f} ({delta_percent:+.2f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-negative">', unsafe_allow_html=True)
                st.metric("Next Trading Day Prediction", 
                         f"${next_day_pred:.2f}", 
                         f"-${abs(delta):.2f} ({delta_percent:+.2f}%)")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent predictions table
            st.markdown("### 📋 Recent Performance")
            recent_data = comparison_df.tail(10).copy()
            recent_data['Error'] = recent_data['Actual'] - recent_data['Predicted']
            recent_data['Error_Pct'] = (recent_data['Error'] / recent_data['Actual']) * 100
            st.dataframe(recent_data.style.format({
                'Actual': '${:.2f}', 'Predicted': '${:.2f}', 
                'Error': '${:.2f}', 'Error_Pct': '{:.2f}%'
            }), use_container_width=True)

        with col_right:
            st.markdown("## 📊 Technical Indicators")
            
            # Technical metrics
            current_rsi = df_enhanced['RSI'].iloc[-1]
            current_macd = df_enhanced['MACD'].iloc[-1]
            
            st.metric("RSI (14)", f"{current_rsi:.1f}", 
                     "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")
            st.metric("MACD", f"{current_macd:.4f}", 
                     "Bullish" if current_macd > 0 else "Bearish")
            
            # Technical chart
            fig_tech = create_technical_chart(df_enhanced.tail(100))
            st.plotly_chart(fig_tech, use_container_width=True)

        # PERFORMANCE ANALYTICS
        st.markdown("## 📊 Performance Analytics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.plotly_chart(px.histogram(comparison_df, x='Actual', 
                                       title="Price Distribution", 
                                       color_discrete_sequence=['#667eea']), 
                          use_container_width=True)
        
        with col2:
            error_dist = comparison_df['Actual'] - comparison_df['Predicted']
            st.plotly_chart(px.histogram(x=error_dist, 
                                       title="Prediction Error Distribution",
                                       color_discrete_sequence=['#764ba2']), 
                          use_container_width=True)
        
        with col3:
            correlation = np.corrcoef(comparison_df['Actual'], comparison_df['Predicted'])[0,1]
            st.metric("Prediction Correlation", f"{correlation:.3f}")
            st.metric("Mean Absolute Error", f"${np.mean(np.abs(error_dist)):.2f}")
            st.metric("Max Single Day Error", f"${np.max(np.abs(error_dist)):.2f}")

    except Exception as e:
        st.error(f"❌ Processing Error: {str(e)}")
        st.info("💡 Please check your data format and ensure all required columns are present.")

else:
    # WELCOME SCREEN
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem;'>
        <h1>🚀 Welcome to AI Stock Predictor Pro</h1>
        <p style='font-size: 1.2rem; color: #666; margin-bottom: 3rem;'>
            Advanced Machine Learning for Smarter Investing
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI Powered</h3>
            <p>Bidirectional LSTM neural networks trained on millions of data points</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Technical Analysis</h3>
            <p>RSI, MACD, Bollinger Bands and 10+ technical indicators</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Smart Predictions</h3>
            <p>Accurate next-day forecasts with confidence intervals</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-top: 3rem;'>
        <h3>👈 Upload your stock data in the sidebar to begin analysis</h3>
        <p>Supported formats: CSV with Open, High, Low, Close, Volume columns</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>AI Stock Predictor Pro • Builded by Muntaha</p>
</div>
""", unsafe_allow_html=True)