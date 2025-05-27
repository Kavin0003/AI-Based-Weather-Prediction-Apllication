# Enhanced Weather Forecasting Application
# Uses real OpenWeatherMap data and advanced modeling for accurate predictions

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import requests
import datetime
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
import pytz
from meteostat import Point, Daily
import holidays




class WeatherData:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY") or "fc3cfaf056bdc92c4a8f43f9858a51a6"
        self.base_url = "http://api.openweathermap.org/data/2.5/weather?"
        self.forecast_url = "http://api.openweathermap.org/data/2.5/forecast?"
        self.historical_url = "http://api.openweathermap.org/data/2.5/onecall/timemachine?"
        
    def get_current_weather(self, city):
        """Get current weather data for a city with error handling"""
        try:
            complete_url = f"{self.base_url}q={city}&appid={self.api_key}&units=metric"
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            if not all(key in data for key in ['main', 'wind', 'dt']):
                raise ValueError("Invalid API response structure")
                
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg', 0),
                'clouds': data['clouds'].get('all', 0),
                'rain': data.get('rain', {}).get('1h', 0),
                'snow': data.get('snow', {}).get('1h', 0),
                'timestamp': datetime.datetime.fromtimestamp(data['dt']),
                'city': city
            }
        except Exception as e:
            print(f"Error getting current weather: {str(e)}")
            return None
    
    def get_historical_data(self, city, lat, lon, days=30):
        """Get historical weather data for a location"""
        try:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Use Meteostat as a fallback (more reliable historical data)
            location = Point(lat, lon)
            data = Daily(location, start_date, end_date)
            data = data.fetch()
            
            if data.empty:
                raise ValueError("No historical data available")
                
            # Process data
            data = data.reset_index()
            data['city'] = city
            data.rename(columns={
                'tavg': 'temperature',
                'wspd': 'wind_speed',
                'wdir': 'wind_deg',
                'pres': 'pressure'
            }, inplace=True)
            
            # Add missing columns
            if 'humidity' not in data:
                data['humidity'] = 50  # Default value if not available
                
            return data[['time', 'city', 'temperature', 'humidity', 
                        'pressure', 'wind_speed', 'wind_deg']]
                        
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            return None
    
    def get_forecast(self, city):
        """Get 5-day weather forecast for a city with error handling"""
        try:
            complete_url = f"{self.forecast_url}q={city}&appid={self.api_key}&units=metric&cnt=40"
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Validate response
            if 'list' not in data or len(data['list']) == 0:
                raise ValueError("Invalid forecast data")
                
            forecasts = []
            for item in data['list']:
                forecasts.append({
                    'timestamp': datetime.datetime.fromtimestamp(item['dt']),
                    'temperature': item['main']['temp'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'wind_speed': item['wind']['speed'],
                    'wind_deg': item['wind'].get('deg', 0),
                    'clouds': item['clouds'].get('all', 0),
                    'rain': item.get('rain', {}).get('3h', 0),
                    'snow': item.get('snow', {}).get('3h', 0),
                    'weather_main': item['weather'][0]['main'],
                    'weather_desc': item['weather'][0]['description']
                })
                
            return pd.DataFrame(forecasts)
        except Exception as e:
            print(f"Error getting forecast: {str(e)}")
            return None
    
    def enrich_features(self, df):
        """Add meaningful features to the dataset"""
        if 'timestamp' not in df.columns:
            raise ValueError("Dataframe must have 'timestamp' column")
            
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['weekday'] = df['timestamp'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        # Weather interaction features
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['wind_chill'] = 13.12 + 0.6215*df['temperature'] - 11.37*(df['wind_speed']**0.16) + 0.3965*df['temperature']*(df['wind_speed']**0.16)
        
        # Add holiday information
        us_holidays = holidays.US()
        df['is_holiday'] = df['timestamp'].apply(lambda x: x in us_holidays).astype(int)
        
        # Lag features (previous day's weather)
        for lag in [1, 2, 3]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
            df[f'pressure_lag_{lag}'] = df['pressure'].shift(lag)
            
        # Rolling features
        df['temp_rolling_3'] = df['temperature'].rolling(3).mean()
        df['pressure_rolling_3'] = df['pressure'].rolling(3).mean()
        
        return df.dropna()

class WeatherModel:
    def __init__(self, model_path='weather_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.features = None
        self.target = 'temperature'
        self.metrics = {}
        
    def load_model(self):
        """Load trained model and scaler if they exist"""
        if os.path.exists(self.model_path):
            try:
                saved_data = joblib.load(self.model_path)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.features = saved_data['features']
                self.metrics = saved_data.get('metrics', {})
                return True
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
        return False
    
    def save_model(self):
        """Save trained model with scaler and feature info"""
        if self.model:
            saved_data = {
                'model': self.model,
                'scaler': self.scaler,
                'features': self.features,
                'metrics': self.metrics
            }
            joblib.dump(saved_data, self.model_path)
    
    def train(self, data):
        """Train weather forecasting model with proper validation"""
        try:
            # Feature selection
            X = data.drop(columns=[self.target, 'timestamp', 'city'], errors='ignore')
            y = data[self.target]
            
            # Select top features using mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=X.columns)
            top_features = mi_scores.nlargest(15).index.tolist()
            self.features = top_features
            
            # Train-test split with temporal validation
            X = X[top_features]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False, random_state=42)
            
            # Create preprocessing pipeline
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)])
            
            # Create and train model pipeline
            self.model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    early_stopping_rounds=10
                ))])
            
            # Train with early stopping
            self.model.fit(
                X_train, y_train,
                regressor__eval_set=[(X_test, y_test)],
                regressor__verbose=False)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            self.metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': self.model.score(X_test, y_test)
            }
            
            self.save_model()
            return self.metrics
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise
    
    def predict(self, input_data):
        """Make predictions with proper input validation"""
        if not self.model:
            raise ValueError("Model not trained. Please train the model first.")
            
        try:
            # Ensure we have all required features
            missing_features = set(self.features) - set(input_data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
                
            # Select only the features we need
            input_data = input_data[self.features]
            
            # Make prediction
            return self.model.predict(input_data)
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Weather Forecasting")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        
        self.weather_data = WeatherData()
        self.weather_model = WeatherModel()
        self.current_city = None
        self.historical_data = None
        
        self.create_widgets()
        
        # Try to load model
        try:
            if not self.weather_model.load_model():
                self.status_var.set("Model not trained. Please train with historical data.")
            else:
                self.status_var.set(f"Model loaded (MAE: {self.weather_model.metrics.get('mae', '?'):.2f}°C)")
        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
    
    def create_widgets(self):
        """Create all UI elements"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(header_frame, text="Advanced Weather Forecasting", 
                 style='Header.TLabel').pack(side='left')
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(header_frame, textvariable=self.status_var).pack(side='right')
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.forecast_tab = self.create_forecast_tab()
        self.data_tab = self.create_data_tab()
        self.model_tab = self.create_model_tab()
        
        self.notebook.add(self.forecast_tab, text="Forecast")
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.model_tab, text="Model")
    
    def create_forecast_tab(self):
        """Create the forecast tab"""
        tab = ttk.Frame(self.notebook)
        
        # Search frame
        search_frame = ttk.Frame(tab)
        search_frame.pack(fill='x', pady=10)
        
        ttk.Label(search_frame, text="City:").pack(side='left', padx=5)
        self.city_entry = ttk.Entry(search_frame, width=25)
        self.city_entry.pack(side='left', padx=5)
        
        ttk.Button(search_frame, text="Get Weather", 
                  command=self.fetch_weather).pack(side='left', padx=5)
        
        # Current weather display
        self.current_weather_frame = ttk.LabelFrame(tab, text="Current Weather")
        self.current_weather_frame.pack(fill='x', padx=10, pady=5)
        
        # Forecast visualization
        self.forecast_viz_frame = ttk.LabelFrame(tab, text="Forecast Visualization")
        self.forecast_viz_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.setup_forecast_graph()
        
        return tab
    
    def setup_forecast_graph(self):
        """Setup the forecast visualization graph"""
        self.forecast_fig = Figure(figsize=(8, 5), dpi=100)
        self.forecast_ax = self.forecast_fig.add_subplot(111)
        self.forecast_canvas = FigureCanvasTkAgg(self.forecast_fig, 
                                               master=self.forecast_viz_frame)
        self.forecast_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_data_tab(self):
        """Create the data tab"""
        tab = ttk.Frame(self.notebook)
        
        # Controls frame
        controls_frame = ttk.Frame(tab)
        controls_frame.pack(fill='x', pady=10)
        
        ttk.Button(controls_frame, text="Fetch Historical Data", 
                  command=self.fetch_historical_data).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="View Data", 
                  command=self.display_data).pack(side='left', padx=5)
        
        # Data display
        self.data_display_frame = ttk.LabelFrame(tab, text="Weather Data")
        self.data_display_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Data visualization
        self.data_viz_frame = ttk.LabelFrame(tab, text="Data Visualization")
        self.data_viz_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.setup_data_graph()
        
        return tab
    
    def setup_data_graph(self):
        """Setup the data visualization graph"""
        self.data_fig = Figure(figsize=(8, 5), dpi=100)
        self.data_ax = self.data_fig.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, 
                                           master=self.data_viz_frame)
        self.data_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_model_tab(self):
        """Create the model tab"""
        tab = ttk.Frame(self.notebook)
        
        # Controls frame
        controls_frame = ttk.Frame(tab)
        controls_frame.pack(fill='x', pady=10)
        
        ttk.Button(controls_frame, text="Train Model", 
                  command=self.train_model).pack(side='left', padx=5)
        ttk.Button(controls_frame, text="Evaluate Model", 
                  command=self.show_model_metrics).pack(side='left', padx=5)
        
        # Model info
        self.model_info_frame = ttk.LabelFrame(tab, text="Model Information")
        self.model_info_frame.pack(fill='x', padx=10, pady=5)
        
        self.model_info_text = tk.Text(self.model_info_frame, height=8, wrap='word')
        self.model_info_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Feature importance
        self.feature_imp_frame = ttk.LabelFrame(tab, text="Feature Importance")
        self.feature_imp_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.setup_feature_importance_graph()
        
        return tab
    
    def setup_feature_importance_graph(self):
        """Setup the feature importance graph"""
        self.feature_fig = Figure(figsize=(8, 5), dpi=100)
        self.feature_ax = self.feature_fig.add_subplot(111)
        self.feature_canvas = FigureCanvasTkAgg(self.feature_fig, 
                                              master=self.feature_imp_frame)
        self.feature_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def fetch_weather(self):
        """Fetch current weather and forecast for a city"""
        city = self.city_entry.get()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
            
        try:
            self.status_var.set(f"Fetching weather for {city}...")
            self.root.update()
            
            # Get current weather
            current = self.weather_data.get_current_weather(city)
            if current is None:
                raise ValueError("Failed to fetch current weather")
                
            self.current_city = city
            self.display_current_weather(current)
            
            # Get forecast (using API)
            forecast = self.weather_data.get_forecast(city)
            if forecast is None:
                raise ValueError("Failed to fetch forecast")
                
            self.display_forecast(forecast)
            
            self.status_var.set(f"Weather data loaded for {city}")
            
        except Exception as e:
            self.status_var.set("Error fetching weather")
            messagebox.showerror("Error", f"Failed to get weather data: {str(e)}")
    
    def display_current_weather(self, data):
        """Display current weather information"""
        # Clear previous widgets
        for widget in self.current_weather_frame.winfo_children():
            widget.destroy()
            
        # Create display
        info_text = (f"City: {data['city']}\n"
                    f"Temperature: {data['temperature']:.1f}°C\n"
                    f"Humidity: {data['humidity']}%\n"
                    f"Pressure: {data['pressure']} hPa\n"
                    f"Wind: {data['wind_speed']} m/s at {data['wind_deg']}°\n"
                    f"Time: {data['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    
        ttk.Label(self.current_weather_frame, text=info_text, 
                 justify='left').pack(anchor='w', padx=10, pady=10)
    
    def display_forecast(self, forecast):
        """Display forecast data visually"""
        # Process forecast data
        forecast['date'] = forecast['timestamp'].dt.date
        daily_forecast = forecast.groupby('date').agg({
            'temperature': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean'
        }).reset_index()
        
        # Plot
        self.forecast_ax.clear()
        
        # Temperature line
        self.forecast_ax.plot(daily_forecast['date'], daily_forecast['temperature'], 
                            'r-o', label='Temperature (°C)')
        
        # Humidity line (on secondary axis)
        ax2 = self.forecast_ax.twinx()
        ax2.plot(daily_forecast['date'], daily_forecast['humidity'], 
                'b--s', label='Humidity (%)')
        
        # Formatting
        self.forecast_ax.set_xlabel('Date')
        self.forecast_ax.set_ylabel('Temperature (°C)', color='r')
        ax2.set_ylabel('Humidity (%)', color='b')
        
        self.forecast_ax.set_title(f'5-Day Forecast for {self.current_city}')
        self.forecast_fig.autofmt_xdate()
        
        # Combine legends
        lines, labels = self.forecast_ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.forecast_ax.legend(lines + lines2, labels + labels2, loc='upper left')
        
        self.forecast_canvas.draw()
    
    def fetch_historical_data(self):
        """Fetch historical weather data for training"""
        if not self.current_city:
            messagebox.showerror("Error", "Please get current weather first to set location")
            return
            
        try:
            self.status_var.set(f"Fetching historical data for {self.current_city}...")
            self.root.update()
            
            # For demo purposes, we'll use generated data but in a real app you would:
            # 1. Get lat/lon for the city
            # 2. Fetch historical data from OpenWeatherMap or another source
            # 3. Enrich with features
            
            # Generate sample data (replace with real API calls)
            data = self.weather_data.generate_sample_data()
            data['city'] = self.current_city
            
            # Enrich with additional features
            self.historical_data = self.weather_data.enrich_features(data)
            
            self.display_data()
            self.status_var.set(f"Historical data loaded ({len(self.historical_data)} records)")
            
        except Exception as e:
            self.status_var.set("Error fetching historical data")
            messagebox.showerror("Error", f"Failed to get historical data: {str(e)}")
    
    def display_data(self):
        """Display the loaded historical data"""
        if self.historical_data is None:
            messagebox.showerror("Error", "No historical data available")
            return
            
        try:
            # Clear previous widgets
            for widget in self.data_display_frame.winfo_children():
                widget.destroy()
                
            # Create a scrollable text widget
            text_widget = tk.Text(self.data_display_frame, wrap='none')
            scroll_y = ttk.Scrollbar(self.data_display_frame, orient='vertical', 
                                   command=text_widget.yview)
            scroll_x = ttk.Scrollbar(self.data_display_frame, orient='horizontal', 
                                   command=text_widget.xview)
            text_widget.configure(yscrollcommand=scroll_y.set, 
                                xscrollcommand=scroll_x.set)
            
            scroll_y.pack(side='right', fill='y')
            scroll_x.pack(side='bottom', fill='x')
            text_widget.pack(fill='both', expand=True)
            
            # Display data summary
            text_widget.insert('end', f"Weather Data for {self.current_city}\n")
            text_widget.insert('end', f"Records: {len(self.historical_data)}\n")
            text_widget.insert('end', f"Date Range: {self.historical_data['timestamp'].min().date()} to "
                                   f"{self.historical_data['timestamp'].max().date()}\n\n")
            
            text_widget.insert('end', "Sample Data:\n")
            text_widget.insert('end', self.historical_data.head(10).to_string())
            
            # Plot data
            self.plot_historical_data()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display data: {str(e)}")
    
    def plot_historical_data(self):
        """Plot the historical weather data"""
        self.data_ax.clear()
        
        # Plot temperature over time
        self.data_ax.plot(self.historical_data['timestamp'], 
                         self.historical_data['temperature'], 
                         'r-', label='Temperature')
        
        # Add rolling average
        rolling_avg = self.historical_data['temperature'].rolling(7).mean()
        self.data_ax.plot(self.historical_data['timestamp'], 
                         rolling_avg, 'b--', label='7-day Avg')
        
        # Formatting
        self.data_ax.set_xlabel('Date')
        self.data_ax.set_ylabel('Temperature (°C)')
        self.data_ax.set_title(f'Historical Temperature for {self.current_city}')
        self.data_ax.legend()
        self.data_fig.autofmt_xdate()
        self.data_canvas.draw()
    
    def train_model(self):
        """Train the weather prediction model"""
        if self.historical_data is None:
            messagebox.showerror("Error", "No historical data available")
            return
            
        try:
            self.status_var.set("Training model...")
            self.root.update()
            
            # Train model
            metrics = self.weather_model.train(self.historical_data)
            
            # Update UI
            self.show_model_metrics()
            self.plot_feature_importance()
            
            self.status_var.set(f"Model trained (MAE: {metrics['mae']:.2f}°C)")
            messagebox.showinfo("Success", 
                              f"Model trained successfully!\n"
                              f"MAE: {metrics['mae']:.2f}°C\n"
                              f"RMSE: {metrics['rmse']:.2f}°C\n"
                              f"R²: {metrics['r2']:.2f}")
            
        except Exception as e:
            self.status_var.set("Error training model")
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def show_model_metrics(self):
        """Display model metrics and information"""
        if not self.weather_model.model:
            messagebox.showerror("Error", "No trained model available")
            return
            
        try:
            self.model_info_text.delete(1.0, 'end')
            
            info_text = (f"Model Type: XGBoost Regressor\n"
                        f"Features Used: {len(self.weather_model.features)}\n"
                        f"Training Metrics:\n"
                        f"  - Mean Absolute Error: {self.weather_model.metrics.get('mae', 'N/A'):.2f}°C\n"
                        f"  - Root Mean Squared Error: {self.weather_model.metrics.get('rmse', 'N/A'):.2f}°C\n"
                        f"  - R² Score: {self.weather_model.metrics.get('r2', 'N/A'):.2f}\n\n"
                        f"Top Features:\n")
            
            # Get feature importance if available
            if hasattr(self.weather_model.model.named_steps['regressor'], 'feature_importances_'):
                importances = self.weather_model.model.named_steps['regressor'].feature_importances_
                features = self.weather_model.features
                
                # Combine and sort
                feature_importance = sorted(zip(features, importances), 
                                          key=lambda x: x[1], reverse=True)
                
                for feature, importance in feature_importance[:10]:
                    info_text += f"  - {feature}: {importance:.3f}\n"
            
            self.model_info_text.insert('end', info_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display model info: {str(e)}")
    
    def plot_feature_importance(self):
        """Plot feature importance from the trained model"""
        if not self.weather_model.model:
            return
            
        try:
            self.feature_ax.clear()
            
            # Get feature importance
            importances = self.weather_model.model.named_steps['regressor'].feature_importances_
            features = self.weather_model.features
            
            # Sort features by importance
            indices = np.argsort(importances)[-10:]  # Top 10 features
            sorted_features = [features[i] for i in indices]
            sorted_importances = [importances[i] for i in indices]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(sorted_features))
            self.feature_ax.barh(y_pos, sorted_importances, align='center')
            self.feature_ax.set_yticks(y_pos)
            self.feature_ax.set_yticklabels(sorted_features)
            self.feature_ax.set_xlabel('Importance Score')
            self.feature_ax.set_title('Top 10 Feature Importance')
            
            self.feature_canvas.draw()
            
        except Exception as e:
            print(f"Error plotting feature importance: {str(e)}")

def main():
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()