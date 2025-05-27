# Weather Forecasting Application
# This application uses historical weather data to predict future weather conditions

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
import datetime
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import json
import seaborn as sns
from tkcalendar import DateEntry
import threading
import time

# Set style for visualizations
plt.style.use('ggplot')
sns.set_palette("viridis")

# Create a custom color scheme
COLORS = {
    'primary': '#1e88e5',   # Blue
    'secondary': '#26a69a',  # Teal
    'accent': '#ff5722',    # Deep Orange
    'background': '#f5f5f5',  # Light Gray
    'text': '#212121',      # Dark Gray
    'warning': '#f9a825',   # Amber
    'success': '#43a047',   # Green
    'error': '#e53935',     # Red
}

# Weather condition icons mapping
WEATHER_ICONS = {
    'clear': '☀️',
    'clouds': '☁️',
    'rain': '🌧️',
    'thunderstorm': '⛈️',
    'snow': '❄️',
    'mist': '🌫️',
    'default': '🌤️'
}


class WeatherData:
    def __init__(self, api_key=None):
        self.api_key = api_key or "fc3cfaf056bdc92c4a8f43f9858a51a6"  # Replace with your OpenWeatherMap API key
        self.base_url = "http://api.openweathermap.org/data/2.5/weather?"
        self.forecast_url = "http://api.openweathermap.org/data/2.5/forecast?"
        self.history_url = "http://api.openweathermap.org/data/2.5/onecall/timemachine?"
        self.geo_url = "http://api.openweathermap.org/geo/1.0/direct?"
        self.cache_dir = "weather_cache"
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
    def get_current_weather(self, city):
        """Get current weather data for a city"""
        # First check if we have coordinates
        coords = self.get_city_coordinates(city)
        if not coords:
            complete_url = f"{self.base_url}q={city}&appid={self.api_key}&units=metric"
        else:
            lat, lon = coords
            complete_url = f"{self.base_url}lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
            
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"current_{city.lower().replace(' ', '_')}.json")
        
        # Only use cache if it's less than 1 hour old
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 3600:  # 1 hour in seconds
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # If not in cache or cache expired, make API request
        try:
            response = requests.get(complete_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                return data
            else:
                return None
        except requests.exceptions.RequestException:
            # If request fails and we have a cache, use it regardless of age
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            return None
        
    def get_forecast(self, city):
        """Get 5-day weather forecast for a city"""
        # First check if we have coordinates
        coords = self.get_city_coordinates(city)
        if not coords:
            complete_url = f"{self.forecast_url}q={city}&appid={self.api_key}&units=metric"
        else:
            lat, lon = coords
            complete_url = f"{self.forecast_url}lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"forecast_{city.lower().replace(' ', '_')}.json")
        
        # Only use cache if it's less than 3 hours old
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 10800:  # 3 hours in seconds
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # If not in cache or cache expired, make API request
        try:
            response = requests.get(complete_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                return data
            else:
                return None
        except requests.exceptions.RequestException:
            # If request fails and we have a cache, use it regardless of age
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            return None
    
    def get_city_coordinates(self, city):
        """Get latitude and longitude for a city"""
        cache_file = os.path.join(self.cache_dir, f"geo_{city.lower().replace(' ', '_')}.json")
        
        # Use cache if it exists (coordinates don't change)
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if data and len(data) > 0:
                    return data[0]['lat'], data[0]['lon']
        
        # If not in cache, make API request
        complete_url = f"{self.geo_url}q={city}&limit=1&appid={self.api_key}"
        try:
            response = requests.get(complete_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    # Save to cache
                    with open(cache_file, 'w') as f:
                        json.dump(data, f)
                    return data[0]['lat'], data[0]['lon']
            return None
        except:
            return None
    
    def get_historical_weather(self, city, days_ago=5):
        """Get historical weather data for a city"""
        coords = self.get_city_coordinates(city)
        if not coords:
            return None
            
        lat, lon = coords
        
        # Calculate timestamp for the requested days ago
        dt = int((datetime.datetime.now() - datetime.timedelta(days=days_ago)).timestamp())
        
        cache_file = os.path.join(self.cache_dir, f"history_{city.lower().replace(' ', '_')}_{days_ago}.json")
        
        # Historical data doesn't change, so we can use cache if it exists
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # If not in cache, make API request
        complete_url = f"{self.history_url}lat={lat}&lon={lon}&dt={dt}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(complete_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Save to cache
                with open(cache_file, 'w') as f:
                    json.dump(data, f)
                return data
            else:
                return None
        except:
            return None
    
    def fetch_real_historical_data(self, city, days=30):
        """Fetch real historical data for the last n days if possible"""
        try:
            data_list = []
            for i in range(days):
                data = self.get_historical_weather(city, i)
                if data and 'current' in data:
                    date = datetime.datetime.fromtimestamp(data['current']['dt'])
                    temp = data['current']['temp']
                    humidity = data['current']['humidity']
                    pressure = data['current']['pressure']
                    wind_speed = data['current']['wind_speed']
                    
                    data_list.append({
                        'date': date,
                        'temperature': temp,
                        'humidity': humidity,
                        'pressure': pressure,
                        'wind_speed': wind_speed
                    })
            
            if data_list:
                df = pd.DataFrame(data_list)
                # Add date features
                df['day_of_year'] = df['date'].dt.dayofyear
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                return df
            else:
                return None
        except Exception as e:
            print(f"Error fetching historical data: {str(e)}")
            return None
    
    def generate_sample_data(self, days=365, location_type="temperate"):
        """Generate sample weather data for training with different climate patterns"""
        dates = [datetime.datetime.now() - datetime.timedelta(days=x) for x in range(days, 0, -1)]
        
        # Different climate types
        climate_patterns = {
            "temperate": {
                "base_temp": 15, 
                "temp_amplitude": 10,
                "humidity_base": 70,
                "humidity_amplitude": 20,
                "pressure_base": 1013,
                "pressure_amplitude": 10,
                "wind_base": 5,
                "wind_amplitude": 3
            },
            "tropical": {
                "base_temp": 28, 
                "temp_amplitude": 5,
                "humidity_base": 80,
                "humidity_amplitude": 15,
                "pressure_base": 1010,
                "pressure_amplitude": 8,
                "wind_base": 7,
                "wind_amplitude": 5
            },
            "desert": {
                "base_temp": 32, 
                "temp_amplitude": 15,
                "humidity_base": 20,
                "humidity_amplitude": 15,
                "pressure_base": 1015,
                "pressure_amplitude": 7,
                "wind_base": 10,
                "wind_amplitude": 8
            },
            "arctic": {
                "base_temp": -5, 
                "temp_amplitude": 20,
                "humidity_base": 50,
                "humidity_amplitude": 10,
                "pressure_base": 1020,
                "pressure_amplitude": 15,
                "wind_base": 12,
                "wind_amplitude": 10
            }
        }
        
        pattern = climate_patterns.get(location_type, climate_patterns["temperate"])
        
        # Generate synthetic weather data with seasonal patterns
        temperature = []
        humidity = []
        pressure = []
        wind_speed = []
        precipitation = []
        cloud_cover = []
        
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            random_variation = np.random.normal(0, 1)  # Daily random factor
            
            # Base temperature with seasonal variation and some randomness
            temp = pattern["base_temp"] + pattern["temp_amplitude"] * season_factor + np.random.normal(0, 3) + random_variation * 2
            temperature.append(temp)
            
            # Humidity inversely related to temperature with some randomness
            hum = pattern["humidity_base"] - pattern["humidity_amplitude"] * season_factor + np.random.normal(0, 10) + random_variation * 5
            humidity.append(max(min(hum, 100), 0))  # Keep between 0-100%
            
            # Pressure with some patterns and randomness
            press = pattern["pressure_base"] + pattern["pressure_amplitude"] * np.sin(day_of_year / 20) + np.random.normal(0, 3) + random_variation * 2
            pressure.append(press)
            
            # Wind speed with some seasonal patterns
            wind = pattern["wind_base"] + pattern["wind_amplitude"] * abs(season_factor) + np.random.normal(0, 2) + random_variation
            wind_speed.append(max(wind, 0))  # Keep positive
            
            # Precipitation (0-100%)
            precip = max(min(40 + 30 * season_factor + np.random.normal(0, 20) + random_variation * 10, 100), 0)
            precipitation.append(precip)
            
            # Cloud cover (0-100%)
            cloud = max(min(50 + 20 * season_factor + np.random.normal(0, 15) + random_variation * 10, 100), 0)
            cloud_cover.append(cloud)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'precipitation': precipitation,
            'cloud_cover': cloud_cover
        })
        
        # Extract date features
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['season'] = df['month'].apply(self._get_season)
        
        return df
    
    def _get_season(self, month):
        """Helper method to get season from month"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
    
    def save_data(self, data, filename="weather_data.csv"):
        """Save weather data to file"""
        data.to_csv(filename, index=False)
        return filename
    
    def load_data(self, filename="weather_data.csv"):
        """Load weather data from file"""
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    
    def get_weather_icon(self, weather_condition):
        """Map OpenWeatherMap condition to icon"""
        condition = weather_condition.lower()
        for key in WEATHER_ICONS:
            if key in condition:
                return WEATHER_ICONS[key]
        return WEATHER_ICONS['default']


class WeatherModel:
    def __init__(self, model_path='weather_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_importance = None
        self.metrics = None
        self.model_type = "RandomForestRegressor"
        
    def load_model(self):
        """Load trained model if it exists"""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data.get('model')
            self.feature_importance = model_data.get('feature_importance')
            self.metrics = model_data.get('metrics')
            self.model_type = model_data.get('model_type', "RandomForestRegressor")
            return True
        return False
    
    def save_model(self):
        """Save trained model and metrics"""
        if self.model:
            model_data = {
                'model': self.model,
                'feature_importance': self.feature_importance,
                'metrics': self.metrics,
                'model_type': self.model_type
            }
            joblib.dump(model_data, self.model_path)
    def train(self, data, model_type="RandomForestRegressor", test_size=0.2):
        """Train weather forecasting model"""
        # Features and target
        features = ['day_of_year', 'month', 'day', 'humidity', 'pressure', 'wind_speed']

        # Add additional features if they exist in the data
        additional_features = ['precipitation', 'cloud_cover', 'day_of_week', 'season']
        for feature in additional_features:
            if feature in data.columns:
                features.append(feature)

        X = data[features]
        y = data['temperature']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Train model based on type
        self.model_type = model_type

        if model_type == "SVR":
            from sklearn.svm import SVR
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
            self.model.fit(X_train_scaled, y_train)
    
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
    
            # For SVR, we need to handle feature importance differently
            self.feature_importance = np.zeros(len(features))
        else:
            # Handle RandomForest and GradientBoosting
            if model_type == "RandomForestRegressor":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "GradientBoostingRegressor":
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:  # Default to RandomForestRegressor
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
            # Train model
            self.model.fit(X_train, y_train)
    
            # Evaluate
            y_pred = self.model.predict(X_test)

            # Get feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
            else:
                self.feature_importance = np.zeros(len(features))

        # Calculate metrics
            self.metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        self.save_model()
        return self.metrics['rmse']
    
    def predict_temperature(self, features_dict):
        """Predict temperature based on input features"""
        if not self.model:
            if not self.load_model():
                raise Exception("Model not trained. Please train the model first.")
        
        # Extract features in the correct order
        features_list = []
        for key, value in features_dict.items():
            features_list.append(value)
        
        features_array = np.array([features_list])
        return self.model.predict(features_array)[0]
    
    def predict_next_days(self, data, days=7):
        """Predict weather for next n days"""
        if not self.model:
            if not self.load_model():
                raise Exception("Model not trained. Please train the model first.")
        
        # Get the last day in the dataset
        last_date = data['date'].max()
        
        predictions = []
        last_temp = data['temperature'].iloc[-1]
        last_humidity = data['humidity'].iloc[-1]
        last_pressure = data['pressure'].iloc[-1]
        last_wind = data['wind_speed'].iloc[-1]
        
        for i in range(1, days+1):
            next_date = last_date + datetime.timedelta(days=i)
            day_of_year = next_date.timetuple().tm_yday
            month = next_date.month
            day = next_date.day
            day_of_week = next_date.weekday()
            season = self._get_season(month)
            
            # Use the average of last 5 days for the other features
            last_5_days = data.sort_values('date', ascending=False).head(5)
            avg_humidity = last_5_days['humidity'].mean()
            avg_pressure = last_5_days['pressure'].mean()
            avg_wind_speed = last_5_days['wind_speed'].mean()
            
            # Prepare feature dictionary
            features = {
                'day_of_year': day_of_year,
                'month': month,
                'day': day,
                'humidity': avg_humidity,
                'pressure': avg_pressure,
                'wind_speed': avg_wind_speed
            }
            
            # Add additional features if the model was trained with them
            if 'precipitation' in data.columns:
                avg_precip = last_5_days['precipitation'].mean() if 'precipitation' in last_5_days else 0
                features['precipitation'] = avg_precip
                
            if 'cloud_cover' in data.columns:
                avg_cloud = last_5_days['cloud_cover'].mean() if 'cloud_cover' in last_5_days else 0
                features['cloud_cover'] = avg_cloud
                
            if 'day_of_week' in data.columns:
                features['day_of_week'] = day_of_week
                
            if 'season' in data.columns:
                features['season'] = season
            
            # Predict temperature
            pred_temp = self.predict_temperature(features)
            
            # Add some realistic variations based on previous day
            humidity_change = np.random.normal(0, 5)
            new_humidity = max(min(avg_humidity + humidity_change, 100), 0)
            
            pressure_change = np.random.normal(0, 2)
            new_pressure = avg_pressure + pressure_change
            
            wind_change = np.random.normal(0, 1)
            new_wind = max(avg_wind_speed + wind_change, 0)
            
            # For precipitation and cloud cover if they exist
            if 'precipitation' in data.columns:
                precip_change = np.random.normal(0, 10)
                new_precip = max(min(avg_precip + precip_change, 100), 0)
            else:
                new_precip = None
                
            if 'cloud_cover' in data.columns:
                cloud_change = np.random.normal(0, 10)
                new_cloud = max(min(avg_cloud + cloud_change, 100), 0)
            else:
                new_cloud = None
            
            # Create prediction dict
            prediction = {
                'date': next_date,
                'temperature': pred_temp,
                'humidity': new_humidity,
                'pressure': new_pressure,
                'wind_speed': new_wind
            }
            
            # Add additional fields if they exist
            if new_precip is not None:
                prediction['precipitation'] = new_precip
            if new_cloud is not None:
                prediction['cloud_cover'] = new_cloud
            
            predictions.append(prediction)
        
        return pd.DataFrame(predictions)
    
    def _get_season(self, month):
        """Helper method to get season from month"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Fall
            
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.model:
            if not self.load_model():
                return "No model available. Please train the model first."
        
        info = f"Model: {self.model_type}\n"
        
        if hasattr(self.model, 'n_estimators'):
            info += f"Number of trees: {self.model.n_estimators}\n"
        
        if self.metrics:
            info += f"RMSE: {self.metrics['rmse']:.2f}\n"
            info += f"MAE: {self.metrics['mae']:.2f}\n"
            info += f"R² Score: {self.metrics['r2']:.4f}\n"
        
        info += "Features: "
        if self.feature_importance is not None:
            features = ['day_of_year', 'month', 'day', 'humidity', 'pressure', 'wind_speed', 
                       'precipitation', 'cloud_cover', 'day_of_week', 'season']
            # Only include features with importance > 0
            used_features = [f for i, f in enumerate(features) if i < len(self.feature_importance) and self.feature_importance[i] > 0]
            info += ", ".join(used_features)
        else:
            info += "Unknown"
        
        return info


class LoadingScreen(tk.Toplevel):
    """A loading screen that can be displayed during long operations"""
    def __init__(self, parent, message="Processing..."):
        super().__init__(parent)
        self.title("Loading")
        
        # Position in center of parent
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        width = 300
        height = 100
        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2
        
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        # Configure grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Loading message
        self.message_label = ttk.Label(self, text=message, font=('Helvetica', 12))
        self.message_label.grid(row=0, column=0, padx=20, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=250)
        self.progress.grid(row=1, column=0, padx=20, pady=10)
        self.progress.start(10)
    
    def update_message(self, message):
        """Update the loading message"""
        self.message_label.config(text=message)
        self.update()


class CustomTooltip:
    """Custom tooltip widget"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Create tooltip window
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack(padx=5, pady=5)
    
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # Configure the canvas to scroll the frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Pack everything
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class WeatherCardFrame(ttk.Frame):
    """A custom frame to display weather information in a card-like layout"""
    def __init__(self, parent, date, temperature, humidity, pressure, wind_speed, 
                 icon=None, precipitation=None, cloud_cover=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create a frame with a border
        self.configure(relief="raised", borderwidth=2, padding=10)
        
        # Format date
        if isinstance(date, datetime.datetime):
            date_str = date.strftime("%a, %b %d")
        else:
            date_str = str(date)
        
        # Configure grid
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        
        # Date header
        date_label = ttk.Label(self, text=date_str, font=('Helvetica', 12, 'bold'))
        date_label.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        # Weather icon if provided
        if icon:
            icon_label = ttk.Label(self, text=icon, font=('Helvetica', 24))
            icon_label.grid(row=1, column=0, rowspan=2, sticky="e", padx=(0, 10))
        
        # Temperature (larger font)
        temp_label = ttk.Label(self, text=f"{temperature:.1f}°C", font=('Helvetica', 18))
        temp_label.grid(row=1, column=1, sticky="w")
        
        # Other weather details
        details_frame = ttk.Frame(self)
        details_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        # Configure details grid
        details_frame.columnconfigure(0, weight=1)
        details_frame.columnconfigure(1, weight=1)
        
        # Humidity
        ttk.Label(details_frame, text="Humidity:").grid(row=0, column=0, sticky="w")
        ttk.Label(details_frame, text=f"{humidity:.1f}%").grid(row=0, column=1, sticky="e")
        
        # Pressure
        ttk.Label(details_frame, text="Pressure:").grid(row=1, column=0, sticky="w")
        ttk.Label(details_frame, text=f"{pressure:.1f} hPa").grid(row=1, column=1, sticky="e")
        
        # Wind Speed
        ttk.Label(details_frame, text="Wind:").grid(row=2, column=0, sticky="w")
        ttk.Label(details_frame, text=f"{wind_speed:.1f} m/s").grid(row=2, column=1, sticky="e")
        
        # Optional fields
        row = 3
        if precipitation is not None:
            ttk.Label(details_frame, text="Precipitation:").grid(row=row, column=0, sticky="w")
            ttk.Label(details_frame, text=f"{precipitation:.1f}%").grid(row=row, column=1, sticky="e")
            row += 1
            
        if cloud_cover is not None:
            ttk.Label(details_frame, text="Cloud Cover:").grid(row=row, column=0, sticky="w")
            ttk.Label(details_frame, text=f"{cloud_cover:.1f}%").grid(row=row, column=1, sticky="e")


class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Forecasting Application")
        self.root.geometry("900x700")
        self.root.configure(bg=COLORS['background'])
        
        self.weather_data = WeatherData()
        self.weather_model = WeatherModel()
        
        # Apply custom style
        self.setup_styles()
        
        # Create main application structure
        self.create_widgets()
        
        # Data variables
        self.data = None
        self.forecast_data = None
        self.current_city = None
        
        # Load or generate sample data
        try:
            if not self.weather_model.load_model():
                self.generate_data()
                self.train_model()
            else:
                self.generate_data()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
    
    def setup_styles(self):
        """Set up custom styles for the application"""
        style = ttk.Style()
        
        # Configure the general style
        style.configure("TFrame", background=COLORS['background'])
        style.configure("TLabel", background=COLORS['background'], foreground=COLORS['text'])
        style.configure("TButton", background=COLORS['primary'], foreground="white")
        
        # Custom styles
        style.configure("Title.TLabel", font=('Helvetica', 16, 'bold'), foreground=COLORS['primary'])
        style.configure("Subtitle.TLabel", font=('Helvetica', 12), foreground=COLORS['secondary'])
        style.configure("Header.TLabel", font=('Helvetica', 14, 'bold'), foreground=COLORS['primary'])
        
        # Card style
        style.configure("Card.TFrame", background="white", relief="raised", borderwidth=2)
        style.configure("Card.TLabel", background="white")
        
        # Button styles
        style.configure("Primary.TButton", background=COLORS['primary'], foreground="white")
        style.configure("Success.TButton", background=COLORS['success'], foreground="white")
        style.configure("Warning.TButton", background=COLORS['warning'], foreground="white")
        
    def create_widgets(self):
        """Create UI elements"""
        # Create a main frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill='both', expand=True)
        
        # Create header with app title and info
        self.create_header()
        
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.forecast_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.forecast_tab, text="Weather Forecast")
        self.notebook.add(self.data_tab, text="Weather Data")
        self.notebook.add(self.model_tab, text="Model Training")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Setup tabs
        self.setup_forecast_tab()
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_settings_tab()
        
        # Status bar at the bottom
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_header(self):
        """Create header section with title and info"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill='x', pady=(0, 10))
        
        title_label = ttk.Label(header_frame, text="Weather Forecasting Application", style="Title.TLabel")
        title_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = ttk.Label(header_frame, text="v2.0", style="Subtitle.TLabel")
        version_label.pack(side=tk.RIGHT)
    
    def setup_forecast_tab(self):
        """Setup the forecast tab"""
        # City search frame
        search_frame = ttk.Frame(self.forecast_tab, padding=10)
        search_frame.pack(fill='x')
        
        ttk.Label(search_frame, text="City:").pack(side=tk.LEFT, padx=5)
        self.city_var = tk.StringVar(value="")
        ttk.Entry(search_frame, textvariable=self.city_var, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(search_frame, text="Get Forecast", command=self.get_forecast, style="Primary.TButton").pack(side=tk.LEFT, padx=5)
        
        # Add some popular city suggestions
        popular_cities = ["New York", "London", "Tokyo", "Sydney", "Paris"]
        ttk.Label(search_frame, text="Popular:").pack(side=tk.LEFT, padx=(20, 5))
        
        for city in popular_cities:
            btn = ttk.Button(search_frame, text=city, command=lambda c=city: self.set_city(c))
            btn.pack(side=tk.LEFT, padx=2)
            CustomTooltip(btn, f"Get forecast for {city}")
        
        # Current weather frame
        self.current_weather_frame = ttk.LabelFrame(self.forecast_tab, text="Current Weather", padding=10)
        self.current_weather_frame.pack(fill='x', padx=10, pady=5)
        
        # Initial "no data" message
        self.no_data_label = ttk.Label(self.current_weather_frame, text="Enter a city name and click 'Get Forecast'")
        self.no_data_label.pack(pady=20)
        
        # Frame for forecast cards
        self.forecast_cards_frame = ttk.Frame(self.forecast_tab, padding=10)
        self.forecast_cards_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Forecasted weather visualization frame
        self.forecast_viz_frame = ttk.LabelFrame(self.forecast_tab, text="Forecast Visualization", padding=10)
        self.forecast_viz_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Setup forecast graph
        self.forecast_fig = Figure(figsize=(8, 4), dpi=100)
        self.forecast_canvas = FigureCanvasTkAgg(self.forecast_fig, master=self.forecast_viz_frame)
        self.forecast_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_data_tab(self):
        """Setup the data tab"""
        # Control frame
        control_frame = ttk.Frame(self.data_tab, padding=10)
        control_frame.pack(fill='x')
        
        # Data source options
        data_source_frame = ttk.LabelFrame(control_frame, text="Data Source", padding=5)
        data_source_frame.pack(side=tk.LEFT, padx=5, fill='x', expand=True)
        
        self.data_source_var = tk.StringVar(value="sample")
        ttk.Radiobutton(data_source_frame, text="Sample Data", variable=self.data_source_var, value="sample").pack(anchor='w')
        ttk.Radiobutton(data_source_frame, text="Real Data (API)", variable=self.data_source_var, value="real").pack(anchor='w')
        
        # Climate type for sample data
        climate_frame = ttk.LabelFrame(control_frame, text="Climate Type", padding=5)
        climate_frame.pack(side=tk.LEFT, padx=5, fill='x', expand=True)
        
        self.climate_var = tk.StringVar(value="temperate")
        ttk.Radiobutton(climate_frame, text="Temperate", variable=self.climate_var, value="temperate").pack(anchor='w')
        ttk.Radiobutton(climate_frame, text="Tropical", variable=self.climate_var, value="tropical").pack(anchor='w')
        ttk.Radiobutton(climate_frame, text="Desert", variable=self.climate_var, value="desert").pack(anchor='w')
        ttk.Radiobutton(climate_frame, text="Arctic", variable=self.climate_var, value="arctic").pack(anchor='w')
        
        # Data control buttons
        buttons_frame = ttk.Frame(control_frame, padding=5)
        buttons_frame.pack(side=tk.LEFT, padx=5, fill='y')
        
        ttk.Button(buttons_frame, text="Generate Data", command=self.generate_data, style="Primary.TButton").pack(fill='x', pady=2)
        ttk.Button(buttons_frame, text="Load Data", command=self.load_data_dialog).pack(fill='x', pady=2)
        ttk.Button(buttons_frame, text="Save Data", command=self.save_data_dialog).pack(fill='x', pady=2)
        
        # Data statistics frame
        self.data_stats_frame = ttk.LabelFrame(self.data_tab, text="Data Statistics", padding=10)
        self.data_stats_frame.pack(fill='x', padx=10, pady=5)
        
        # Initial "no data" message
        self.stats_text = tk.Text(self.data_stats_frame, height=5, width=60)
        self.stats_text.pack(fill='both', expand=True)
        self.stats_text.insert(tk.END, "No data available. Generate or load data first.")
        self.stats_text.config(state='disabled')
        
        # Data visualization notebook (for multiple plots)
        self.data_viz_notebook = ttk.Notebook(self.data_tab)
        self.data_viz_notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Temperature tab
        self.temp_viz_frame = ttk.Frame(self.data_viz_notebook)
        self.data_viz_notebook.add(self.temp_viz_frame, text="Temperature")
        
        # Create figure and canvas for temperature
        self.temp_fig = Figure(figsize=(8, 4), dpi=100)
        self.temp_canvas = FigureCanvasTkAgg(self.temp_fig, master=self.temp_viz_frame)
        self.temp_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Humidity tab
        self.humidity_viz_frame = ttk.Frame(self.data_viz_notebook)
        self.data_viz_notebook.add(self.humidity_viz_frame, text="Humidity")
        
        # Create figure and canvas for humidity
        self.humidity_fig = Figure(figsize=(8, 4), dpi=100)
        self.humidity_canvas = FigureCanvasTkAgg(self.humidity_fig, master=self.humidity_viz_frame)
        self.humidity_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Other metrics tab
        self.other_viz_frame = ttk.Frame(self.data_viz_notebook)
        self.data_viz_notebook.add(self.other_viz_frame, text="Other Metrics")
        
        # Create figure and canvas for other metrics
        self.other_fig = Figure(figsize=(8, 4), dpi=100)
        self.other_canvas = FigureCanvasTkAgg(self.other_fig, master=self.other_viz_frame)
        self.other_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Correlation tab
        self.corr_viz_frame = ttk.Frame(self.data_viz_notebook)
        self.data_viz_notebook.add(self.corr_viz_frame, text="Correlations")
        
        # Create figure and canvas for correlations
        self.corr_fig = Figure(figsize=(8, 4), dpi=100)
        self.corr_canvas = FigureCanvasTkAgg(self.corr_fig, master=self.corr_viz_frame)
        self.corr_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_model_tab(self):
        """Setup the model tab"""
        # Split into left and right panels
        panel_frame = ttk.Frame(self.model_tab)
        panel_frame.pack(fill='both', expand=True)
        
        # Left panel - model configuration
        left_panel = ttk.Frame(panel_frame, padding=10)
        left_panel.pack(side=tk.LEFT, fill='both', expand=True)
        
        # Model configuration
        config_frame = ttk.LabelFrame(left_panel, text="Model Configuration", padding=10)
        config_frame.pack(fill='x', pady=5)
        
        # Model type
        ttk.Label(config_frame, text="Model Type:").grid(row=0, column=0, sticky='w', pady=2)
        self.model_type_var = tk.StringVar(value="RandomForestRegressor")
        model_type_combo = ttk.Combobox(config_frame, textvariable=self.model_type_var, state="readonly")
        model_type_combo.grid(row=0, column=1, sticky='ew', pady=2)
        model_type_combo['values'] = ["RandomForestRegressor", "GradientBoostingRegressor", "SVR"]
        
        # Test size
        ttk.Label(config_frame, text="Test Split Size:").grid(row=1, column=0, sticky='w', pady=2)
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_scale = ttk.Scale(config_frame, from_=0.1, to=0.5, variable=self.test_size_var, orient="horizontal")
        test_size_scale.grid(row=1, column=1, sticky='ew', pady=2)
        ttk.Label(config_frame, textvariable=tk.StringVar(value=lambda: f"{self.test_size_var.get():.1f}")).grid(row=1, column=2, padx=5)
        
        # Training buttons
        train_button = ttk.Button(config_frame, text="Train Model", command=self.train_model, style="Primary.TButton")
        train_button.grid(row=2, column=0, sticky='ew', pady=5, padx=5)
        
        evaluate_button = ttk.Button(config_frame, text="Evaluate Model", command=self.evaluate_model)
        evaluate_button.grid(row=2, column=1, sticky='ew', pady=5, padx=5)
        
        # Model info frame
        self.model_info_frame = ttk.LabelFrame(left_panel, text="Model Information", padding=10)
        self.model_info_frame.pack(fill='both', expand=True, pady=5)
        
        self.model_info_text = tk.Text(self.model_info_frame, height=10, width=50)
        self.model_info_text.pack(fill='both', expand=True, padx=5, pady=5)
        if self.weather_model.model:
            self.model_info_text.insert(tk.END, self.weather_model.get_model_info())
        else:
            self.model_info_text.insert(tk.END, "No model available. Please train the model first.")
        
        # Right panel - visualizations
        right_panel = ttk.Frame(panel_frame, padding=10)
        right_panel.pack(side=tk.RIGHT, fill='both', expand=True)
        
        # Feature importance frame
        self.feature_imp_frame = ttk.LabelFrame(right_panel, text="Feature Importance", padding=10)
        self.feature_imp_frame.pack(fill='both', expand=True, pady=5)
        
        self.model_fig = Figure(figsize=(6, 4), dpi=100)
        self.model_canvas = FigureCanvasTkAgg(self.model_fig, master=self.feature_imp_frame)
        self.model_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Prediction error frame
        self.pred_error_frame = ttk.LabelFrame(right_panel, text="Prediction Error Analysis", padding=10)
        self.pred_error_frame.pack(fill='both', expand=True, pady=5)
        
        self.error_fig = Figure(figsize=(6, 4), dpi=100)
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, master=self.pred_error_frame)
        self.error_canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def setup_settings_tab(self):
        """Setup the settings tab"""
        settings_frame = ttk.Frame(self.settings_tab, padding=10)
        settings_frame.pack(fill='both', expand=True)
        
        # API Settings
        api_frame = ttk.LabelFrame(settings_frame, text="API Settings", padding=10)
        api_frame.pack(fill='x', pady=5)
        
        ttk.Label(api_frame, text="OpenWeatherMap API Key:").grid(row=0, column=0, sticky='w', pady=5)
        self.api_key_var = tk.StringVar(value=self.weather_data.api_key)
        api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=40)
        api_key_entry.grid(row=0, column=1, sticky='ew', pady=5, padx=5)
        
        ttk.Button(api_frame, text="Save API Key", command=self.save_api_key).grid(row=0, column=2, padx=5)
        
        # Display Settings
        display_frame = ttk.LabelFrame(settings_frame, text="Display Settings", padding=10)
        display_frame.pack(fill='x', pady=5)
        
        # Temperature unit
        ttk.Label(display_frame, text="Temperature Unit:").grid(row=0, column=0, sticky='w', pady=5)
        self.temp_unit_var = tk.StringVar(value="Celsius")
        temp_unit_combo = ttk.Combobox(display_frame, textvariable=self.temp_unit_var, state="readonly")
        temp_unit_combo.grid(row=0, column=1, sticky='ew', pady=5, padx=5)
        temp_unit_combo['values'] = ["Celsius", "Fahrenheit"]
        
        # Chart style
        ttk.Label(display_frame, text="Chart Style:").grid(row=1, column=0, sticky='w', pady=5)
        self.chart_style_var = tk.StringVar(value="ggplot")
        chart_style_combo = ttk.Combobox(display_frame, textvariable=self.chart_style_var, state="readonly")
        chart_style_combo.grid(row=1, column=1, sticky='ew', pady=5, padx=5)
        chart_style_combo['values'] = ["ggplot", "seaborn", "bmh", "dark_background", "classic"]
        
        ttk.Button(display_frame, text="Apply Settings", command=self.apply_display_settings).grid(row=1, column=2, padx=5)
        
        # Data Settings
        data_frame = ttk.LabelFrame(settings_frame, text="Data Settings", padding=10)
        data_frame.pack(fill='x', pady=5)
        
        ttk.Label(data_frame, text="Default Sample Days:").grid(row=0, column=0, sticky='w', pady=5)
        self.sample_days_var = tk.IntVar(value=365)
        ttk.Spinbox(data_frame, from_=30, to=1000, increment=30, textvariable=self.sample_days_var, width=10).grid(row=0, column=1, sticky='w', pady=5, padx=5)
        
        ttk.Label(data_frame, text="Forecast Days:").grid(row=1, column=0, sticky='w', pady=5)
        self.forecast_days_var = tk.IntVar(value=7)
        ttk.Spinbox(data_frame, from_=1, to=30, increment=1, textvariable=self.forecast_days_var, width=10).grid(row=1, column=1, sticky='w', pady=5, padx=5)
        
        # About section
        about_frame = ttk.LabelFrame(settings_frame, text="About", padding=10)
        about_frame.pack(fill='x', pady=5)
        
        about_text = "Weather Forecasting Application v2.0\n"
        about_text += "This application uses machine learning to predict weather conditions based on historical data.\n"
        about_text += "Created as a demonstration project.\n"
        
        about_label = ttk.Label(about_frame, text=about_text, wraplength=500, justify='left')
        about_label.pack(fill='x', pady=5)
    
    def set_city(self, city):
        """Set city in the entry and get forecast"""
        self.city_var.set(city)
        self.get_forecast()
    
    def save_api_key(self):
        """Save the API key"""
        self.weather_data.api_key = self.api_key_var.get()
        messagebox.showinfo("Success", "API key saved successfully!")
    
    def apply_display_settings(self):
        """Apply the display settings"""
        # Set chart style
        plt.style.use(self.chart_style_var.get())
        
        # Update existing charts
        if self.data is not None:
            self.update_data_visualizations()
        if self.forecast_data is not None:
            self.update_forecast_visualizations()
        
        messagebox.showinfo("Success", "Display settings applied successfully!")
    
    def save_data_dialog(self):
        """Open dialog to save data"""
        if self.data is None or self.data.empty:
            messagebox.showerror("Error", "No data available to save.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.weather_data.save_data(self.data, filepath)
                messagebox.showinfo("Success", f"Data saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")
    
    def load_data_dialog(self):
        """Open dialog to load data"""
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.data = self.weather_data.load_data(filepath)
                if self.data is not None:
                    messagebox.showinfo("Success", f"Data loaded from {filepath}")
                    self.update_data_stats()
                    self.update_data_visualizations()
                else:
                    messagebox.showerror("Error", "Failed to load data.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def get_forecast(self):
        """Get weather forecast for entered city"""
        city = self.city_var.get()
        if not city:
            messagebox.showerror("Error", "Please enter a city name.")
            return
        
        self.current_city = city
        self.status_var.set(f"Fetching forecast for {city}...")
        self.root.update()
        
        # Start loading screen
        loading = LoadingScreen(self.root, f"Fetching forecast for {city}...")
        
        # Use threading to prevent UI freezing
        def fetch_forecast():
            try:
                # Try to get real forecast data if available
                current_weather = self.weather_data.get_current_weather(city)
                forecast_weather = self.weather_data.get_forecast(city)
                
                # Generate predictions using our model
                if self.data is None:
                    self.generate_data()
                
                days = self.forecast_days_var.get()
                predictions = self.weather_model.predict_next_days(self.data, days=days)
                self.forecast_data = predictions
                
                # Update UI with results
                self.root.after(0, lambda: self.update_forecast_ui(city, current_weather, forecast_weather, predictions))
                self.root.after(0, lambda: loading.destroy())
                self.root.after(0, lambda: self.status_var.set(f"Forecast loaded for {city}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to get forecast: {str(e)}"))
                self.root.after(0, lambda: loading.destroy())
                self.root.after(0, lambda: self.status_var.set("Error fetching forecast"))
        
        thread = threading.Thread(target=fetch_forecast)
        thread.daemon = True
        thread.start()
    
    def update_forecast_ui(self, city, current_weather, forecast_weather, predictions):
        """Update UI with forecast information"""
        # Clear existing widgets
        for widget in self.current_weather_frame.winfo_children():
            widget.destroy()
        
        for widget in self.forecast_cards_frame.winfo_children():
            widget.destroy()
        
        # Current weather section
        if current_weather and current_weather.get('cod') == 200:
            # Use real API data
            temp = current_weather['main']['temp']
            humidity = current_weather['main']['humidity']
            pressure = current_weather['main']['pressure']
            wind_speed = current_weather['wind']['speed']
            weather_condition = current_weather['weather'][0]['main']
            weather_icon = self.weather_data.get_weather_icon(weather_condition)
            
            city_name = current_weather['name']
            country = current_weather['sys']['country']
            
            # Display current weather
            header_label = ttk.Label(self.current_weather_frame, text=f"Current Weather in {city_name}, {country}", font=('Helvetica', 14, 'bold'))
            header_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)
            
            icon_label = ttk.Label(self.current_weather_frame, text=weather_icon, font=('Helvetica', 36))
            icon_label.grid(row=1, column=0, rowspan=2, padx=20, pady=10)
            
            temp_label = ttk.Label(self.current_weather_frame, text=f"{temp:.1f}°C", font=('Helvetica', 24))
            temp_label.grid(row=1, column=1, sticky="w")
            
            cond_label = ttk.Label(self.current_weather_frame, text=weather_condition, font=('Helvetica', 16))
            cond_label.grid(row=2, column=1, sticky="w")
            
            # Details frame
            details_frame = ttk.Frame(self.current_weather_frame)
            details_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
            
            # Configure details grid
            details_frame.columnconfigure(0, weight=1)
            details_frame.columnconfigure(1, weight=1)
            details_frame.columnconfigure(2, weight=1)
            
            # Continuing from previous code...

            # Humidity
            ttk.Label(details_frame, text="Humidity:", font=('Helvetica', 10)).grid(row=0, column=0, sticky="w")
            ttk.Label(details_frame, text=f"{humidity}%", font=('Helvetica', 10, 'bold')).grid(row=0, column=1, sticky="w")
            
            # Pressure
            ttk.Label(details_frame, text="Pressure:", font=('Helvetica', 10)).grid(row=0, column=2, sticky="w")
            ttk.Label(details_frame, text=f"{pressure} hPa", font=('Helvetica', 10, 'bold')).grid(row=0, column=3, sticky="w")
            
            # Wind
            ttk.Label(details_frame, text="Wind Speed:", font=('Helvetica', 10)).grid(row=1, column=0, sticky="w")
            ttk.Label(details_frame, text=f"{wind_speed} m/s", font=('Helvetica', 10, 'bold')).grid(row=1, column=1, sticky="w")
            
            # Last updated time
            last_updated = datetime.datetime.fromtimestamp(current_weather['dt'])
            ttk.Label(details_frame, text=f"Last Updated: {last_updated.strftime('%H:%M:%S')}",
                     font=('Helvetica', 9)).grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))
        else:
            # No API data, show info about model predictions only
            ttk.Label(self.current_weather_frame, text=f"Weather forecast for {city}", 
                     font=('Helvetica', 14, 'bold')).pack(pady=10)
            ttk.Label(self.current_weather_frame, text="Using machine learning model predictions", 
                     font=('Helvetica', 10, 'italic')).pack()
        
        # Display forecast cards in a scrollable frame
        scrollable = ScrollableFrame(self.forecast_cards_frame)
        scrollable.pack(fill="both", expand=True)
        
        # Create a frame for the cards with horizontal layout
        cards_container = ttk.Frame(scrollable.scrollable_frame)
        cards_container.pack(fill="x", expand=True)
        
        # Process model predictions
        for i, row in predictions.iterrows():
            date = row['date']
            temperature = row['temperature']
            humidity = row['humidity']
            pressure = row['pressure']
            wind_speed = row['wind_speed']
            
            # Optional fields
            precipitation = row.get('precipitation')
            cloud_cover = row.get('cloud_cover')
            
            # Estimate weather icon based on temperature and precipitation if available
            icon = WEATHER_ICONS['default']
            if precipitation is not None and precipitation > 70:
                icon = WEATHER_ICONS['rain']
            elif precipitation is not None and precipitation > 40:
                icon = WEATHER_ICONS['clouds']
            elif cloud_cover is not None and cloud_cover > 70:
                icon = WEATHER_ICONS['clouds']
            elif temperature > 30:
                icon = WEATHER_ICONS['clear']
            elif temperature < 0:
                icon = WEATHER_ICONS['snow']
            
            # Create weather card
            card = WeatherCardFrame(cards_container, date, temperature, humidity, pressure, 
                                    wind_speed, icon, precipitation, cloud_cover)
            card.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Update forecast visualization
        self.update_forecast_visualizations()
    
    def update_forecast_visualizations(self):
        """Update forecast visualization charts"""
        if self.forecast_data is None or self.forecast_data.empty:
            return
        
        # Clear previous plots
        self.forecast_fig.clear()
        
        # Create subplots
        ax1 = self.forecast_fig.add_subplot(211)  # For temperature
        ax2 = self.forecast_fig.add_subplot(212)  # For humidity/pressure
        
        # Format dates for x-axis
        dates = [d.strftime('%m-%d') for d in self.forecast_data['date']]
        
        # Plot temperature
        temp_line = ax1.plot(dates, self.forecast_data['temperature'], 'o-', color=COLORS['primary'], label='Temperature')
        ax1.set_ylabel('Temperature (°C)', color=COLORS['primary'])
        ax1.set_title('Forecasted Temperature')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot humidity
        hum_line = ax2.plot(dates, self.forecast_data['humidity'], 'o-', color=COLORS['secondary'], label='Humidity')
        ax2.set_ylabel('Humidity (%)', color=COLORS['secondary'])
        ax2.set_xlabel('Date')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot pressure on secondary y-axis
        ax3 = ax2.twinx()
        pres_line = ax3.plot(dates, self.forecast_data['pressure'], 'o-', color=COLORS['accent'], label='Pressure')
        ax3.set_ylabel('Pressure (hPa)', color=COLORS['accent'])
        
        # Create legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1, loc='upper right')
        
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax2.legend(lines2 + lines3, labels2 + labels3, loc='upper right')
        
        # Adjust layout and redraw
        self.forecast_fig.tight_layout()
        self.forecast_canvas.draw()
    
    def generate_data(self):
        """Generate sample weather data"""
        self.status_var.set("Generating weather data...")
        self.root.update()
        
        # Start loading screen
        loading = LoadingScreen(self.root, "Generating weather data...")
        
        # Use threading to prevent UI freezing
        def generate():
            try:
                source = self.data_source_var.get()
                if source == "real" and self.current_city:
                    # Try to get real historical data
                    data = self.weather_data.fetch_real_historical_data(self.current_city)
                    if data is None or data.empty:
                        # Fall back to sample data
                        days = self.sample_days_var.get()
                        climate = self.climate_var.get()
                        data = self.weather_data.generate_sample_data(days, climate)
                else:
                    # Generate sample data
                    days = self.sample_days_var.get()
                    climate = self.climate_var.get()
                    data = self.weather_data.generate_sample_data(days, climate)
                
                self.data = data
                
                # Update UI
                self.root.after(0, self.update_data_stats)
                self.root.after(0, self.update_data_visualizations)
                self.root.after(0, lambda: loading.destroy())
                self.root.after(0, lambda: self.status_var.set("Data generated successfully"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to generate data: {str(e)}"))
                self.root.after(0, lambda: loading.destroy())
                self.root.after(0, lambda: self.status_var.set("Error generating data"))
        
        thread = threading.Thread(target=generate)
        thread.daemon = True
        thread.start()
    
    def update_data_stats(self):
        """Update data statistics display"""
        if self.data is None or self.data.empty:
            return
        
        # Enable text widget for editing
        self.stats_text.config(state='normal')
        self.stats_text.delete(1.0, tk.END)
        
        # Calculate basic statistics
        stats = self.data.describe()
        
        # Format for display
        text = f"Dataset Summary:\n"
        text += f"{'=' * 40}\n"
        text += f"Total records: {len(self.data)}\n"
        text += f"Date range: {self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}\n"
        text += f"{'=' * 40}\n"
        text += f"Temperature (°C):\n"
        text += f"  Mean: {stats['temperature']['mean']:.2f}\n"
        text += f"  Min: {stats['temperature']['min']:.2f}\n"
        text += f"  Max: {stats['temperature']['max']:.2f}\n"
        text += f"  Std Dev: {stats['temperature']['std']:.2f}\n"
        text += f"{'=' * 40}\n"
        text += f"Humidity (%):\n"
        text += f"  Mean: {stats['humidity']['mean']:.2f}\n"
        text += f"  Min: {stats['humidity']['min']:.2f}\n"
        text += f"  Max: {stats['humidity']['max']:.2f}\n"
        
        # Add correlations
        text += f"{'=' * 40}\n"
        text += "Correlations with Temperature:\n"
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlations = {}
        for col in numeric_cols:
            if col != 'temperature' and col not in ['date', 'day_of_year', 'month', 'day']:
                correlations[col] = self.data['temperature'].corr(self.data[col])
        
        # Sort correlations by absolute value
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for col, corr in sorted_corrs:
            text += f"  {col}: {corr:.4f}\n"
        
        self.stats_text.insert(tk.END, text)
        self.stats_text.config(state='disabled')
    
    def update_data_visualizations(self):
        """Update data visualization charts"""
        if self.data is None or self.data.empty:
            return
        
        # Temperature plot
        self.temp_fig.clear()
        ax = self.temp_fig.add_subplot(111)
        
        # Plot temperature time series
        x = self.data['date']
        y = self.data['temperature']
        ax.plot(x, y, color=COLORS['primary'], alpha=0.7)
        ax.set_title('Temperature Time Series')
        ax.set_xlabel('Date')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis to show fewer date labels
        locator = plt.MaxNLocator(10)
        ax.xaxis.set_major_locator(locator)
        
        # Rotate date labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        self.temp_fig.tight_layout()
        self.temp_canvas.draw()
        
        # Humidity plot
        self.humidity_fig.clear()
        ax = self.humidity_fig.add_subplot(111)
        
        # Plot humidity time series
        x = self.data['date']
        y = self.data['humidity']
        ax.plot(x, y, color=COLORS['secondary'], alpha=0.7)
        ax.set_title('Humidity Time Series')
        ax.set_xlabel('Date')
        ax.set_ylabel('Humidity (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Format x-axis to show fewer date labels
        locator = plt.MaxNLocator(10)
        ax.xaxis.set_major_locator(locator)
        
        # Rotate date labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        self.humidity_fig.tight_layout()
        self.humidity_canvas.draw()
        
        # Other metrics plot
        self.other_fig.clear()
        ax = self.other_fig.add_subplot(111)
        
        # Plot pressure and wind speed
        x = self.data['date']
        y1 = self.data['pressure']
        ax.plot(x, y1, color=COLORS['accent'], alpha=0.7, label='Pressure')
        ax.set_title('Pressure and Wind Speed')
        ax.set_xlabel('Date')
        ax.set_ylabel('Pressure (hPa)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Plot wind speed on secondary y-axis
        ax2 = ax.twinx()
        y2 = self.data['wind_speed']
        ax2.plot(x, y2, color=COLORS['warning'], alpha=0.7, label='Wind Speed')
        ax2.set_ylabel('Wind Speed (m/s)')
        
        # Format x-axis to show fewer date labels
        locator = plt.MaxNLocator(10)
        ax.xaxis.set_major_locator(locator)
        
        # Rotate date labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        self.other_fig.tight_layout()
        self.other_canvas.draw()
        
        # Correlation matrix plot
        self.corr_fig.clear()
        ax = self.corr_fig.add_subplot(111)
        
        # Select numeric columns for correlation
        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed']
        # Add optional columns if they exist
        if 'precipitation' in self.data.columns:
            numeric_cols.append('precipitation')
        if 'cloud_cover' in self.data.columns:
            numeric_cols.append('cloud_cover')
        
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        
        # Create heatmap
        im = ax.imshow(corr_matrix, cmap='coolwarm')
        
        # Add colorbar
        cbar = self.corr_fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
        
        # Add labels
        ax.set_title('Correlation Matrix')
        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_yticklabels(numeric_cols)
        
        # Add correlation values to cells
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                       ha="center", va="center", color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
        
        self.corr_fig.tight_layout()
        self.corr_canvas.draw()
    
    def train_model(self):
        """Train weather forecasting model"""
        if self.data is None or self.data.empty:
            messagebox.showerror("Error", "No data available. Generate or load data first.")
            return
        
        self.status_var.set("Training model...")
        self.root.update()
        
        # Start loading screen
        loading = LoadingScreen(self.root, "Training model...")
        
        # Use threading to prevent UI freezing
        def train():
            try:
                model_type = self.model_type_var.get()
                test_size = self.test_size_var.get()
                
                loading.update_message(f"Training {model_type}...")
                self.root.update()
                
                # Train model
                error = self.weather_model.train(self.data, model_type=model_type, test_size=test_size)
                
                # Update UI
                self.root.after(0, self.update_model_info)
                self.root.after(0, self.update_model_visualizations)
                self.root.after(0, lambda: loading.destroy())
                self.root.after(0, lambda: self.status_var.set(f"Model trained successfully. RMSE: {error:.4f}"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to train model: {str(e)}"))
                self.root.after(0, lambda: loading.destroy())
                self.root.after(0, lambda: self.status_var.set("Error training model"))
        
        thread = threading.Thread(target=train)
        thread.daemon = True
        thread.start()
    
    def update_model_info(self):
        """Update model information display"""
        self.model_info_text.config(state='normal')
        self.model_info_text.delete(1.0, tk.END)
        
        info = self.weather_model.get_model_info()
        self.model_info_text.insert(tk.END, info)
        self.model_info_text.config(state='disabled')
    
    def update_model_visualizations(self):
        """Update model visualization charts"""
        if not self.weather_model.model:
            return
        
        # Feature importance plot
        self.model_fig.clear()
        ax = self.model_fig.add_subplot(111)
        
        if self.weather_model.feature_importance is not None:
            # Get feature names
            features = ['day_of_year', 'month', 'day', 'humidity', 'pressure', 'wind_speed', 
                       'precipitation', 'cloud_cover', 'day_of_week', 'season']
            
            # Filter features to match importance array length
            features = features[:len(self.weather_model.feature_importance)]
            
            # Sort by importance
            indices = np.argsort(self.weather_model.feature_importance)
            sorted_features = [features[i] for i in indices]
            sorted_importance = [self.weather_model.feature_importance[i] for i in indices]
            
            # Plot horizontal bar chart
            y_pos = np.arange(len(sorted_features))
            ax.barh(y_pos, sorted_importance, align='center', color=COLORS['primary'], alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features)
            ax.invert_yaxis()  # Display features from top to bottom
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            
            self.model_fig.tight_layout()
            self.model_canvas.draw()
        
        # Error analysis plot
        self.error_fig.clear()
        self.error_canvas.draw()
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if not self.weather_model.model:
            messagebox.showerror("Error", "No model available. Please train the model first.")
            return
        
        if self.data is None or self.data.empty:
            messagebox.showerror("Error", "No data available. Generate or load data first.")
            return
        
        self.status_var.set("Evaluating model...")
        self.root.update()
        
        # Compare predictions with actual values (using test-train split)
        from sklearn.model_selection import train_test_split
        
        # Features and target
        features = ['day_of_year', 'month', 'day', 'humidity', 'pressure', 'wind_speed']
        
        # Add additional features if they exist in the data
        additional_features = ['precipitation', 'cloud_cover', 'day_of_week', 'season']
        for feature in additional_features:
            if feature in self.data.columns:
                features.append(feature)
        
        X = self.data[features]
        y = self.data['temperature']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Make predictions
        y_pred = self.weather_model.model.predict(X_test)
        
        # Visualize predictions vs actual
        self.error_fig.clear()
        ax = self.error_fig.add_subplot(111)
        
        ax.scatter(y_test, y_pred, alpha=0.5, color=COLORS['primary'])
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Actual Temperature (°C)')
        ax.set_ylabel('Predicted Temperature (°C)')
        ax.set_title('Predicted vs Actual Temperature')
        
        # Add statistics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        stats_text = f"RMSE: {rmse:.2f}\nR²: {r2:.2f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        self.error_fig.tight_layout()
        self.error_canvas.draw()
        
        self.status_var.set("Model evaluation complete")


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()