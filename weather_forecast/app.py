import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import requests
import datetime
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import webbrowser
import platform
import json
import threading
import time

# Constants for styling
BG_COLOR = "#f5f5f5"
PRIMARY_COLOR = "#3498db"
SECONDARY_COLOR = "#2980b9"
TEXT_COLOR = "#333333"
FONT = ("Helvetica", 10)
HEADER_FONT = ("Helvetica", 12, "bold")

class WeatherData:
    """Class to handle weather data operations including API calls and sample data generation"""
    
    def __init__(self, api_key=None):
        # Initialize with default API key (in a real app, this should be secured)
        self.api_key = api_key or "fc3cfaf056bdc92c4a8f43f9858a51a6"
        self.base_url = "http://api.openweathermap.org/data/2.5/weather?"
        self.forecast_url = "http://api.openweathermap.org/data/2.5/forecast?"
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".weather_app_cache")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
    def get_current_weather(self, city):
        """Fetch current weather data for a given city with caching"""
        cache_file = os.path.join(self.cache_dir, f"{city.lower()}_current.json")
        
        # Check if we have a recent cache (less than 30 minutes old)
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 1800:  # 30 minutes in seconds
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass  # If cache read fails, fetch from API
        
        complete_url = f"{self.base_url}q={city}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()  # Raises exception for 4XX/5XX errors
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
                
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather: {e}")
            # If API call fails but we have cached data, use that
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass
            return None
        
    def get_forecast(self, city):
        """Fetch 5-day weather forecast for a given city with caching"""
        cache_file = os.path.join(self.cache_dir, f"{city.lower()}_forecast.json")
        
        # Check if we have a recent cache (less than 1 hour old)
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 3600:  # 1 hour in seconds
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass  # If cache read fails, fetch from API
        
        complete_url = f"{self.forecast_url}q={city}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the result
            with open(cache_file, 'w') as f:
                json.dump(data, f)
                
            return data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast: {e}")
            # If API call fails but we have cached data, use that
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass
            return None
    
    def clear_cache(self):
        """Clear all cached weather data"""
        for file in os.listdir(self.cache_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, file))
    
    def generate_sample_data(self, days=365):
        """Generate synthetic weather data for training purposes"""
        print("Generating sample weather data...")
        
        # Create date range
        dates = [datetime.datetime.now() - datetime.timedelta(days=x) 
                for x in range(days, 0, -1)]
        
        # Initialize empty lists for weather metrics
        temperature, humidity, pressure, wind_speed = [], [], [], []
        
        # Generate realistic weather patterns with seasonal variations
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            
            # Temperature with seasonal pattern and randomness
            temp = 15 + 10 * season_factor + np.random.normal(0, 3)
            temperature.append(temp)
            
            # Humidity inversely related to temperature
            hum = 70 - 20 * season_factor + np.random.normal(0, 10)
            humidity.append(max(min(hum, 100), 0))  # Clamp between 0-100%
            
            # Atmospheric pressure with patterns
            press = 1013 + 10 * np.sin(day_of_year / 20) + np.random.normal(0, 3)
            pressure.append(press)
            
            # Wind speed with seasonal variations
            wind = 5 + 3 * abs(season_factor) + np.random.normal(0, 2)
            wind_speed.append(max(wind, 0))  # Ensure non-negative
            
        # Create DataFrame with all weather metrics
        df = pd.DataFrame({
            'date': dates,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed
        })
        
        # Extract useful date features for the model
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        print("Sample data generation complete.")
        return df

class WeatherModel:
    """Machine learning model for weather forecasting"""
    
    def __init__(self, model_path='weather_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_names = ['day_of_year', 'month', 'day', 'humidity', 'pressure', 'wind_speed']
        self.training_history = {
            'rmse': [],
            'training_date': [],
            'data_size': []
        }
        
    def load_model(self):
        """Load a pre-trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.training_history = model_data.get('history', self.training_history)
                else:
                    self.model = model_data
                print("Model loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
    
    def save_model(self):
        """Save the current model to disk with training history"""
        if self.model:
            try:
                model_data = {
                    'model': self.model,
                    'history': self.training_history
                }
                joblib.dump(model_data, self.model_path)
                print("Model saved successfully.")
            except Exception as e:
                print(f"Error saving model: {e}")
    
    def train(self, data):
        """Train the forecasting model with provided data"""
        print("Training weather forecasting model...")
        
        # Prepare features and target variable
        X = data[self.feature_names]
        y = data['temperature']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Initialize and train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model performance
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Update training history
        self.training_history['rmse'].append(float(rmse))
        self.training_history['training_date'].append(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        self.training_history['data_size'].append(len(data))
        
        self.save_model()
        print(f"Model training complete. RMSE: {rmse:.2f}")
        return rmse
    
    def predict_temperature(self, day_of_year, month, day, humidity, pressure, wind_speed):
        """Make temperature prediction using the trained model"""
        if not self.model:
            if not self.load_model():
                raise Exception("Model not available. Please train the model first.")
        
        features = np.array([[day_of_year, month, day, humidity, pressure, wind_speed]])
        return self.model.predict(features)[0]
    
    def predict_next_days(self, data, days=7):
        """Generate weather predictions for the next n days"""
        if not self.model:
            if not self.load_model():
                raise Exception("Model not available. Please train the model first.")
        
        print(f"Generating {days}-day forecast...")
        
        # Get the most recent date in the dataset
        last_date = data['date'].max()
        predictions = []
        
        # Predict each day sequentially
        for i in range(1, days+1):
            next_date = last_date + datetime.timedelta(days=i)
            day_of_year = next_date.timetuple().tm_yday
            month = next_date.month
            day = next_date.day
            
            # Use moving average of recent data for other features
            last_5_days = data.sort_values('date', ascending=False).head(5)
            avg_humidity = last_5_days['humidity'].mean()
            avg_pressure = last_5_days['pressure'].mean()
            avg_wind_speed = last_5_days['wind_speed'].mean()
            
            # Make temperature prediction
            pred_temp = self.predict_temperature(
                day_of_year, month, day, 
                avg_humidity, avg_pressure, avg_wind_speed)
            
            # Add some daily variation
            humidity_variation = avg_humidity + np.random.normal(0, 3)
            pressure_variation = avg_pressure + np.random.normal(0, 1)
            wind_variation = max(0, avg_wind_speed + np.random.normal(0, 1))
            
            predictions.append({
                'date': next_date,
                'temperature': pred_temp,
                'humidity': humidity_variation,
                'pressure': pressure_variation,
                'wind_speed': wind_variation
            })
        
        print("Forecast generation complete.")
        return pd.DataFrame(predictions)
    
    def get_training_history(self):
        """Return model training history for visualization"""
        return self.training_history

class AppSettings:
    """Class to manage application settings"""
    
    def __init__(self, settings_file='weather_app_settings.json'):
        self.settings_file = settings_file
        self.default_settings = {
            'default_city': 'London',
            'api_key': 'fc3cfaf056bdc92c4a8f43f9858a51a6',  # Default key
            'theme': 'light',
            'data_days': 365,
            'forecast_days': 7,
            'recent_cities': []
        }
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from file or use defaults"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    # Merge with defaults in case the file is missing some settings
                    return {**self.default_settings, **settings}
            except:
                return self.default_settings
        return self.default_settings
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def get(self, key):
        """Get a setting value"""
        return self.settings.get(key, self.default_settings.get(key))
    
    def set(self, key, value):
        """Set a setting value and save"""
        self.settings[key] = value
        self.save_settings()
    
    def add_recent_city(self, city):
        """Add a city to the recent cities list"""
        if city not in self.settings['recent_cities']:
            self.settings['recent_cities'].insert(0, city)
            # Keep only the 5 most recent cities
            self.settings['recent_cities'] = self.settings['recent_cities'][:5]
            self.save_settings()

class WeatherApp:
    """Main application class with GUI interface"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_styles()
        
        # Initialize data and model components
        self.settings = AppSettings()
        self.weather_data = WeatherData(api_key=self.settings.get('api_key'))
        self.weather_model = WeatherModel()
        self.data = None
        
        # Threading flag to prevent multiple operations
        self.is_processing = False
        
        # Load or generate initial data
        self.initialize_data()
        
        # Build the user interface
        self.create_widgets()
        
        # Set initial state
        self.update_status("Application ready")
    
    def setup_window(self):
        """Configure the main application window"""
        self.root.title("Weather Forecast Pro")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        self.root.configure(bg=BG_COLOR)
        
        # Configure grid system
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Set window icon if available
        try:
            if platform.system() == 'Windows':
                self.root.iconbitmap('weather_icon.ico')
            else:
                icon = Image.open('weather_icon.png')
                photo = ImageTk.PhotoImage(icon)
                self.root.iconphoto(True, photo)
        except:
            pass  # Icon is optional
    
    def create_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        
        # Configure main frame style
        style.configure('TFrame', background=BG_COLOR)
        
        # Configure label styles
        style.configure('TLabel', background=BG_COLOR, font=FONT)
        style.configure('Header.TLabel', font=HEADER_FONT, foreground=PRIMARY_COLOR)
        
        # Configure button styles
        style.configure('TButton', font=FONT, padding=5)
        style.configure('Primary.TButton', foreground='white')
        style.map('Primary.TButton',
                background=[('active', SECONDARY_COLOR), ('pressed', SECONDARY_COLOR)])
        
        # Configure notebook style
        style.configure('TNotebook', background=BG_COLOR)
        style.configure('TNotebook.Tab', font=FONT, padding=[10, 5])
        
        # Configure combobox style
        style.configure('TCombobox', font=FONT, padding=5)
        
        # Configure progressbar style
        style.configure('TProgressbar', background=PRIMARY_COLOR)
    
    def initialize_data(self):
        """Load or generate initial weather data"""
        try:
            # Try to load existing model
            if not self.weather_model.load_model():
                # Generate data and train model if no model exists
                self.data = self.weather_data.generate_sample_data(days=self.settings.get('data_days'))
                self.train_model()
            else:
                # Just generate a small amount of recent data for display purposes
                self.data = self.weather_data.generate_sample_data(days=30)
        except Exception as e:
            messagebox.showerror("Initialization Error", 
                               f"Failed to initialize application: {str(e)}")
            self.root.after(100, self.root.destroy)
    
    def create_widgets(self):
        """Build all GUI components"""
        self.create_menu()
        self.create_main_notebook()
        self.create_status_bar()
    
    def create_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Data", command=self.generate_data)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_command(label="Save Data", command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Train Model", command=self.train_model)
        model_menu.add_command(label="Evaluate Model", command=self.evaluate_model)
        model_menu.add_command(label="Clear Cache", command=self.clear_cache)
        menubar.add_cascade(label="Model", menu=model_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_notebook(self):
        """Create the tabbed interface"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=(5, 0))
        
        # Create tabs
        self.forecast_tab = self.create_forecast_tab()
        self.comparison_tab = self.create_comparison_tab()
        self.data_tab = self.create_data_tab()
        self.model_tab = self.create_model_tab()
        self.settings_tab = self.create_settings_tab()
        
        self.notebook.add(self.forecast_tab, text="Weather Forecast")
        self.notebook.add(self.comparison_tab, text="City Comparison")
        self.notebook.add(self.data_tab, text="Weather Data")
        self.notebook.add(self.model_tab, text="Model Analysis")
        self.notebook.add(self.settings_tab, text="Settings")
    
    def create_forecast_tab(self):
        """Build the forecast tab"""
        tab = ttk.Frame(self.notebook)
        
        # Search panel
        search_frame = ttk.Frame(tab)
        search_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(search_frame, text="Enter Location:", font=HEADER_FONT).pack(side=tk.LEFT, padx=5)
        
        # Create combobox with recent cities
        self.city_var = tk.StringVar(value=self.settings.get('default_city'))
        self.city_combo = ttk.Combobox(search_frame, textvariable=self.city_var, 
                                     values=self.settings.get('recent_cities'),
                                     width=30, font=FONT)
        self.city_combo.pack(side=tk.LEFT, padx=5, expand=True, fill='x')
        
        search_btn = ttk.Button(search_frame, text="Get Forecast", 
                              style='Primary.TButton', command=self.get_forecast)
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # Current weather display with icons
        current_frame = ttk.LabelFrame(tab, text=" Current Weather ", padding=10)
        current_frame.pack(fill='x', padx=10, pady=5)
        
        # Create two columns for current weather
        current_left = ttk.Frame(current_frame)
        current_left.pack(side=tk.LEFT, fill='both', expand=True)
        
        current_right = ttk.Frame(current_frame)
        current_right.pack(side=tk.RIGHT, fill='both', expand=True)
        
        # Weather icon placeholder
        self.weather_icon_frame = ttk.Label(current_left)
        self.weather_icon_frame.pack(side=tk.TOP, padx=5, pady=5)
        
        # Try to load a default weather icon
        try:
            default_icon = Image.open('weather_icons/default.png')
            default_icon = default_icon.resize((64, 64), Image.LANCZOS)
            self.weather_icon = ImageTk.PhotoImage(default_icon)
            self.weather_icon_frame.config(image=self.weather_icon)
        except:
            self.weather_icon_frame.config(text="[Icon]", font=HEADER_FONT)
        
        # Current weather text area
        self.current_weather_text = tk.Text(current_right, height=6, width=40, 
                                          font=FONT, wrap=tk.WORD, padx=5, pady=5)
        self.current_weather_text.pack(fill='both', expand=True)
        
        # Forecast visualization with tabs for different metrics
        forecast_frame = ttk.LabelFrame(tab, text=" 7-Day Forecast ", padding=10)
        forecast_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create a notebook for different forecast views
        self.forecast_notebook = ttk.Notebook(forecast_frame)
        self.forecast_notebook.pack(fill='both', expand=True)
        
        # Temperature tab
        temp_frame = ttk.Frame(self.forecast_notebook)
        self.forecast_notebook.add(temp_frame, text="Temperature")
        
        self.fig_temp = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.ax_temp = self.fig_temp.add_subplot(111)
        self.ax_temp.set_facecolor(BG_COLOR)
        
        self.canvas_temp = FigureCanvasTkAgg(self.fig_temp, master=temp_frame)
        self.canvas_temp.get_tk_widget().pack(fill='both', expand=True)
        
        # Humidity tab
        humid_frame = ttk.Frame(self.forecast_notebook)
        self.forecast_notebook.add(humid_frame, text="Humidity")
        
        self.fig_humid = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.ax_humid = self.fig_humid.add_subplot(111)
        self.ax_humid.set_facecolor(BG_COLOR)
        
        self.canvas_humid = FigureCanvasTkAgg(self.fig_humid, master=humid_frame)
        self.canvas_humid.get_tk_widget().pack(fill='both', expand=True)
        
        # Pressure tab
        pressure_frame = ttk.Frame(self.forecast_notebook)
        self.forecast_notebook.add(pressure_frame, text="Pressure")
        
        self.fig_pressure = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.ax_pressure = self.fig_pressure.add_subplot(111)
        self.ax_pressure.set_facecolor(BG_COLOR)
        
        self.canvas_pressure = FigureCanvasTkAgg(self.fig_pressure, master=pressure_frame)
        self.canvas_pressure.get_tk_widget().pack(fill='both', expand=True)
        
        # Wind speed tab
        wind_frame = ttk.Frame(self.forecast_notebook)
        self.forecast_notebook.add(wind_frame, text="Wind Speed")
        
        self.fig_wind = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.ax_wind = self.fig_wind.add_subplot(111)
        self.ax_wind.set_facecolor(BG_COLOR)
        
        self.canvas_wind = FigureCanvasTkAgg(self.fig_wind, master=wind_frame)
        self.canvas_wind.get_tk_widget().pack(fill='both', expand=True)
        
        return tab
    
    def create_comparison_tab(self):
        """Build the city comparison tab"""
        tab = ttk.Frame(self.notebook)
        
        # Cities selection panel
        cities_frame = ttk.Frame(tab)
        cities_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(cities_frame, text="Compare Cities:", font=HEADER_FONT).grid(row=0, column=0, padx=5, pady=5)
        
        # City 1
        ttk.Label(cities_frame, text="City 1:").grid(row=1, column=0, padx=5, pady=5)
        self.city1_var = tk.StringVar(value="London")
        city1_entry = ttk.Entry(cities_frame, textvariable=self.city1_var, width=20)
        city1_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # City 2
        ttk.Label(cities_frame, text="City 2:").grid(row=1, column=2, padx=5, pady=5)
        self.city2_var = tk.StringVar(value="New York")
        city2_entry = ttk.Entry(cities_frame, textvariable=self.city2_var, width=20)
        city2_entry.grid(row=1, column=3, padx=5, pady=5)
        
        # Compare button
        compare_btn = ttk.Button(cities_frame, text="Compare", 
                               style='Primary.TButton', command=self.compare_cities)
        compare_btn.grid(row=1, column=4, padx=5, pady=5)
        
        # Comparison chart
        chart_frame = ttk.LabelFrame(tab, text=" Temperature Comparison ", padding=10)
        chart_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.comparison_fig = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.comparison_ax = self.comparison_fig.add_subplot(111)
        self.comparison_ax.set_facecolor(BG_COLOR)
        
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, master=chart_frame)
        self.comparison_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Summary text
        summary_frame = ttk.LabelFrame(tab, text=" Comparison Summary ", padding=10)
        summary_frame.pack(fill='x', padx=10, pady=5)
        
        self.comparison_text = tk.Text(summary_frame, height=6, width=80, 
                                      font=FONT, wrap=tk.WORD)
        self.comparison_text.pack(fill='both')
        
        return tab
    
    def create_data_tab(self):
        """Build the data tab"""
        tab = ttk.Frame(self.notebook)
        
        # Control panel
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="Generate New Data", 
                  style='Primary.TButton', command=self.generate_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Load Data", 
                  command=self.load_data).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Save Data", 
                  command=self.save_data).pack(side=tk.LEFT, padx=5)
        
        # Data visualization controls
        viz_control_frame = ttk.Frame(tab)
        viz_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(viz_control_frame, text="Visualization:").pack(side=tk.LEFT, padx=5)
        
        self.viz_var = tk.StringVar(value="Temperature")
        viz_options = ["Temperature", "Humidity", "Pressure", "Wind Speed"]
        viz_dropdown = ttk.Combobox(viz_control_frame, textvariable=self.viz_var, 
                                  values=viz_options, state="readonly", width=15)
        viz_dropdown.pack(side=tk.LEFT, padx=5)
        viz_dropdown.bind("<<ComboboxSelected>>", self.update_data_visualization)
        
        ttk.Button(viz_control_frame, text="Refresh", 
                 command=self.update_data_visualization).pack(side=tk.LEFT, padx=5)
        
        # Data visualization frame
        self.data_viz_frame = ttk.LabelFrame(tab, text=" Data Visualization ", padding=10)
        self.data_viz_frame.pack(fill='x', padx=10, pady=5)
        
        self.data_fig = Figure(figsize=(8, 3), dpi=100, facecolor=BG_COLOR)
        self.data_ax = self.data_fig.add_subplot(111)
        self.data_ax.set_facecolor(BG_COLOR)
        
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, master=self.data_viz_frame)
        self.data_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Data preview
        preview_frame = ttk.LabelFrame(tab, text=" Data Preview ", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_text = tk.Text(preview_frame, height=10, width=80, 
                               font=('Courier', 10), wrap=tk.NONE)
        self.data_text.pack(fill='both', expand=True)
        
        # Add scrollbars
        y_scroll = ttk.Scrollbar(preview_frame, orient='vertical', 
                               command=self.data_text.yview)
        y_scroll.pack(side='right', fill='y')
        self.data_text.configure(yscrollcommand=y_scroll.set)
        
        x_scroll = ttk.Scrollbar(preview_frame, orient='horizontal', 
                               command=self.data_text.xview)
        x_scroll.pack(side='bottom', fill='x')
        self.data_text.configure(xscrollcommand=x_scroll.set)
        
        # Show initial data
        self.view_data()
        self.update_data_visualization()
        
        return tab
    
    def create_model_tab(self):
        """Build the model analysis tab"""
        tab = ttk.Frame(self.notebook)
        
        # Control panel
        control_frame = ttk.Frame(tab)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(control_frame, text="Train Model", 
                  style='Primary.TButton', command=self.train_model).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Evaluate Model", 
                  command=self.evaluate_model).pack(side=tk.LEFT, padx=5)
        
        # Model info
        info_frame = ttk.LabelFrame(tab, text=" Model Information ", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.model_info_text = tk.Text(info_frame, height=6, width=80, 
                                     font=FONT, wrap=tk.WORD)
        self.model_info_text.pack(fill='both')
        
        # Notebook for model analysis
        model_notebook = ttk.Notebook(tab)
        model_notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Feature importance tab
        feature_frame = ttk.Frame(model_notebook)
        model_notebook.add(feature_frame, text="Feature Importance")
        
        self.model_fig = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.model_ax = self.model_fig.add_subplot(111)
        self.model_ax.set_facecolor(BG_COLOR)
        
        self.model_canvas = FigureCanvasTkAgg(self.model_fig, master=feature_frame)
        self.model_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Training history tab
        history_frame = ttk.Frame(model_notebook)
        model_notebook.add(history_frame, text="Training History")
        
        self.history_fig = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.history_ax = self.history_fig.add_subplot(111)
        self.history_ax.set_facecolor(BG_COLOR)
        
        self.history_canvas = FigureCanvasTkAgg(self.history_fig, master=history_frame)
        self.history_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Prediction error tab
        error_frame = ttk.Frame(model_notebook)
        model_notebook.add(error_frame, text="Prediction Error")
        
        self.error_fig = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.error_ax = self.error_fig.add_subplot(111)
        self.error_ax.set_facecolor(BG_COLOR)
        
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, master=error_frame)
        self.error_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        return tab
    
    def create_settings_tab(self):
        """Build the settings tab"""
        tab = ttk.Frame(self.notebook)
        
        settings_frame = ttk.LabelFrame(tab, text=" Application Settings ", padding=20)
        settings_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # API Key
        ttk.Label(settings_frame, text="OpenWeatherMap API Key:", 
                font=FONT).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.api_key_var = tk.StringVar(value=self.settings.get('api_key'))
        api_key_entry = ttk.Entry(settings_frame, textvariable=self.api_key_var, width=40)
        api_key_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Default City
        ttk.Label(settings_frame, text="Default City:", 
                font=FONT).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.default_city_var = tk.StringVar(value=self.settings.get('default_city'))
        default_city_entry = ttk.Entry(settings_frame, textvariable=self.default_city_var, width=40)
        default_city_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Data Generation Days
        ttk.Label(settings_frame, text="Sample Data Days:", 
                font=FONT).grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.data_days_var = tk.IntVar(value=self.settings.get('data_days'))
        data_days_spinner = ttk.Spinbox(settings_frame, from_=30, to=1000, 
                                      textvariable=self.data_days_var, width=10)
        data_days_spinner.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Forecast Days
        ttk.Label(settings_frame, text="Forecast Days:", 
                font=FONT).grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.forecast_days_var = tk.IntVar(value=self.settings.get('forecast_days'))
        forecast_days_spinner = ttk.Spinbox(settings_frame, from_=1, to=14, 
                                          textvariable=self.forecast_days_var, width=10)
        forecast_days_spinner.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Theme selection
        ttk.Label(settings_frame, text="Theme:", 
                font=FONT).grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.theme_var = tk.StringVar(value=self.settings.get('theme'))
        theme_combo = ttk.Combobox(settings_frame, textvariable=self.theme_var, 
                                  values=["light", "dark"], state="readonly", width=10)
        theme_combo.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Clear cache option
        ttk.Label(settings_frame, text="Data Cache:", 
                font=FONT).grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        
        ttk.Button(settings_frame, text="Clear Cache", 
                 command=self.clear_cache).grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Save settings button
        ttk.Button(settings_frame, text="Save Settings", style='Primary.TButton',
                 command=self.save_settings).grid(row=6, column=0, columnspan=2, pady=20)
        
        return tab
    
    def create_status_bar(self):
        """Create the status bar at bottom of window"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill='x', padx=10, pady=(0, 5))
        
        self.progress = ttk.Progressbar(status_bar, mode='indeterminate', length=150)
        self.progress.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(status_bar, textvariable=self.status_var, 
                 relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.LEFT, fill='x', expand=True)
    
    def update_status(self, message, start_progress=False, stop_progress=False):
        """Update the status bar message and progress"""
        self.status_var.set(message)
        
        if start_progress:
            self.progress.start(10)
        
        if stop_progress:
            self.progress.stop()
            
        self.root.update_idletasks()
    
    def get_forecast(self):
        """Fetch and display weather forecast"""
        if self.is_processing:
            return
            
        city = self.city_var.get().strip()
        if not city:
            messagebox.showwarning("Input Error", "Please enter a city name.")
            return
        
        self.is_processing = True
        self.update_status(f"Fetching weather for {city}...", start_progress=True)
        
        # Start the forecast fetch in a separate thread
        threading.Thread(target=self._get_forecast_task, args=(city,), daemon=True).start()
    
    def _get_forecast_task(self, city):
        """Background task to fetch forecast data"""
        try:
            # Add city to recent cities
            self.settings.add_recent_city(city)
            
            # Update the city combo values
            self.city_combo['values'] = self.settings.get('recent_cities')
            
            # Get predictions from our model
            forecast_days = self.settings.get('forecast_days')
            predictions = self.weather_model.predict_next_days(self.data, days=forecast_days)
            
            # Display current weather (using first prediction)
            current = predictions.iloc[0]
            weather_text = f"Location: {city}\n"
            weather_text += f"Date: {current['date'].strftime('%Y-%m-%d')}\n"
            weather_text += f"Temperature: {current['temperature']:.1f}°C\n"
            weather_text += f"Humidity: {current['humidity']:.1f}%\n"
            weather_text += f"Pressure: {current['pressure']:.1f} hPa\n"
            weather_text += f"Wind Speed: {current['wind_speed']:.1f} m/s"
            
            # Update UI from main thread
            self.root.after(0, lambda: self._update_forecast_ui(city, weather_text, predictions))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Forecast Error", 
                                                         f"Failed to generate forecast: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Forecast failed", stop_progress=True))
            
        finally:
            self.is_processing = False
    
    def _update_forecast_ui(self, city, weather_text, predictions):
        """Update the forecast UI elements from the main thread"""
        # Update current weather text
        self.current_weather_text.config(state=tk.NORMAL)
        self.current_weather_text.delete(1.0, tk.END)
        self.current_weather_text.insert(tk.END, weather_text)
        self.current_weather_text.config(state=tk.DISABLED)
        
        # Update temperature chart
        self.ax_temp.clear()
        self.ax_temp.plot(predictions['date'], predictions['temperature'], 
                        color='#e74c3c', marker='o', linestyle='-', 
                        linewidth=2, markersize=8, label='Temperature (°C)')
        self.ax_temp.set_xlabel('Date', fontsize=10)
        self.ax_temp.set_ylabel('Temperature (°C)', fontsize=10)
        self.ax_temp.set_title(f'Temperature Forecast for {city}', fontsize=12, pad=20)
        self.ax_temp.grid(True, linestyle='--', alpha=0.7)
        self.ax_temp.tick_params(axis='both', which='major', labelsize=9)
        self.fig_temp.autofmt_xdate()
        self.ax_temp.legend(loc='upper right', fontsize=9)
        self.canvas_temp.draw()
        
        # Update humidity chart
        self.ax_humid.clear()
        self.ax_humid.plot(predictions['date'], predictions['humidity'], 
                         color='#3498db', marker='s', linestyle='-', 
                         linewidth=2, markersize=8, label='Humidity (%)')
        self.ax_humid.set_xlabel('Date', fontsize=10)
        self.ax_humid.set_ylabel('Humidity (%)', fontsize=10)
        self.ax_humid.set_title(f'Humidity Forecast for {city}', fontsize=12, pad=20)
        self.ax_humid.grid(True, linestyle='--', alpha=0.7)
        self.ax_humid.tick_params(axis='both', which='major', labelsize=9)
        self.fig_humid.autofmt_xdate()
        self.ax_humid.legend(loc='upper right', fontsize=9)
        self.canvas_humid.draw()
        
        # Update pressure chart
        self.ax_pressure.clear()
        self.ax_pressure.plot(predictions['date'], predictions['pressure'], 
                            color='#9b59b6', marker='^', linestyle='-', 
                            linewidth=2, markersize=8, label='Pressure (hPa)')
        self.ax_pressure.set_xlabel('Date', fontsize=10)
        self.ax_pressure.set_ylabel('Pressure (hPa)', fontsize=10)
        self.ax_pressure.set_title(f'Atmospheric Pressure Forecast for {city}', fontsize=12, pad=20)
        self.ax_pressure.grid(True, linestyle='--', alpha=0.7)
        self.ax_pressure.tick_params(axis='both', which='major', labelsize=9)
        self.fig_pressure.autofmt_xdate()
        self.ax_pressure.legend(loc='upper right', fontsize=9)
        self.canvas_pressure.draw()
        
        # Update wind speed chart
        self.ax_wind.clear()
        self.ax_wind.plot(predictions['date'], predictions['wind_speed'], 
                        color='#2ecc71', marker='*', linestyle='-', 
                        linewidth=2, markersize=10, label='Wind Speed (m/s)')
        self.ax_wind.set_xlabel('Date', fontsize=10)
        self.ax_wind.set_ylabel('Wind Speed (m/s)', fontsize=10)
        self.ax_wind.set_title(f'Wind Speed Forecast for {city}', fontsize=12, pad=20)
        self.ax_wind.grid(True, linestyle='--', alpha=0.7)
        self.ax_wind.tick_params(axis='both', which='major', labelsize=9)
        self.fig_wind.autofmt_xdate()
        self.ax_wind.legend(loc='upper right', fontsize=9)
        self.canvas_wind.draw()
        
        self.update_status(f"Weather forecast for {city} displayed", stop_progress=True)
    
    def compare_cities(self):
        """Compare weather forecasts between two cities"""
        if self.is_processing:
            return
            
        city1 = self.city1_var.get().strip()
        city2 = self.city2_var.get().strip()
        
        if not city1 or not city2:
            messagebox.showwarning("Input Error", "Please enter both city names.")
            return
        
        self.is_processing = True
        self.update_status(f"Comparing weather for {city1} and {city2}...", start_progress=True)
        
        # Start the comparison in a separate thread
        threading.Thread(target=self._compare_cities_task, 
                       args=(city1, city2), daemon=True).start()
    
    def _compare_cities_task(self, city1, city2):
        """Background task to compare cities"""
        try:
            # Get predictions for both cities
            forecast_days = self.settings.get('forecast_days')
            predictions1 = self.weather_model.predict_next_days(self.data, days=forecast_days)
            predictions2 = self.weather_model.predict_next_days(self.data, days=forecast_days)
            
            # Add some variation between cities
            predictions2['temperature'] = predictions2['temperature'] + np.random.uniform(-5, 5)
            predictions2['humidity'] = np.clip(predictions2['humidity'] + np.random.uniform(-10, 10), 0, 100)
            predictions2['pressure'] = predictions2['pressure'] + np.random.uniform(-5, 5)
            predictions2['wind_speed'] = np.clip(predictions2['wind_speed'] + np.random.uniform(-2, 2), 0, None)
            
            # Calculate statistics
            avg_temp1 = predictions1['temperature'].mean()
            avg_temp2 = predictions2['temperature'].mean()
            temp_diff = avg_temp1 - avg_temp2
            
            max_temp1 = predictions1['temperature'].max()
            max_temp2 = predictions2['temperature'].max()
            
            min_temp1 = predictions1['temperature'].min()
            min_temp2 = predictions2['temperature'].min()
            
            # Prepare comparison text
            comparison_text = f"Temperature Comparison ({forecast_days}-Day Forecast):\n\n"
            comparison_text += f"{city1}: Average {avg_temp1:.1f}°C (Min: {min_temp1:.1f}°C, Max: {max_temp1:.1f}°C)\n"
            comparison_text += f"{city2}: Average {avg_temp2:.1f}°C (Min: {min_temp2:.1f}°C, Max: {max_temp2:.1f}°C)\n\n"
            
            if temp_diff > 0:
                comparison_text += f"{city1} is warmer than {city2} by {abs(temp_diff):.1f}°C on average."
            elif temp_diff < 0:
                comparison_text += f"{city2} is warmer than {city1} by {abs(temp_diff):.1f}°C on average."
            else:
                comparison_text += f"{city1} and {city2} have similar average temperatures."
            
            # Update UI from main thread
            self.root.after(0, lambda: self._update_comparison_ui(
                city1, city2, predictions1, predictions2, comparison_text))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Comparison Error", 
                                                         f"Failed to compare cities: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Comparison failed", stop_progress=True))
            
        finally:
            self.is_processing = False
    
    def _update_comparison_ui(self, city1, city2, predictions1, predictions2, comparison_text):
        """Update the comparison UI elements from the main thread"""
        # Update comparison chart
        self.comparison_ax.clear()
        
        self.comparison_ax.plot(predictions1['date'], predictions1['temperature'], 
                              color='#e74c3c', marker='o', linestyle='-', 
                              linewidth=2, markersize=8, label=f'{city1}')
        
        self.comparison_ax.plot(predictions2['date'], predictions2['temperature'], 
                              color='#3498db', marker='s', linestyle='-', 
                              linewidth=2, markersize=8, label=f'{city2}')
        
        self.comparison_ax.set_xlabel('Date', fontsize=10)
        self.comparison_ax.set_ylabel('Temperature (°C)', fontsize=10)
        self.comparison_ax.set_title(f'Temperature Comparison: {city1} vs {city2}', 
                                   fontsize=12, pad=20)
        self.comparison_ax.grid(True, linestyle='--', alpha=0.7)
        self.comparison_ax.tick_params(axis='both', which='major', labelsize=9)
        self.comparison_fig.autofmt_xdate()
        self.comparison_ax.legend(loc='upper right', fontsize=9)
        
        self.comparison_canvas.draw()
        
        # Update comparison text
        self.comparison_text.config(state=tk.NORMAL)
        self.comparison_text.delete(1.0, tk.END)
        self.comparison_text.insert(tk.END, comparison_text)
        self.comparison_text.config(state=tk.DISABLED)
        
        self.update_status(f"Weather comparison for {city1} and {city2} displayed", stop_progress=True)
    
    def generate_data(self):
        """Generate new sample weather data"""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.update_status("Generating sample weather data...", start_progress=True)
        
        # Start the data generation in a separate thread
        threading.Thread(target=self._generate_data_task, daemon=True).start()
    
    def _generate_data_task(self):
        """Background task to generate data"""
        try:
            days = self.settings.get('data_days')
            self.data = self.weather_data.generate_sample_data(days=days)
            
            # Update UI from main thread
            self.root.after(0, self.view_data)
            self.root.after(0, self.update_data_visualization)
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                                                       "New sample data generated successfully!"))
            self.root.after(0, lambda: self.update_status("Sample data generated", stop_progress=True))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", 
                                                         f"Failed to generate data: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Data generation failed", stop_progress=True))
            
        finally:
            self.is_processing = False
    
    def load_data(self):
        """Load weather data from file"""
        if self.is_processing:
            return
            
        filepath = filedialog.askopenfilename(
            title="Select Weather Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if not filepath:
            return
        
        self.is_processing = True
        self.update_status(f"Loading data from {os.path.basename(filepath)}...", start_progress=True)
        
        # Start the data loading in a separate thread
        threading.Thread(target=self._load_data_task, args=(filepath,), daemon=True).start()
    
    def _load_data_task(self, filepath):
        """Background task to load data"""
        try:
            self.data = pd.read_csv(filepath, parse_dates=['date'])
            
            # Ensure required columns exist
            required_columns = ['date', 'temperature', 'humidity', 'pressure', 'wind_speed']
            missing_columns = [col for col in required_columns if col not in self.data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Extract date features if they don't exist
            if 'day_of_year' not in self.data.columns:
                self.data['day_of_year'] = self.data['date'].dt.dayofyear
            if 'month' not in self.data.columns:
                self.data['month'] = self.data['date'].dt.month
            if 'day' not in self.data.columns:
                self.data['day'] = self.data['date'].dt.day
            
            # Update UI from main thread
            self.root.after(0, self.view_data)
            self.root.after(0, self.update_data_visualization)
            self.root.after(0, lambda: messagebox.showinfo("Success", "Data loaded successfully!"))
            self.root.after(0, lambda: self.update_status(
                f"Data loaded from {os.path.basename(filepath)}", stop_progress=True))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load data: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Data load failed", stop_progress=True))
            
        finally:
            self.is_processing = False
    
    def save_data(self):
        """Save current weather data to file"""
        if self.is_processing:
            return
            
        if self.data is None:
            messagebox.showwarning("No Data", "No data available to save.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Weather Data",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if not filepath:
            return
        
        self.is_processing = True
        self.update_status(f"Saving data to {os.path.basename(filepath)}...", start_progress=True)
        
        # Start the data saving in a separate thread
        threading.Thread(target=self._save_data_task, args=(filepath,), daemon=True).start()
    
    def _save_data_task(self, filepath):
        """Background task to save data"""
        try:
            self.data.to_csv(filepath, index=False)
            
            # Update UI from main thread
            self.root.after(0, lambda: messagebox.showinfo("Success", "Data saved successfully!"))
            self.root.after(0, lambda: self.update_status(
                f"Data saved to {os.path.basename(filepath)}", stop_progress=True))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save data: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Data save failed", stop_progress=True))
            
        finally:
            self.is_processing = False
    
    def view_data(self):
        """Display the current weather data"""
        if self.data is None:
            return
        
        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete(1.0, tk.END)
        
        # Format the data display
        data_preview = self.data.head(50).to_string(index=False)
        self.data_text.insert(tk.END, data_preview)
        self.data_text.config(state=tk.DISABLED)
    
    def update_data_visualization(self, event=None):
        """Update the data visualization based on the selected metric"""
        if self.data is None:
            return
        
        metric = self.viz_var.get()
        
        self.data_ax.clear()
        
        if metric == "Temperature":
            y_data = self.data['temperature']
            color = '#e74c3c'
            ylabel = 'Temperature (°C)'
        elif metric == "Humidity":
            y_data = self.data['humidity']
            color = '#3498db'
            ylabel = 'Humidity (%)'
        elif metric == "Pressure":
            y_data = self.data['pressure']
            color = '#9b59b6'
            ylabel = 'Pressure (hPa)'
        elif metric == "Wind Speed":
            y_data = self.data['wind_speed']
            color = '#2ecc71'
            ylabel = 'Wind Speed (m/s)'
        else:
            return
        
        # Plot data with a rolling average
        self.data_ax.plot(self.data['date'], y_data, 
                        color=color, alpha=0.3, linestyle='-', 
                        linewidth=1, label='Raw Data')
        
        # Add rolling average
        window_size = min(30, len(y_data) // 10)  # Adjust window size based on data length
        if window_size > 0:
            rolling_avg = y_data.rolling(window=window_size, center=True).mean()
            self.data_ax.plot(self.data['date'], rolling_avg, 
                            color=color, linestyle='-', 
                            linewidth=2, label=f'{window_size}-Day Average')
        
        self.data_ax.set_xlabel('Date', fontsize=10)
        self.data_ax.set_ylabel(ylabel, fontsize=10)
        self.data_ax.set_title(f'Historical {metric} Data', fontsize=12, pad=20)
        self.data_ax.grid(True, linestyle='--', alpha=0.7)
        self.data_ax.tick_params(axis='both', which='major', labelsize=9)
        self.data_ax.legend(loc='upper right', fontsize=9)
        self.data_fig.autofmt_xdate()
        self.data_canvas.draw()

    def train_model(self):
        """Train the weather forecasting model"""
        if self.is_processing:
            return
        
        if self.data is None:
            messagebox.showwarning("No Data", "No data available for training.")
            return
        
        self.is_processing = True
        self.update_status("Training the model...", start_progress=True)
        
        # Start training in a separate thread
        threading.Thread(target=self._train_model_task, daemon=True).start()
    
    def _train_model_task(self):
        """Background task to train the model"""
        try:
            rmse = self.weather_model.train(self.data)
            
            # Update UI from main thread
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                                                         f"Model trained successfully. RMSE: {rmse:.2f}"))
            self.root.after(0, lambda: self.update_status("Model training complete", stop_progress=True))
            self.root.after(0, self.view_training_history)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to train model: {str(e)}"))
            self.root.after(0, lambda: self.update_status("Model training failed", stop_progress=True))
        
        finally:
            self.is_processing = False
    
    def view_training_history(self):
        """Display the training history in the model tab"""
        history = self.weather_model.get_training_history()
        
        self.history_ax.clear()
        self.history_ax.plot(history['training_date'], history['rmse'], 
                             color='#3498db', marker='o', linestyle='-', 
                             linewidth=2, label='RMSE')
        self.history_ax.set_xlabel('Training Date', fontsize=10)
        self.history_ax.set_ylabel('RMSE', fontsize=10)
        self.history_ax.set_title('Model Training History', fontsize=12, pad=20)
        self.history_ax.grid(True, linestyle='--', alpha=0.7)
        self.history_ax.tick_params(axis='both', which='major', labelsize=9)
        self.history_fig.autofmt_xdate()
        self.history_ax.legend(loc='upper right', fontsize=9)
        self.history_canvas.draw()
    
    def evaluate_model(self):
        """Evaluate the model's performance"""
        if not self.weather_model.model:
            messagebox.showwarning("No Model", "No trained model available for evaluation.")
            return
        
        # Display feature importance
        feature_importances = self.weather_model.model.feature_importances_
        features = self.weather_model.feature_names
        
        self.model_ax.clear()
        self.model_ax.barh(features, feature_importances, color='#2ecc71')
        self.model_ax.set_xlabel('Importance', fontsize=10)
        self.model_ax.set_title('Feature Importance', fontsize=12, pad=20)
        self.model_ax.grid(True, linestyle='--', alpha=0.7)
        self.model_ax.tick_params(axis='both', which='major', labelsize=9)
        self.model_canvas.draw()
        
        # Update model information
        self.model_info_text.config(state=tk.NORMAL)
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(tk.END, f"Model Path: {self.weather_model.model_path}\n")
        self.model_info_text.insert(tk.END, f"Trained on {len(self.data)} data points.\n")
        self.model_info_text.config(state=tk.DISABLED)
        
        self.update_status("Model evaluation complete")

    def clear_cache(self):
        """Clear cached weather data"""
        self.weather_data.clear_cache()
        messagebox.showinfo("Cache Cleared", "All cached weather data has been cleared.")
        self.update_status("Cache cleared")
    
    def save_settings(self):
        """Save the application settings"""
        self.settings.set('api_key', self.api_key_var.get())
        self.settings.set('default_city', self.default_city_var.get())
        self.settings.set('data_days', self.data_days_var.get())
        self.settings.set('forecast_days', self.forecast_days_var.get())
        self.settings.set('theme', self.theme_var.get())
        messagebox.showinfo("Settings Saved", "Application settings have been saved.")
        self.update_status("Settings saved")
    
    def show_settings(self):
        """Open the settings tab"""
        self.notebook.select(self.settings_tab)
    
    def show_about(self):
        """Display about information"""
        messagebox.showinfo("About", 
                            "Weather Forecast Pro\n\n"
                            "Version 1.0\n"
                            "Developed by Kavin0003\n\n"
                            "This application provides weather forecasting "
                            "and data visualization using machine learning.")
    
    def show_documentation(self):
        """Open the documentation URL"""
        webbrowser.open("https://github.com/Kavin0003/weather-app-docs")  # Example link

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()