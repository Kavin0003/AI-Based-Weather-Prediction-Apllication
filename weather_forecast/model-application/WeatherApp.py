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
        
        
    def get_current_weather(self, city):
        """Fetch current weather data for a given city"""
        complete_url = f"{self.base_url}q={city}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()  # Raises exception for 4XX/5XX errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather: {e}")
            return None
        
    def get_forecast(self, city):
        """Fetch 5-day weather forecast for a given city"""
        complete_url = f"{self.forecast_url}q={city}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(complete_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast: {e}")
            return None
    
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
        
    def load_model(self):
        """Load a pre-trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Model loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
    
    def save_model(self):
        """Save the current model to disk"""
        if self.model:
            try:
                joblib.dump(self.model, self.model_path)
                print("Model saved successfully.")
            except Exception as e:
                print(f"Error saving model: {e}")
    
    def train(self, data):
        """Train the forecasting model with provided data"""
        print("Training weather forecasting model...")
        
        # Prepare features and target variable
        X = data[['day_of_year', 'month', 'day', 'humidity', 'pressure', 'wind_speed']]
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
            
            predictions.append({
                'date': next_date,
                'temperature': pred_temp,
                'humidity': avg_humidity,
                'pressure': avg_pressure,
                'wind_speed': avg_wind_speed
            })
        
        print("Forecast generation complete.")
        return pd.DataFrame(predictions)

class WeatherApp:
    """Main application class with GUI interface"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_styles()
        
        # Initialize data and model components
        self.weather_data = WeatherData()
        self.weather_model = WeatherModel()
        self.data = None
        
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
        style.configure('Primary.TButton', foreground='white', background=PRIMARY_COLOR)
        style.map('Primary.TButton',
                background=[('active', SECONDARY_COLOR), ('pressed', SECONDARY_COLOR)])
        
        # Configure notebook style
        style.configure('TNotebook', background=BG_COLOR)
        style.configure('TNotebook.Tab', font=FONT, padding=[10, 5])
    
    def initialize_data(self):
        """Load or generate initial weather data"""
        try:
            # Try to load existing model or generate new data
            if not self.weather_model.load_model():
                self.data = self.weather_data.generate_sample_data()
                self.train_model()
            else:
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
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
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
        self.data_tab = self.create_data_tab()
        self.model_tab = self.create_model_tab()
        
        self.notebook.add(self.forecast_tab, text="Weather Forecast")
        self.notebook.add(self.data_tab, text="Weather Data")
        self.notebook.add(self.model_tab, text="Model Analysis")
    
    def create_forecast_tab(self):
        """Build the forecast tab"""
        tab = ttk.Frame(self.notebook)
        
        # Search panel
        search_frame = ttk.Frame(tab)
        search_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(search_frame, text="Enter Location:", font=HEADER_FONT).pack(side=tk.LEFT, padx=5)
        
        self.city_var = tk.StringVar()
        city_entry = ttk.Entry(search_frame, textvariable=self.city_var, width=30, font=FONT)
        city_entry.pack(side=tk.LEFT, padx=5, expand=True, fill='x')
        
        search_btn = ttk.Button(search_frame, text="Get Forecast", 
                              style='Primary.TButton', command=self.get_forecast)
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # Current weather display
        current_frame = ttk.LabelFrame(tab, text=" Current Weather ", padding=10)
        current_frame.pack(fill='x', padx=10, pady=5)
        
        self.current_weather_text = tk.Text(current_frame, height=6, width=60, 
                                          font=FONT, wrap=tk.WORD, padx=5, pady=5)
        self.current_weather_text.pack(fill='both')
        
        # Forecast visualization
        forecast_frame = ttk.LabelFrame(tab, text=" 7-Day Forecast ", padding=10)
        forecast_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.fig = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG_COLOR)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=forecast_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
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
        
        # Data preview
        preview_frame = ttk.LabelFrame(tab, text=" Data Preview ", padding=10)
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_text = tk.Text(preview_frame, height=15, width=80, 
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
        
        # Feature importance visualization
        viz_frame = ttk.LabelFrame(tab, text=" Feature Importance ", padding=10)
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.model_fig = Figure(figsize=(8, 4), dpi=100, facecolor=BG_COLOR)
        self.model_ax = self.model_fig.add_subplot(111)
        self.model_ax.set_facecolor(BG_COLOR)
        
        self.model_canvas = FigureCanvasTkAgg(self.model_fig, master=viz_frame)
        self.model_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        return tab
    
    def create_status_bar(self):
        """Create the status bar at bottom of window"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill='x', padx=10, pady=(0, 5))
        
        ttk.Label(status_bar, textvariable=self.status_var, 
                 relief=tk.SUNKEN, anchor=tk.W).pack(fill='x')
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def get_forecast(self):
        """Fetch and display weather forecast"""
        city = self.city_var.get().strip()
        if not city:
            messagebox.showwarning("Input Error", "Please enter a city name.")
            return
        
        self.update_status(f"Fetching weather for {city}...")
        
        try:
            # Get predictions from our model
            predictions = self.weather_model.predict_next_days(self.data)
            
            # Display current weather (using first prediction)
            current = predictions.iloc[0]
            weather_text = f"Location: {city}\n"
            weather_text += f"Date: {current['date'].strftime('%Y-%m-%d')}\n"
            weather_text += f"Temperature: {current['temperature']:.1f}°C\n"
            weather_text += f"Humidity: {current['humidity']:.1f}%\n"
            weather_text += f"Pressure: {current['pressure']:.1f} hPa\n"
            weather_text += f"Wind Speed: {current['wind_speed']:.1f} m/s"
            
            self.current_weather_text.config(state=tk.NORMAL)
            self.current_weather_text.delete(1.0, tk.END)
            self.current_weather_text.insert(tk.END, weather_text)
            self.current_weather_text.config(state=tk.DISABLED)
            
            # Plot forecast
            self.ax.clear()
            
            # Customize plot appearance
            self.ax.plot(predictions['date'], predictions['temperature'], 
                        color=PRIMARY_COLOR, marker='o', linestyle='-', 
                        linewidth=2, markersize=8, label='Temperature (°C)')
            
            # Add plot decorations
            self.ax.set_xlabel('Date', fontsize=10)
            self.ax.set_ylabel('Temperature (°C)', fontsize=10)
            self.ax.set_title(f'7-Day Temperature Forecast for {city}', 
                            fontsize=12, pad=20)
            
            # Format axes
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.tick_params(axis='both', which='major', labelsize=9)
            
            # Rotate date labels
            self.fig.autofmt_xdate()
            
            # Add legend
            self.ax.legend(loc='upper right', fontsize=9)
            
            # Redraw canvas
            self.canvas.draw()
            
            self.update_status(f"Weather forecast for {city} displayed")
            
        except Exception as e:
            messagebox.showerror("Forecast Error", 
                               f"Failed to generate forecast: {str(e)}")
            self.update_status("Forecast failed")
    
    def generate_data(self):
        """Generate new sample weather data"""
        self.update_status("Generating sample weather data...")
        
        try:
            self.data = self.weather_data.generate_sample_data()
            self.view_data()
            messagebox.showinfo("Success", "New sample data generated successfully!")
            self.update_status("Sample data generated")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate data: {str(e)}")
            self.update_status("Data generation failed")
    
    def load_data(self):
        """Load weather data from file"""
        filepath = filedialog.askopenfilename(
            title="Select Weather Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if not filepath:
            return
        
        self.update_status(f"Loading data from {os.path.basename(filepath)}...")
        
        try:
            self.data = pd.read_csv(filepath, parse_dates=['date'])
            self.view_data()
            messagebox.showinfo("Success", "Data loaded successfully!")
            self.update_status(f"Data loaded from {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.update_status("Data load failed")
    
    def save_data(self):
        """Save current weather data to file"""
        if self.data is None:
            messagebox.showwarning("No Data", "No data available to save.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Weather Data",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if not filepath:
            return
        
        self.update_status(f"Saving data to {os.path.basename(filepath)}...")
        
        try:
            self.data.to_csv(filepath, index=False)
            messagebox.showinfo("Success", "Data saved successfully!")
            self.update_status(f"Data saved to {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
            self.update_status("Data save failed")
    
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
    
    def train_model(self):
        """Train the weather prediction model"""
        if self.data is None:
            messagebox.showwarning("No Data", "Please generate or load data first.")
            return
        
        self.update_status("Training weather model...")
        
        try:
            rmse = self.weather_model.train(self.data)
            messagebox.showinfo("Success", 
                             f"Model trained successfully!\nRMSE: {rmse:.2f}°C")
            self.evaluate_model()
            self.update_status("Model training complete")
        except Exception as e:
            messagebox.showerror("Error", f"Model training failed: {str(e)}")
            self.update_status("Model training failed")
    
    def evaluate_model(self):
        """Evaluate and display model performance"""
        if not self.weather_model.model:
            messagebox.showwarning("No Model", "Please train the model first.")
            return
        
        self.update_status("Evaluating model...")
        
        try:
            # Display model information
            model_info = "Model Type: Random Forest Regressor\n"
            model_info += f"Number of Trees: {self.weather_model.model.n_estimators}\n"
            model_info += "Features Used:\n"
            model_info += " - Day of year\n - Month\n - Day\n"
            model_info += " - Humidity\n - Pressure\n - Wind speed\n"
            
            self.model_info_text.config(state=tk.NORMAL)
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(tk.END, model_info)
            self.model_info_text.config(state=tk.DISABLED)
            
            # Plot feature importance
            self.model_ax.clear()
            
            features = ['Day of Year', 'Month', 'Day', 'Humidity', 'Pressure', 'Wind Speed']
            importances = self.weather_model.model.feature_importances_
            
            # Sort features by importance
            sorted_idx = np.argsort(importances)
            features = [features[i] for i in sorted_idx]
            importances = [importances[i] for i in sorted_idx]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(features))
            bars = self.model_ax.barh(y_pos, importances, align='center', 
                                     color=PRIMARY_COLOR, alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                self.model_ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                                  f'{width:.2f}', va='center')
            
            # Customize chart appearance
            self.model_ax.set_yticks(y_pos)
            self.model_ax.set_yticklabels(features)
            self.model_ax.set_xlabel('Importance Score', fontsize=10)
            self.model_ax.set_title('Feature Importance', fontsize=12, pad=20)
            self.model_ax.grid(True, linestyle='--', alpha=0.3)
            
            self.model_canvas.draw()
            self.update_status("Model evaluation complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Model evaluation failed: {str(e)}")
            self.update_status("Model evaluation failed")
    
    def show_documentation(self):
        """Open documentation in web browser"""
        webbrowser.open("https://github.com/example/weather-forecast-app/docs")
    
    def show_about(self):
        """Show about dialog"""
        about_text = "Weather Forecast Pro\n\n"
        about_text += "Version 1.0\n"
        about_text += "© 2023 Weather Forecasting Inc.\n\n"
        about_text += "A machine learning application for weather prediction."
        
        messagebox.showinfo("About Weather Forecast Pro", about_text)

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    try:
        # Initialize and run application
        app = WeatherApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()