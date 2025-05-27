import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import os
import joblib
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class WeatherData:
    def __init__(self):
        self.api_key = "YOUR_API_KEY_HERE"  # Replace with actual API key
        
    def generate_sample_data(self, days=365):
        """Generate sample weather data for training with seasonal patterns"""
        dates = [datetime.datetime.now() - datetime.timedelta(days=x) for x in range(days, 0, -1)]
        
        temperature, humidity, pressure, wind_speed = [], [], [], []
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            season_factor = np.sin(2 * np.pi * day_of_year / 365)
            
            temp = 15 + 10 * season_factor + np.random.normal(0, 3)
            temperature.append(temp)
            
            hum = 70 - 20 * season_factor + np.random.normal(0, 10)
            humidity.append(max(min(hum, 100), 0))
            
            press = 1013 + 10 * np.sin(day_of_year / 20) + np.random.normal(0, 3)
            pressure.append(press)
            
            wind = 5 + 3 * abs(season_factor) + np.random.normal(0, 2)
            wind_speed.append(max(wind, 0))
        
        df = pd.DataFrame({
            'date': dates, 'temperature': temperature, 'humidity': humidity,
            'pressure': pressure, 'wind_speed': wind_speed
        })
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        return df

class WeatherModel:
    def __init__(self, model_path='weather_model.pkl'):
        self.model_path = model_path
        self.model = None if not os.path.exists(model_path) else joblib.load(model_path)
    
    def train(self, data):
        """Train weather forecasting model"""
        X = data[['day_of_year', 'month', 'day', 'humidity', 'pressure', 'wind_speed']]
        y = data['temperature']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        joblib.dump(self.model, self.model_path)
        return np.sqrt(((self.model.predict(X_test) - y_test) ** 2).mean())
    
    def predict_next_days(self, data, days=7):
        """Predict weather for next n days"""
        if not self.model:
            raise Exception("Model not trained. Please train the model first.")
        
        last_date = data['date'].max()
        predictions = []
        
        for i in range(1, days+1):
            next_date = last_date + datetime.timedelta(days=i)
            day_of_year = next_date.timetuple().tm_yday
            month = next_date.month
            day = next_date.day
            
            last_5_days = data.sort_values('date', ascending=False).head(5)
            avg_humidity = last_5_days['humidity'].mean()
            avg_pressure = last_5_days['pressure'].mean()
            avg_wind_speed = last_5_days['wind_speed'].mean()
            
            features = np.array([[day_of_year, month, day, avg_humidity, avg_pressure, avg_wind_speed]])
            pred_temp = self.model.predict(features)[0]
            
            predictions.append({
                'date': next_date, 'temperature': pred_temp, 'humidity': avg_humidity,
                'pressure': avg_pressure, 'wind_speed': avg_wind_speed
            })
        
        return pd.DataFrame(predictions)

class WeatherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Forecasting App")
        self.root.geometry("800x600")
        
        self.weather_data = WeatherData()
        self.weather_model = WeatherModel()
        self.data = self.weather_data.generate_sample_data()
        
        # Create tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create and configure tabs
        self.forecast_tab = ttk.Frame(self.notebook)
        self.data_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.forecast_tab, text="Weather Forecast")
        self.notebook.add(self.data_tab, text="Weather Data")
        self.notebook.add(self.model_tab, text="Model Training")
        
        self.setup_ui()
        
    def setup_ui(self):
        # Forecast Tab
        forecast_frame = ttk.Frame(self.forecast_tab)
        forecast_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(forecast_frame, text="City:").grid(row=0, column=0, padx=5, pady=5)
        self.city_var = tk.StringVar(value="London")
        ttk.Entry(forecast_frame, textvariable=self.city_var).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(forecast_frame, text="Get Forecast", command=self.get_forecast).grid(row=0, column=2, padx=5, pady=5)
        
        self.forecast_fig = Figure(figsize=(7, 4))
        self.forecast_ax = self.forecast_fig.add_subplot(111)
        self.forecast_canvas = FigureCanvasTkAgg(self.forecast_fig, master=forecast_frame)
        self.forecast_canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, padx=5, pady=5)
        
        # Model Tab
        model_frame = ttk.Frame(self.model_tab)
        model_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Button(model_frame, text="Train Model", command=self.train_model).grid(row=0, column=0, padx=5, pady=5)
        self.model_info = tk.Text(model_frame, height=5, width=60)
        self.model_info.grid(row=1, column=0, padx=5, pady=5)
    
    def get_forecast(self):
        city = self.city_var.get()
        if not city:
            messagebox.showerror("Error", "Please enter a city name.")
            return
        
        try:
            predictions = self.weather_model.predict_next_days(self.data)
            
            self.forecast_ax.clear()
            self.forecast_ax.plot(predictions['date'], predictions['temperature'], 'r-o', label='Temperature (°C)')
            self.forecast_ax.set_xlabel('Date')
            self.forecast_ax.set_ylabel('Temperature (°C)')
            self.forecast_ax.set_title(f'7-Day Temperature Forecast for {city}')
            self.forecast_ax.grid(True)
            self.forecast_fig.autofmt_xdate()
            self.forecast_ax.legend()
            self.forecast_canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get forecast: {str(e)}")
    
    def train_model(self):
        try:
            rmse = self.weather_model.train(self.data)
            self.model_info.delete(1.0, tk.END)
            self.model_info.insert(tk.END, f"Model trained successfully!\nRMSE: {rmse:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")

def main():
    root = tk.Tk()
    app = WeatherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()