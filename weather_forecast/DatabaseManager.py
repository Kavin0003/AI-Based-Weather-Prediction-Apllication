import mysql.connector
from mysql.connector import Error
import pandas as pd
from datetime import datetime

class DatabaseManager:
    """Handles all database operations for the weather application"""
    
    def __init__(self, host="localhost", user="root", password="", database="weather_db"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.connect()
        self.initialize_database()

    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print("Connected to MySQL database")
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise

    def initialize_database(self):
        """Create database tables if they don't exist"""
        try:
            cursor = self.connection.cursor()
            
            # Create database if not exists
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            cursor.execute(f"USE {self.database}")
            
            # Create locations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS locations (
                    location_id INT AUTO_INCREMENT PRIMARY KEY,
                    city VARCHAR(100) NOT NULL,
                    country VARCHAR(100),
                    latitude DECIMAL(10,6),
                    longitude DECIMAL(10,6),
                    UNIQUE KEY (city)
                )
            """)
            
            # Create weather_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weather_data (
                    data_id INT AUTO_INCREMENT PRIMARY KEY,
                    location_id INT NOT NULL,
                    date DATE NOT NULL,
                    temperature DECIMAL(5,2),
                    humidity DECIMAL(5,2),
                    pressure DECIMAL(7,2),
                    wind_speed DECIMAL(5,2),
                    FOREIGN KEY (location_id) REFERENCES locations(location_id),
                    UNIQUE KEY (location_id, date)
                )
            """)
            
            # Create forecasts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id INT AUTO_INCREMENT PRIMARY KEY,
                    location_id INT NOT NULL,
                    forecast_date DATE NOT NULL,
                    prediction_date DATE NOT NULL,
                    temperature DECIMAL(5,2),
                    humidity DECIMAL(5,2),
                    pressure DECIMAL(7,2),
                    wind_speed DECIMAL(5,2),
                    FOREIGN KEY (location_id) REFERENCES locations(location_id),
                    UNIQUE KEY (location_id, forecast_date, prediction_date)
                )
            """)
            
            self.connection.commit()
            cursor.close()
            
        except Error as e:
            print(f"Error initializing database: {e}")
            raise

    def save_weather_data(self, city, weather_df):
        """Save weather data to database"""
        try:
            cursor = self.connection.cursor()
            
            # Insert or get location
            cursor.execute(
                "INSERT IGNORE INTO locations (city) VALUES (%s)",
                (city,)
            )
            cursor.execute(
                "SELECT location_id FROM locations WHERE city = %s",
                (city,)
            )
            location_id = cursor.fetchone()[0]
            
            # Insert weather data
            for _, row in weather_df.iterrows():
                cursor.execute("""
                    INSERT INTO weather_data (
                        location_id, date, temperature, 
                        humidity, pressure, wind_speed
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        temperature = VALUES(temperature),
                        humidity = VALUES(humidity),
                        pressure = VALUES(pressure),
                        wind_speed = VALUES(wind_speed)
                """, (
                    location_id,
                    row['date'].strftime('%Y-%m-%d'),
                    float(row['temperature']),
                    float(row['humidity']),
                    float(row['pressure']),
                    float(row['wind_speed'])
                ))
            
            self.connection.commit()
            cursor.close()
            return True
            
        except Error as e:
            print(f"Error saving weather data: {e}")
            return False

    def get_historical_data(self, city, days=30):
        """Retrieve historical weather data for a city"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Get location ID
            cursor.execute(
                "SELECT location_id FROM locations WHERE city = %s",
                (city,)
            )
            result = cursor.fetchone()
            if not result:
                return None
                
            location_id = result['location_id']
            
            # Get historical data
            cursor.execute("""
                SELECT date, temperature, humidity, pressure, wind_speed
                FROM weather_data
                WHERE location_id = %s
                ORDER BY date DESC
                LIMIT %s
            """, (location_id, days))
            
            data = cursor.fetchall()
            cursor.close()
            
            if not data:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            return df
            
        except Error as e:
            print(f"Error getting historical data: {e}")
            return None

    def save_forecast(self, city, forecast_df):
        """Save forecast data to database"""
        try:
            cursor = self.connection.cursor()
            
            # Get location ID
            cursor.execute(
                "SELECT location_id FROM locations WHERE city = %s",
                (city,)
            )
            result = cursor.fetchone()
            if not result:
                return False
                
            location_id = result[0]
            forecast_date = datetime.now().date()
            
            # Insert forecast data
            for _, row in forecast_df.iterrows():
                cursor.execute("""
                    INSERT INTO forecasts (
                        location_id, forecast_date, prediction_date,
                        temperature, humidity, pressure, wind_speed
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        temperature = VALUES(temperature),
                        humidity = VALUES(humidity),
                        pressure = VALUES(pressure),
                        wind_speed = VALUES(wind_speed)
                """, (
                    location_id,
                    forecast_date,
                    row['date'].strftime('%Y-%m-%d'),
                    float(row['temperature']),
                    float(row['humidity']),
                    float(row['pressure']),
                    float(row['wind_speed'])
                ))
            
            self.connection.commit()
            cursor.close()
            return True
            
        except Error as e:
            print(f"Error saving forecast: {e}")
            return False

    def get_forecast_history(self, city, days=7):
        """Retrieve forecast history for a city"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            
            # Get location ID
            cursor.execute(
                "SELECT location_id FROM locations WHERE city = %s",
                (city,)
            )
            result = cursor.fetchone()
            if not result:
                return None
                
            location_id = result['location_id']
            
            # Get forecast history
            cursor.execute("""
                SELECT prediction_date, temperature, humidity, 
                       pressure, wind_speed, forecast_date
                FROM forecasts
                WHERE location_id = %s
                ORDER BY forecast_date DESC, prediction_date ASC
                LIMIT %s
            """, (location_id, days))
            
            data = cursor.fetchall()
            cursor.close()
            
            if not data:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['prediction_date'] = pd.to_datetime(df['prediction_date'])
            df['forecast_date'] = pd.to_datetime(df['forecast_date'])
            return df
            
        except Error as e:
            print(f"Error getting forecast history: {e}")
            return None

    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed")