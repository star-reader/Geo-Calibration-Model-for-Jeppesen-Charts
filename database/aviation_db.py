import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AviationDatabase:
    """Handler for aviation database operations."""
    
    def __init__(self, db_path='database/aviation.db'):
        self.db_path = Path(db_path)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database if it doesn't exist."""
        if not self.db_path.exists():
            logger.info(f"Creating new aviation database at {self.db_path}")
            self._create_database()
        else:
            logger.info(f"Using existing aviation database at {self.db_path}")
    
    def _create_database(self):
        """Create the database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create airports table
        cursor.execute('''
        CREATE TABLE airports (
            icao TEXT PRIMARY KEY,
            iata TEXT,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            elevation_ft INTEGER,
            city TEXT,
            country TEXT,
            region TEXT
        )
        ''')
        
        # Create runways table
        cursor.execute('''
        CREATE TABLE runways (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            airport_icao TEXT NOT NULL,
            designation TEXT NOT NULL,
            length_ft INTEGER,
            width_ft INTEGER,
            surface TEXT,
            lighted INTEGER,
            closed INTEGER,
            latitude_start REAL NOT NULL,
            longitude_start REAL NOT NULL,
            latitude_end REAL NOT NULL,
            longitude_end REAL NOT NULL,
            elevation_ft INTEGER,
            FOREIGN KEY (airport_icao) REFERENCES airports(icao)
        )
        ''')
        
        # Create navaids table
        cursor.execute('''
        CREATE TABLE navaids (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ident TEXT NOT NULL,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            frequency REAL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            elevation_ft INTEGER,
            airport_icao TEXT,
            FOREIGN KEY (airport_icao) REFERENCES airports(icao)
        )
        ''')
        
        # Create waypoints table
        cursor.execute('''
        CREATE TABLE waypoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ident TEXT NOT NULL,
            type TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            region TEXT
        )
        ''')
        
        # Create procedures table
        cursor.execute('''
        CREATE TABLE procedures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            airport_icao TEXT NOT NULL,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            runway TEXT,
            FOREIGN KEY (airport_icao) REFERENCES airports(icao)
        )
        ''')
        
        # Create procedure_waypoints table
        cursor.execute('''
        CREATE TABLE procedure_waypoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            procedure_id INTEGER NOT NULL,
            sequence INTEGER NOT NULL,
            waypoint_id INTEGER,
            navaid_id INTEGER,
            latitude REAL,
            longitude REAL,
            altitude_restriction TEXT,
            speed_restriction TEXT,
            fix_type TEXT,
            turn_direction TEXT,
            FOREIGN KEY (procedure_id) REFERENCES procedures(id),
            FOREIGN KEY (waypoint_id) REFERENCES waypoints(id),
            FOREIGN KEY (navaid_id) REFERENCES navaids(id)
        )
        ''')
        
        # Create charts table
        cursor.execute('''
        CREATE TABLE charts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            airport_icao TEXT NOT NULL,
            chart_type TEXT NOT NULL,
            chart_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            calibrated INTEGER DEFAULT 0,
            FOREIGN KEY (airport_icao) REFERENCES airports(icao)
        )
        ''')
        
        # Create chart_calibration table
        cursor.execute('''
        CREATE TABLE chart_calibration (
            chart_id INTEGER PRIMARY KEY,
            transformation_matrix TEXT,
            reference_points TEXT,
            accuracy_score REAL,
            calibration_date TEXT,
            FOREIGN KEY (chart_id) REFERENCES charts(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        
        # Add sample data
        self._add_sample_data()
    
    def _add_sample_data(self):
        """Add sample data to the database for testing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sample airports
        airports = [
            ('KSFO', 'SFO', 'San Francisco International Airport', 37.6188, -122.3750, 13, 'San Francisco', 'United States', 'NA'),
            ('KJFK', 'JFK', 'John F Kennedy International Airport', 40.6399, -73.7787, 13, 'New York', 'United States', 'NA'),
            ('EGLL', 'LHR', 'London Heathrow Airport', 51.4775, -0.4614, 83, 'London', 'United Kingdom', 'EU'),
            ('RJTT', 'HND', 'Tokyo Haneda International Airport', 35.5533, 139.7810, 35, 'Tokyo', 'Japan', 'AS'),
            ('ZBAA', 'PEK', 'Beijing Capital International Airport', 40.0799, 116.6031, 116, 'Beijing', 'China', 'AS')
        ]
        
        cursor.executemany('INSERT INTO airports VALUES (?,?,?,?,?,?,?,?,?)', airports)
        
        # Sample runways
        runways = [
            (None, 'KSFO', '01L/19R', 11870, 200, 'ASPH-CONC', 1, 0, 37.6188, -122.3750, 37.6388, -122.3750, 13),
            (None, 'KSFO', '01R/19L', 11381, 200, 'ASPH-CONC', 1, 0, 37.6188, -122.3770, 37.6388, -122.3770, 13),
            (None, 'KJFK', '04L/22R', 12079, 200, 'ASPH-CONC', 1, 0, 40.6399, -73.7787, 40.6599, -73.7587, 13),
            (None, 'EGLL', '09L/27R', 12799, 164, 'ASPH-CONC', 1, 0, 51.4775, -0.4614, 51.4775, -0.4414, 83),
            (None, 'RJTT', '16L/34R', 11024, 196, 'ASPH-CONC', 1, 0, 35.5533, 139.7810, 35.5733, 139.7810, 35)
        ]
        
        cursor.executemany('INSERT INTO runways VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', runways)
        
        # Sample navaids
        navaids = [
            (None, 'SFO', 'SAN FRANCISCO VOR/DME', 'VOR/DME', 116.8, 37.6188, -122.3750, 13, 'KSFO'),
            (None, 'JFK', 'KENNEDY VOR/DME', 'VOR/DME', 115.9, 40.6399, -73.7787, 13, 'KJFK'),
            (None, 'LHR', 'LONDON VOR/DME', 'VOR/DME', 113.6, 51.4775, -0.4614, 83, 'EGLL'),
            (None, 'HND', 'TOKYO VOR/DME', 'VOR/DME', 112.2, 35.5533, 139.7810, 35, 'RJTT'),
            (None, 'PEK', 'BEIJING VOR/DME', 'VOR/DME', 116.4, 40.0799, 116.6031, 116, 'ZBAA')
        ]
        
        cursor.executemany('INSERT INTO navaids VALUES (?,?,?,?,?,?,?,?,?)', navaids)
        
        # Sample waypoints
        waypoints = [
            (None, 'DUMBA', 'RNAV', 37.7000, -122.5000, 'NA'),
            (None, 'FIMLA', 'RNAV', 40.7000, -73.9000, 'NA'),
            (None, 'GEGMU', 'RNAV', 51.5000, -0.5000, 'EU'),
            (None, 'AROSA', 'RNAV', 35.6000, 139.8000, 'AS'),
            (None, 'BOBAK', 'RNAV', 40.1000, 116.7000, 'AS')
        ]
        
        cursor.executemany('INSERT INTO waypoints VALUES (?,?,?,?,?,?)', waypoints)
        
        # Sample charts
        charts = [
            (None, 'KSFO', 'SID', 'OFFSHORE ONE DEPARTURE', 'charts/KSFO_SID_OFFSHORE1.png', 0),
            (None, 'KSFO', 'STAR', 'DYAMD TWO ARRIVAL', 'charts/KSFO_STAR_DYAMD2.png', 0),
            (None, 'KSFO', 'IAP', 'ILS OR LOC RWY 28L', 'charts/KSFO_IAP_ILS28L.png', 0),
            (None, 'KSFO', 'APD', 'AIRPORT DIAGRAM', 'charts/KSFO_APD.png', 0),
            (None, 'KJFK', 'APD', 'AIRPORT DIAGRAM', 'charts/KJFK_APD.png', 0)
        ]
        
        cursor.executemany('INSERT INTO charts VALUES (?,?,?,?,?,?)', charts)
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Get a connection to the database."""
        return sqlite3.connect(self.db_path)
    
    def get_airports(self):
        """Get all airports."""
        conn = self.get_connection()
        df = pd.read_sql("SELECT * FROM airports", conn)
        conn.close()
        return df
    
    def get_airport(self, icao):
        """Get airport by ICAO code."""
        conn = self.get_connection()
        df = pd.read_sql(f"SELECT * FROM airports WHERE icao = '{icao}'", conn)
        conn.close()
        return df
    
    def get_runways(self, airport_icao=None):
        """Get runways, optionally filtered by airport."""
        conn = self.get_connection()
        query = "SELECT * FROM runways"
        if airport_icao:
            query += f" WHERE airport_icao = '{airport_icao}'"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def get_charts(self, airport_icao=None, chart_type=None):
        """Get charts, optionally filtered by airport and type."""
        conn = self.get_connection()
        query = "SELECT * FROM charts"
        filters = []
        
        if airport_icao:
            filters.append(f"airport_icao = '{airport_icao}'")
        
        if chart_type:
            filters.append(f"chart_type = '{chart_type}'")
        
        if filters:
            query += " WHERE " + " AND ".join(filters)
        
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def get_navaids_near_airport(self, icao, radius_nm=50):
        """Get navaids near an airport within a specified radius."""
        airport = self.get_airport(icao)
        if airport.empty:
            return pd.DataFrame()
        
        lat = airport.iloc[0]['latitude']
        lon = airport.iloc[0]['longitude']
        
        conn = self.get_connection()
        
        # Approximate 1 degree latitude = 60 nautical miles
        # Approximate 1 degree longitude = 60 * cos(latitude) nautical miles
        lat_delta = radius_nm / 60.0
        lon_delta = radius_nm / (60.0 * np.cos(np.radians(lat)))
        
        query = f"""
        SELECT * FROM navaids
        WHERE latitude BETWEEN {lat - lat_delta} AND {lat + lat_delta}
        AND longitude BETWEEN {lon - lon_delta} AND {lon + lon_delta}
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Calculate actual distances and filter
        if not df.empty:
            df['distance_nm'] = df.apply(
                lambda row: self._haversine(lat, lon, row['latitude'], row['longitude']),
                axis=1
            )
            df = df[df['distance_nm'] <= radius_nm]
        
        return df
    
    def _haversine(self, lat1, lon1, lat2, lon2):
        """Calculate distance between points in nautical miles."""
        R = 3440.065  # Earth radius in nautical miles
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def save_calibration(self, chart_id, transformation_matrix, reference_points, accuracy_score):
        """Save chart calibration data."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Convert numpy arrays to string
        if isinstance(transformation_matrix, np.ndarray):
            transformation_matrix = transformation_matrix.tolist()
        
        import json
        from datetime import datetime
        
        # Update calibration data
        cursor.execute(
            '''
            INSERT OR REPLACE INTO chart_calibration
            (chart_id, transformation_matrix, reference_points, accuracy_score, calibration_date)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (
                chart_id,
                json.dumps(transformation_matrix),
                json.dumps(reference_points),
                float(accuracy_score),
                datetime.now().isoformat()
            )
        )
        
        # Mark chart as calibrated
        cursor.execute(
            "UPDATE charts SET calibrated = 1 WHERE id = ?",
            (chart_id,)
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_calibration(self, chart_id):
        """Get calibration data for a chart."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM chart_calibration WHERE chart_id = ?",
            (chart_id,)
        )
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        import json
        
        return {
            'chart_id': row[0],
            'transformation_matrix': json.loads(row[1]),
            'reference_points': json.loads(row[2]),
            'accuracy_score': row[3],
            'calibration_date': row[4]
        }
