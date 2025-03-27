import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AviationDatabase:
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
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        
        # navaids
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
        
        # waypoints
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
        
        # procedures
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
        
        # procedure_waypoints
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
        
        # charts
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
        
        # chart_calibration
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
    
        return sqlite3.connect(self.db_path)
    
    def get_airports(self):
        conn = self.get_connection()
        df = pd.read_sql("SELECT * FROM airports", conn)
        conn.close()
        return df
    
    def get_airport(self, icao):
        conn = self.get_connection()
        df = pd.read_sql(f"SELECT * FROM airports WHERE icao = '{icao}'", conn)
        conn.close()
        return df
    
    def get_runways(self, airport_icao=None):
        conn = self.get_connection()
        query = "SELECT * FROM runways"
        if airport_icao:
            query += f" WHERE airport_icao = '{airport_icao}'"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def get_charts(self, airport_icao=None, chart_type=None):
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
        airport = self.get_airport(icao)
        if airport.empty:
            return pd.DataFrame()
        
        lat = airport.iloc[0]['latitude']
        lon = airport.iloc[0]['longitude']
        
        conn = self.get_connection()
        
        # 算一下距离
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
        R = 3440.065  # 海里
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def save_calibration(self, chart_id, transformation_matrix, reference_points, accuracy_score):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 把numpy数组转成string
        if isinstance(transformation_matrix, np.ndarray):
            transformation_matrix = transformation_matrix.tolist()
        
        import json
        from datetime import datetime
        
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
        
        # 标记一下calibrated
        cursor.execute(
            "UPDATE charts SET calibrated = 1 WHERE id = ?",
            (chart_id,)
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_calibration(self, chart_id):
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
