"""
Database Module for Audio Analyzer LUFS Application

Manages persistent storage of analysis results using SQLite.
Stores metadata about audio files, LUFS measurements, and analysis history.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class AnalysisDatabase:
    """Manages database operations for audio analysis results."""

    def __init__(self, db_path: str = 'analysis_history.db'):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection = None
        self.init_db()

    def init_db(self) -> None:
        """Initialize database with required tables."""
        self.connection = sqlite3.connect(str(self.db_path))
        cursor = self.connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                duration REAL,
                integrated_lufs REAL,
                short_term_lufs REAL,
                momentary_lufs REAL,
                true_peak REAL
            )
        ''')
        self.connection.commit()

    def save_analysis(self, filename: str, duration: float, integrated_lufs: float,
                     short_term_lufs: float, momentary_lufs: float,
                     true_peak: float) -> int:
        """Save analysis results to database.
        
        Args:
            filename: Name of analyzed audio file
            duration: Duration of audio in seconds
            integrated_lufs: Integrated LUFS measurement
            short_term_lufs: Short-term LUFS measurement
            momentary_lufs: Momentary LUFS measurement
            true_peak: True Peak measurement
            
        Returns:
            ID of inserted record
        """
        cursor = self.connection.cursor()
        cursor.execute('''
            INSERT INTO analyses
            (filename, duration, integrated_lufs, short_term_lufs,
             momentary_lufs, true_peak)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, duration, integrated_lufs, short_term_lufs,
              momentary_lufs, true_peak))
        self.connection.commit()
        return cursor.lastrowid

    def get_analysis_history(self, limit: int = 100) -> List[Dict]:
        """Retrieve analysis history from database.
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of analysis records
        """
        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?',
                      (limit,))
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
