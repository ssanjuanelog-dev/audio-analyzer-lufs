#!/usr/bin/env python3
"""
Audio Analyzer - Espectro Completo y LUFS (ITU-R BS.1770-4)
Aplicación profesional de análisis de audio en tiempo real

Autor: Desarrollo Audio Profesional
Licencia: MIT
Versión: 1.0.0
"""

import sys
from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow

def main():
    """
    Punto de entrada principal de la aplicación
    """
    app = QApplication(sys.argv)
    
    # Configurar información de la aplicación
    app.setApplicationName("Audio Analyzer LUFS")
    app.setApplicationVersion("1.0.0")
    
    # Crear ventana principal
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
