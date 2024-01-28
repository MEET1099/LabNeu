#!/bin/bash

# Verificar si está en Windows o Mac
if [ "$OSTYPE" == "msys" ] || [ "$OSTYPE" == "win32" ]; then
  PIP=pip
  PYTHON=python
elif [ "$OSTYPE" == "darwin"* ]; then
  PIP=pip3
  PYTHON=python3  
fi

# Instalar librerías
$PIP install numpy
$PIP install opencv-python
$PIP install pyqt5
$PIP install pyqt5-sip

echo "Librerías instaladas con éxito"

if [ "$OSTYPE" == "msys" ] || [ "$OSTYPE" == "win32" ]; then
  $PYTHON requirements.py
elif [ "$OSTYPE" == "darwin"* ]; then
  $PYTHON3 requirements.py  
fi

echo "Dependencias de Python instaladas"
echo "Listo para ejecutar el código"
