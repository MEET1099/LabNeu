import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox, QGraphicsDropShadowEffect, QHBoxLayout, QDialog, QCheckBox, QDialogButtonBox, QSpinBox, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QIcon, QColor, QCursor
from PyQt5.QtCore import Qt, QSize, QRect

import cv2
import pickle
import numpy as np

class CustomDialog(QDialog):
    def __init__(self, title, image):
        super().__init__()

        # Configurar la ventana emergente
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 500, 500)

        # Mostrar la imagen en la ventana emergente
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 500, 500)
        self.show_image(image)

        # Conectar la señal accepted a la función que libera la imagen
        self.accepted.connect(self.release_image)

    def release_image(self):
        # Liberar la imagen cuando se cierra la ventana emergente
        self.label.clear()

    def show_image(self, image):
        if image is not None:
            if len(image.shape) == 2:  # Imagen en escala de grises
                qImg = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)
            else:  # Imagen en color
                qImg = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qImg)

            # Redimensionar la imagen solo para mostrarla en la interfaz gráfica
            scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            self.label.setScaledContents(True)
            self.update()
class ImageProcessingDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Configuración de Procesamiento de Imágenes")
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Tamaño del kernel laplaciano
        self.label_kernel_size = QLabel("Tamaño del Kernel Laplaciano:")
        self.spin_kernel_size = QSpinBox()
        self.spin_kernel_size.setMinimum(1)
        self.spin_kernel_size.setValue(3)
        self.spin_kernel_size.setMaximum(255)  

        layout.addWidget(self.label_kernel_size)
        layout.addWidget(self.spin_kernel_size)

        # Opciones de operaciones morfológicas
        self.label_morphology = QLabel("Operaciones Morfológicas:")
        self.checkbox_erosion = QCheckBox("Erosión")
        self.checkbox_dilation = QCheckBox("Dilatación")
        self.checkbox_opening = QCheckBox("Apertura")
        self.checkbox_closing = QCheckBox("Cierre")

        layout.addWidget(self.label_morphology)
        layout.addWidget(self.checkbox_erosion)
        layout.addWidget(self.checkbox_dilation)
        layout.addWidget(self.checkbox_opening)
        layout.addWidget(self.checkbox_closing)

        # Tamaños para las operaciones morfológicas
        self.label_morphology_size = QLabel("Tamaño para Operaciones Morfológicas:")
        self.spin_morphology_size = QSpinBox()
        self.spin_morphology_size.setMinimum(1)
        self.spin_morphology_size.setValue(2)
        self.spin_morphology_size.setMaximum(15)

        layout.addWidget(self.label_morphology_size)
        layout.addWidget(self.spin_morphology_size)

        # Botones Aceptar y Cancelar
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_parameters(self):
        return {
            "kernel_size": self.spin_kernel_size.value(),
            "erosion": self.checkbox_erosion.isChecked(),
            "dilation": self.checkbox_dilation.isChecked(),
            "opening": self.checkbox_opening.isChecked(),
            "closing": self.checkbox_closing.isChecked(),
            "morphology_size": self.spin_morphology_size.value()
        }
    
class NeuronImageAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()

        # Configurar ventana
        self.setWindowTitle("Neuron Image Analyzer")
        self.setGeometry(100, 100, 250, 500)

        # Fondo oscuro
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(79, 79, 84))  # Cambio de color de fondo
        self.setPalette(p)

        # Botones de opciones del menú
        self.menu_buttons = QWidget(self)
        menu_layout = QVBoxLayout()

        button_info = [
            ("Abrir Imagen", 'app_icons/icons8-image-50.png', self.open_image),
            ("Procesar", 'app_icons/icons8-process-50.png', self.process_image),
            ("Contar", 'app_icons/icons8-count-50.png', self.count_elements),
            ("Entrenar Modelo", 'app_icons/icons8-gym-50.png', self.train_model),
            ("Cargar Modelo", 'app_icons/icons8-load-balancer-50.png', self.load_model),
            ("Guardar Modelo", 'app_icons/icons8-artificial-intelligence-50.png', self.save_model),
            ("Predecir", 'app_icons/icons8-predict-50.png', self.predict),
            ("Guardar Imagen", 'app_icons/icons8-save-50.png', self.save_image)
        ]

        button_width = 250  # Ancho de los botones
        button_height = 70  # Alto de los botones

        for text, icon_path, callback in button_info:
            button = QPushButton()
            button.setText(text)
            button.setIcon(QIcon(icon_path))
            icon_size = 100  # Tamaño de los iconos
            button.setIconSize(QSize(icon_size, icon_size))
            button.setCursor(QCursor(Qt.PointingHandCursor))
            button.setStyleSheet('''
            QPushButton {
                color: white;
                background-color: transparent;
                padding: 5px 10px 5px 10px;
                font-size: 20px;
                text-align: left;
                width: 10px;
            }
            QPushButton:hover {
                border: 3px solid orange;
                border-radius: 5px;
            }
            ''')
            button.clicked.connect(callback)
            menu_layout.addWidget(button)
            button.setMaximumSize(button_width, button_height)

        # Checkbox para el paso a paso
        self.step_by_step_checkbox = QCheckBox("Paso a paso")
        self.step_by_step_checkbox.setChecked(False)
        self.step_by_step_checkbox.setStyleSheet('''
        QCheckBox {
                color: white;
                background-color: transparent;
                padding: 5x 15px 5px 10px;
                font-size: 20px;
                text-align: left;
                width: 10px;
            }
            QPushButton:hover {
                border: 3px solid orange;
                border-radius: 5px;
            }
            ''')
                                                 
        menu_layout.addWidget(self.step_by_step_checkbox)

        # Agregar el atributo label
        self.label = QLabel(self)

        # Coloca las imágenes a la derecha de los botones
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addLayout(menu_layout)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setLayout(horizontal_layout)

        # Imágenes original y procesada
        self.orig_image = None
        self.proc_image = None

        # Configuración de sombra
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.setGraphicsEffect(shadow)

    # Funciones principales

    def open_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Option.ReadOnly
        fname, _ = QFileDialog.getOpenFileName(self, 'Open image', '', 'Image files (*.jpg *.png *.tif);;All Files (*)', options=options)
        if fname:
            self.orig_image = cv2.imread(fname)
            dialog = CustomDialog("Imagen Original", self.orig_image.copy())
            dialog.exec_()

            # Reiniciar los contornos de la imagen
            self.image_contours = None

            if self.proc_image is not None:
                self.show_image(self.orig_image)

    def process_image(self):
        if self.orig_image is None:
            print("Abre una imagen primero")
            return
        
        # Crear y mostrar la ventana de diálogo
        dialog = ImageProcessingDialog()
        result = dialog.exec_()

        # Obtener los parámetros seleccionados
        if result == QDialog.Accepted:
            params = dialog.get_parameters()

            # Accede a los parámetros que necesitas
            kernel_size = params["kernel_size"]
            use_erosion = params["erosion"]
            use_dilation = params["dilation"]
            use_opening = params["opening"]
            use_closing = params["closing"]
            morphology_size = params["morphology_size"]

        if self.step_by_step_checkbox.isChecked():
            # Mostrar paso a paso en una ventana emergente personalizada
            dialog = CustomDialog("Imagen Original", self.orig_image.copy())
            dialog.exec_()

            # Procesar imagen

            # Conversion a escala de grises
            gray = self.grayscale(self.orig_image.copy())
            dialog = CustomDialog("Escala de grises", gray)
            dialog.exec_()

            # Aplicar mejora de brillo adaptativa (CLAHE)
            enhanced_image = self.adaptive_brightness(gray)
            dialog = CustomDialog("Brillo Adaptativo", enhanced_image)
            dialog.exec_()

            # Normalizar la imagen para escalar los valores a un rango visualizable
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            dialog = CustomDialog("Escala de grises normalizada", gray)
            dialog.exec_()

            # Aplicacion de kernel laplaciano
            laplacian = self.apply_kernel(gray, self.laplacian_kernel(kernel_size if kernel_size % 2 == 1 else kernel_size + 1))
            dialog = CustomDialog("Kernel Laplaciano", laplacian)
            dialog.exec_()

            # Umbralizacion
            threshold = self.threshold(laplacian)
            dialog = CustomDialog("Umbralizacion", threshold)
            dialog.exec_()

            # Aplicar otras operaciones morfológicas según la selección del usuario
            if use_erosion:
                morph = self.erode(threshold, morphology_size)

            if use_dilation:
                morph = self.dilate(threshold, morphology_size)

            if use_opening:
                morph = self.opening(threshold, morphology_size)

            if use_closing:
                morph = self.closing(threshold, morphology_size)

            dialog = CustomDialog("Operaciones Morfologicas", morph)
            dialog.exec_()

            # Encontrar los bordes
            self.image_contours = self.contours(morph.copy())

            # Eliminar los bordes más pequeños y el más grande
            self.image_contours = self.filter_contours(self.image_contours)

            # imagen final
            self.proc_image = self.draw_contours(self.orig_image.copy(), self.image_contours)
            dialog = CustomDialog("Imagen Original", self.orig_image.copy())
            dialog = CustomDialog("Imagen Final", self.proc_image)
            dialog.exec_()
            
        else:
            # Procesar imagen

            # Conversion a escala de grises
            gray = self.grayscale(self.orig_image.copy())

            # Aplicar mejora de brillo adaptativa (CLAHE)
            enhanced_image = self.adaptive_brightness(gray)

            # Normalizar la imagen para escalar los valores a un rango visualizable
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            # Aplicacion de kernel laplaciano
            laplacian = self.apply_kernel(gray, self.laplacian_kernel(kernel_size if kernel_size % 2 == 1 else kernel_size + 1))

            # Umbralizacion
            threshold = self.threshold(laplacian)

            # Aplicar otras operaciones morfológicas según la selección del usuario
            if use_erosion:
                morph = self.erode(threshold, morphology_size)

            if use_dilation:
                morph = self.dilate(threshold, morphology_size)

            if use_opening:
                morph = self.opening(threshold, morphology_size)

            if use_closing:
                morph = self.closing(threshold, morphology_size)

            # Encontrar los bordes
            self.image_contours = self.contours(morph.copy())

            # Eliminar los bordes más pequeños y el más grande
            self.image_contours = self.filter_contours(self.image_contours)

            # Dibujar los bordes
            self.proc_image = self.draw_contours(self.orig_image.copy(), self.image_contours)

            # Mostrar la imagen procesada
            dialog = CustomDialog("Imagen Original", self.orig_image.copy())
            dialog = CustomDialog("Imagen Final", self.proc_image)
            dialog.exec_()


    def count_elements(self):
        if self.proc_image is None:
            print("Procesa la imagen primero")
            return

        if self.image_contours is None:
            print("No hay contornos disponibles.")
            return

        # Contar neuronas
        neuron_count = self.count(self.image_contours)

        # Mostrar el número de células en una ventana emergente
        message_box = QMessageBox()
        message_box.setWindowTitle("Número de Células")
        message_box.setText(f"Se encontraron {neuron_count} células.")
        message_box.setIcon(QMessageBox.Information)
        message_box.exec_()


    def load_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Option.ReadOnly
        fname, _ = QFileDialog.getOpenFileName(self, 'Load ML model', '', 'Pickle files (*.pkl);;All Files (*)', options=options)
        if fname:
            with open(fname, 'rb') as f:
                self.ml_model = pickle.load(f)

    def train_model(self):
        # Entrenar modelo (reemplaza esto con tu lógica de entrenamiento)
        print("Entrenando modelo...")

    def save_model(self):
        options = QFileDialog.Options()
        options |= QFileDialog.Option.ReadOnly
        fname, _ = QFileDialog.getSaveFileName(self, 'Save ML model', '', 'Pickle files (*.pkl);;All Files (*)', options=options)
        if fname:
            with open(fname, 'wb') as f:
                pickle.dump(self.ml_model, f)

    def predict(self):
        if self.proc_image is None:
            print("Procesa la imagen primero")
            return

        if self.ml_model is None:
            print("Carga un modelo primero")
            return

        # Predecir 
        print("Prediciendo...")

    def save_image(self):
        if self.proc_image is None:
            print("Procesa la imagen primero")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.Option.ReadOnly
        fname, _ = QFileDialog.getSaveFileName(self, 'Save image', '', 'Image files (*.jpg *.png);;All Files (*)', options=options)
        if fname:
            cv2.imwrite(fname, self.proc_image)

    # Utilidades
    def show_image(self, image):
        if image is not None:
            if len(image.shape) == 2:  # Imagen en escala de grises
                qImg = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8)
            else:  # Imagen en color
                qImg = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qImg)

            # Redimensionar la imagen solo para mostrarla en la interfaz gráfica
            scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            self.label.setScaledContents(True)
            self.update()

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def adaptive_brightness(self, image):
        # Configurar el objeto CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Aplicar el CLAHE a la imagen
        enhanced_image = clahe.apply(image)

        return enhanced_image
    
    def laplacian_kernel(self, size):
        # Crear un kernel laplaciano
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2  # Posición central del kernel
        sigma = 1.0  # Desviación estándar para el cálculo del kernel

        if sigma == 0:
            return kernel  # Evitar división por cero

        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = -(1 - (x**2 + y**2) / (2 * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

        # Normalizar el kernel para que la suma sea cero, evitando la división por cero
        kernel_sum = np.sum(np.abs(kernel))
        if kernel_sum != 0:
            kernel /= kernel_sum

        return kernel

    def apply_kernel(self, image, kernel):
        # Normalizar el kernel antes de aplicarlo
        kernel = kernel.astype(np.float32)
        
        # Aplicar el kernel laplaciano a la imagen
        laplacian_image = cv2.filter2D(image, cv2.CV_32F, kernel)

        # Asegurarse de que no hay valores NaN o infinitos
        laplacian_image = np.nan_to_num(laplacian_image)

        return laplacian_image.astype(np.uint8)

    def threshold(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    def kernel(self, size):
        return np.ones((size, size), np.uint8)
    
    def erode(self, image, size):
        return cv2.erode(image, self.kernel(size), iterations=2)
    
    def dilate(self, image, size):
        return cv2.dilate(image, self.kernel(size), iterations=2)
    
    def opening(self, image, size):
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, self.kernel(size))
    
    def closing(self, image, size):
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, self.kernel(size))
    
    def contours(self, image):
        # Encuentra los contornos en la imagen
        contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours


    def filter_contours(self, contours):
        if not contours:
            return []

        # Ordenar los contornos de mayor a menor área
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Excluir el contorno más grande
        filtered_contours = sorted_contours[1:]

        return filtered_contours

    def draw_contours(self, image, contours):
        return cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    def count(self, contours):
        return len(contours)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Establece la ruta al directorio donde se encuentra el script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    window = NeuronImageAnalyzer()
    window.show()
    sys.exit(app.exec())
