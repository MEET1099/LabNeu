    self.spin_kernel_size.setValue(3)
            self.checkbox_erosion.setChecked(False)
            self.checkbox_dilation.setChecked(False)
            self.checkbox_opening.setChecked(False)
            self.checkbox_closing.setChecked(False)
            self.spin_morphology_size.setValue(2)

            # Si se cerró la ventana de imagen final, se restablece la imagen original
            if result == QDialog.Rejected:
                self.show_image(self.orig_image)