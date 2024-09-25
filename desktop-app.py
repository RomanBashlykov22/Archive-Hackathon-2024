import tkinter as tk
from tkinter import filedialog
import PIL
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageTk, ImageFile


def restore_scanned_document(image: PIL.ImageFile):
    gray_image = image.convert("L")

    enhancer = ImageEnhance.Contrast(gray_image)
    contrast_image = enhancer.enhance(2.0)
    open_cv_image = np.array(contrast_image)

    adaptive_thresh = cv2.adaptiveThreshold(
        open_cv_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(adaptive_thresh, -1, kernel)
    restored_image = Image.fromarray(sharpened)
    
    return restored_image


class ImageRestorationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Реставрация документов")

        self.canvas_width = 400
        self.canvas_height = 300

        self.canvas_original = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas_restored = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas_original.grid(row=0, column=0, padx=10, pady=10)
        self.canvas_restored.grid(row=0, column=1, padx=10, pady=10)

        self.button_choose = tk.Button(root, text="Выбрать изображение", command=self.choose_image)
        self.button_restore = tk.Button(root, text="Реставрировать изображение", command=self.restore_image)
        self.button_choose.grid(row=1, column=0, pady=10)
        self.button_restore.grid(row=1, column=1, pady=10)

        self.original_image = None
        self.restored_image = None 

        self.zoom_factor = 3

        self.canvas_original.bind("<Motion>", self.zoom_original)
        self.canvas_restored.bind("<Motion>", self.zoom_restored)

    def choose_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if filepath:
            self.original_image = Image.open(filepath)
            self.display_image(self.canvas_original, self.original_image)

    def restore_image(self):
        if self.original_image:
            self.restored_image = restore_scanned_document(self.original_image)
            self.display_image(self.canvas_restored, self.restored_image)

    def display_image(self, canvas, image):
        resized_image = self.get_preview_image(image)
        image_tk = ImageTk.PhotoImage(resized_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk

    def get_preview_image(self, image):
        img_width, img_height = image.size
        scale = min(self.canvas_width / img_width, self.canvas_height / img_height)

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image

    def zoom_original(self, event):
        if self.original_image:
            self.zoom_image(self.canvas_original, self.original_image, event)

    def zoom_restored(self, event):
        if self.restored_image:
            self.zoom_image(self.canvas_restored, self.restored_image, event)

    def zoom_image(self, canvas, image, event):
        x, y = event.x, event.y

        img_width, img_height = image.size
        canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()

        scale_x = img_width / canvas_width
        scale_y = img_height / canvas_height

        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        zoom_size = 50

        left = max(0, original_x - zoom_size // 2)
        top = max(0, original_y - zoom_size // 2)
        right = min(image.width, original_x + zoom_size // 2)
        bottom = min(image.height, original_y + zoom_size // 2)

        cropped_image = image.crop((left, top, right, bottom))
        zoomed_image = cropped_image.resize(
            (cropped_image.width * self.zoom_factor, cropped_image.height * self.zoom_factor),
            Image.LANCZOS
        )

        zoomed_image_tk = ImageTk.PhotoImage(zoomed_image)

        if hasattr(self, 'zoom_window'):
            canvas.delete(self.zoom_window)

        self.zoom_window = canvas.create_image(
            x + zoom_size, y + zoom_size, anchor=tk.NW, image=zoomed_image_tk
        )


root = tk.Tk()
root.geometry("850x450")

app = ImageRestorationApp(root)
root.mainloop()
