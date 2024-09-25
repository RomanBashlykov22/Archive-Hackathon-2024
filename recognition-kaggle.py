import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

# Загрузка изображения
img = cv2.imread('/kaggle/input/images/111.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Удаление шума
denoised = cv2.medianBlur(gray, 3)

# Увеличение резкости
sharpened = cv2.filter2D(denoised, -1, kernel=np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))

# Коррекция контрастности
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(sharpened)

# Бинаризация
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Инициализация EasyOCR
reader = easyocr.Reader(['en', 'ru'])  # Укажите необходимые языки
results = reader.readtext(binary)

# Создание цветного изображения из бинарного
colored_binary = np.zeros((binary.shape[0], binary.shape[1], 3), dtype=np.uint8)
colored_binary[binary == 255] = [255, 255, 255]  # Белые области остаются белыми
colored_binary[binary == 0] = [0, 0, 0]          # Черные области остаются черными

# Рисуем прямоугольники на цветном бинарном изображении
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    # Рисуем зеленый прямоугольник вокруг распознанного текста
    cv2.rectangle(colored_binary, top_left, bottom_right, (0, 255, 0), 2)  # Зеленый цвет

# Отображение бинарного изображения с распознанным текстом
plt.figure(figsize=(10, 6))
plt.imshow(colored_binary)
plt.axis('off')
plt.title('Binary Image with Recognized Text')
plt.show()
fr_text = read_text("/kaggle/input/images/111.jpg", reader_en_fr)
print(fr_text)

# Применение beam search
model_name = "distilgpt2"  # Вы можете выбрать другую модель
nlp = pipeline("text-generation", model=model_name)

# Примените beam search
generated_text = nlp(fr_text, max_length=50, num_beams=5, num_return_sequences=1)
print("Сгенерированный текст:")
print(generated_text[0]['generated_text'])