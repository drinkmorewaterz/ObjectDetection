# Импортируем необходимые библиотеки
from ultralytics import YOLO
import numpy as np
import cv2
import threading
import pygame

# Инициализируем камеру
camera = cv2.VideoCapture(1)

# Загружаем модель YOLO с предобученными весами
model = YOLO("25epoch.pt")

# Создаем словарь меток объектов и соответствующих им цветов
labels = {0: u'car', 1: u'person', 2: u'rail-track', 3:'traffic-light', 4:'train'}
colors = [(89, 161, 197), (255, 0, 0), (0, 0, 255), (255, 120, 50), (100, 100, 100)]

# Инициализируем pygame для воспроизведения звуковых сигналов
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load('beep.wav')
def play_audio(interseption):
    """Функция для воспроизведения звукового сигнала"""
    # Проверяем, не проигрывается ли уже звуковой сигнал
    check = pygame.mixer.music.get_busy()
    if check:
        return
    # Если обнаружено пересечение объектом рельсовой колеи запускаем звуковой сигнал
    if interseption:
        pygame.mixer.music.play()
    # Иначе останавливаем проигрывание звука
    else:
        pygame.mixer.music.stop()

def draw_box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    """Функция для отрисовки прямоугольных рамок с метками классов"""
    # Определение толщины линии для рисования прямоугольника
    lw = max(round(sum(size_image) / 2 * 0.003), 2)
    # Определение координат двух противоположных углов прямоугольника
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    # Рисование прямоугольника на изображении
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    # Если задан текст метки, то рисуем его
    if label:
        # Определение толщины линии и размеров текста
        tf = max(lw - 1, 1)
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        # Определение координат двух противоположных углов для текстового блока
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # Рисование текстового блока на изображении
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        # Рисование текста метки на изображении
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

def draw_bounding_boxes(image, boxes, score=True, conf=None):
    """ Функция для отрисовки ограничивающих рамок на изображении"""
    # Для каждой рамки в списке рамок
    for box in boxes:
        # Получаем номер класса объекта на изображении
        class_num = int(box[-1])
        # Если номер класса не равен 2 (то есть не рельсовая колея)
        if class_num != 2:
            # Если нужно отобразить оценку правильности обнаружения объекта
            if score:
                # Создаем строку label с названием класса объекта и процентом правильности обнаружения
                label = f'{labels[class_num]} {str(round(100 * float(box[-2]), 1))}%'
            else:
                # Иначе просто создаем строку label с названием класса объекта
                label = labels[class_num]
                # Если задано значение порога правильности детектирования и текущий объект превышает этот порог
            if conf and box[-2] > conf:
                # Получаем цвет для рамки объекта на изображении
                color = colors[class_num]
                # Отрисовываем рамку со строкой label на изображении
                draw_box_label(image, box, label, color)
            else:
                color = colors[class_num]
                draw_box_label(image, box, label, color)

#Пока камера открыта
while camera.isOpened():
    # Получаем кадр с камеры
    _, img = camera.read()
    # Получаем размеры кадра
    size_image = img.shape
    # Получаем высоту, ширину и число каналов кадра
    H, W, count_canal = size_image
    #Создаем черную маску
    black_mask = np.zeros((H, W, count_canal), dtype=np.uint8)
    # Флаг пересечения
    interseption = False
    # Получаем предсказание модели
    result = model.predict(source=img, imgsz=1280)
    # Если есть маски
    if result[0].masks:
        # Проходимся по всем объектам на изображении
        for j in range(len(result[0].boxes.boxes)):
            # Если это железнодорожный путь
            if int(result[0].boxes.boxes[j][-1]) == 2:
                # Получаем маску объекта
                mask_lst = result[0].masks.segments[j]
                # Получаем координаты маски
                x = (mask_lst[:, 0] * W).astype('int')
                y = (mask_lst[:, 1] * H).astype('int')
                points = np.array([(x[i], y[i]) for i in range(len(x))], dtype=np.int32)
                # Закрашиваем область железнодорожного пути в зеленый цвет
                cv2.fillPoly(black_mask, [points], color=(0, 255, 0))
            # Добавляем маску железнодорожного пути на изображение
            rail_masks = cv2.addWeighted(img, 1, black_mask.astype('uint8'), 0.5, 0)
        # Проходимся по всем ограничивающим рамкам
        for i in result[0].boxes:
            # Получаем номер класса
            num_class = i.boxes[0][-1]
            # Если это человек
            if int(num_class) == 1:
                # Создаем маску для рамки
                bbox_mask = np.zeros((H, W, count_canal), dtype=np.uint8)
                # Получаем координаты рамки
                x1, y1, x2, y2 = map(int, i.xyxy[0])
                # Закрашиваем область рамки
                cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), color=200, thickness=-1)
                # Объединяем маски железнодорожного пути и рамки
                all_masks = cv2.bitwise_or(bbox_mask, black_mask)
                # Конвертируем изображение в оттенки серого
                gray = cv2.cvtColor(all_masks, cv2.COLOR_BGR2GRAY)
                # получаем уникальные значения цветов
                unique_colors = np.unique(gray)
                #Если количество уникальных цветов больше либо равно четырем, то устанавливаем флаг пересечения
                if len(unique_colors) >= 4:
                    interseption = True
        # Создаем поток для проигрывания звука
        thread = threading.Thread(target=play_audio, args=(interseption,))
        thread.start()
        # Вызываем функцию отрисовки ограничивающих рамок на изображении
        draw_bounding_boxes(rail_masks, result[0].boxes.boxes, score=False)
        # Отображение окна с обработанным изображением
        cv2.imshow('window', rail_masks)
        # Прерывание выполнения обработки видео при нажатии кнопки пробел
        if cv2.waitKey(1) == ord(' '):
            break