﻿# Classification_of_emotions
Реализованы нейронные сети для анализа mel-спектрограмм, для распознавания эмоций в речи
Для тренировки использовались данные из датасета Dusha

# utils.py
Файл с вспомогательными функциями

# dataset.py 
В данном файле реализованы два датасета. 
Первый для аудио записей формата .wav, он получает папку с данными, config и функцию transform, которая преобразовывает .wav в mel-спектрограмму.
Второй для готовых спектрограмм в формате .np

# models
В данной папке содержатся реализованные модели. У каждой для удобства есть файл train.py и config.json
В train реализован процесс тренировки, а config содержит в себе информацию о данных и структуре модели.
В настоящий момент реализованы CNN, VGG и ResNet

# Предварительные результаты
![image](https://github.com/Abramov-Kirill/classification_of_emotions/assets/99170345/0ad0a219-ccce-48a2-99df-898cfe012f0c)
