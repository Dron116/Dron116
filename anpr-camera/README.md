# Распознавание автомобильных номеров с IP-камеры

Веб-приложение показывает живой поток с камеры и распознаёт российские государственные номерные знаки. Результаты пишутся в журнал, последний номер отображается в виде таблички.

Камера по умолчанию: `192.168.83.22`, пользователь `admin`. Пароль задаётся только в файле `.env` и в git не попадает.

## Что умеет

- Подключение к IP-камере по HTTP-снимку (Hikvision, Dahua, Xiongmai и др.) или RTSP
- Автопоиск типичных URL, если точный поток неизвестен
- Демо-режим, если камера недоступна из текущей сети
- Распознавание стандартных номеров формата `А123ВС77` / `А123ВС777`
- Журнал фиксаций, поиск, загрузка фото для разового распознавания

## Запуск

Нужны Python 3.12+, Tesseract OCR и FFmpeg.

```bash
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus ffmpeg
cd anpr-camera
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

В `.env` укажите пароль камеры:

```
CAMERA_HOST=192.168.83.22
CAMERA_USER=admin
CAMERA_PASSWORD=ваш_пароль
```

Запуск:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Откройте http://127.0.0.1:8080 в браузере. Компьютер должен быть в той же сети, что и камера (`192.168.83.0/24`).

Если известен точный адрес снимка или RTSP, пропишите его:

```
CAMERA_SNAPSHOT_URL=http://192.168.83.22/ISAPI/Streaming/channels/101/picture
CAMERA_RTSP_URL=rtsp://admin:пароль@192.168.83.22:554/Streaming/Channels/101
```

## Docker

```bash
cd anpr-camera
docker build -t anpr-camera .
docker run --rm -p 8080:8080 --env-file .env anpr-camera
```

Сеть Docker должна видеть камеру. На Linux часто достаточно `--network host`.

## Тесты

```bash
cd anpr-camera
python -m pytest -q
```

## Замечания

- Приложение не открывает камеру из интернета: адрес `192.168.83.22` локальный.
- Для стабильного RTSP нужны FFmpeg и доступ к порту 554.
- OCR лучше работает при хорошем освещении и крупном номере в кадре.
