import cv2
import numpy as np

# Video dosyasini ac
cap = cv2.VideoCapture('video.avi')

# Guzergahi saklamak icin bos bir liste olustur
route_points = []
prev_frame_time = 0

# Zaman araliklari icin sinirlar belirle
time_intervals = [(0, 5), (6, 7), (8, 10), (11, 13)]

# Her zaman araligi icin ortalama hizlari saklamak icin bos bir liste olustur
average_speeds = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Goruntuyu HSV renk uzayina donustur
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Kirmizi bilardo toplarinin renk araligini belirle
    lower_red = np.array([115, 100, 100])
    upper_red = np.array([190, 255, 255])

    # Kirmizi toplarin maskesini olustur
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # Gurultuyu azaltmak icin morfolojik islemler (erozyon ve genlesme)
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.erode(mask_red, kernel, iterations=1)
    mask_red = cv2.dilate(mask_red, kernel, iterations=1)

    # Konturlari bul
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kontur analizi yap
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Kontur alani bir esik degerinden buyukse
            # Konturun sinirlayici kutusunu bul
            x, y, w, h = cv2.boundingRect(contour)
            # Topun merkez noktasini bul
            center_x = x + w // 2
            center_y = y + h // 2
            center = (center_x, center_y)
            # Merkez noktayi guzergah listesine ekle
            route_points.append(center)
    
    # Guzergahi ciz
    if len(route_points) > 1:
        for i in range(1, len(route_points)):
            cv2.line(frame, route_points[i - 1], route_points[i], (0, 0, 0), 2)  # Guzergah rengini siyah yap

    # Hizi hesapla ve goruntuye ekle
    if len(route_points) > 1:
        current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        for interval in time_intervals:
            if interval[0] <= current_frame_time < interval[1]:
                distance = np.sqrt((route_points[-1][0] - route_points[-2][0])**2 + (route_points[-1][1] - route_points[-2][1])**2)
                fps = cap.get(cv2.CAP_PROP_FPS)
                speed = distance * fps
                average_speeds.append(speed)
                cv2.putText(frame, f"Ortalama Hiz: {np.mean(average_speeds):.2f} birim/s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                break
    

    # Son kareye gelindiginde guzergahi ciz ve hizi hesapla
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
        if len(route_points) > 0:
            cv2.circle(frame, route_points[-1], 5, (0, 255, 0), -1)  # Son noktayi yesil bir daire olarak ekle
        cv2.imwrite('son_kare.png', frame)  # Son kareyi kaydet
        break
    
    # Goruntuyu goster
    cv2.imshow('Kirmizi Bilardo Toplari', frame)
    
    # 'q' tusuna basilirsa donguden cik
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
