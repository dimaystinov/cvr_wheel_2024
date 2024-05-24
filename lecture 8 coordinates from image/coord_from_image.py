import math

def img_to_local_coord(x_px, l):
    Wpx = 640
    Hpx = 480
    H = 0 # см
    cam_angle_x = 100 # в градусах, угол бетта. При подстановке а тангенс перевести в радианы

    y = math.sqrt(l ** 2 - H ** 2)

    print(y)

    x = y * 2 * (x_px - Wpx / 2)  * math.tan(math.radians(cam_angle_x) / 2) / Wpx
    return [round(x), round(y)] # в сантиметрах

print(img_to_local_coord(500, 150))

