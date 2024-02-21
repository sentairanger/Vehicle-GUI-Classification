from guizero import App, Drawing, PushButton
from button_app import *

def vehicle_detection():
    image_car = plt_show(convert_result(compiled_model_re, image_show(), resize_image(), box_car()))
    return image_car

def show_vehicle():
    viewer.image(20, 10, "car-gui-detect.jpg")
    
app = App(title="Vehicle GUI detection and recognition")
button = PushButton(app, text="Take picture", command=capture)
button2 = PushButton(app, text="Detect Vehicle", command=vehicle_detection)
button3 = PushButton(app, text="Show result", command=show_vehicle)
viewer = Drawing(app, width="fill", height="fill")
app.display()