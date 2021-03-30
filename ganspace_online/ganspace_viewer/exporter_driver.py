# import sys
# import zmq 
# import json

# #TODO change this so that it works anywhere
# from export import GanModel
# from django import forms

# car = GanModel(model='StyleGAN2', class_name='car', layer='style', n=1_000_000, b=10_000)

# class ganspace_state(forms.Form):
#     model_name = forms.CharField(label = 'model_name', max_length=50)
#     layer_start_slider_value = forms.IntegerField()
#     layer_end_slider_value = forms.IntegerField()


# def sendDataToExporter(layer_start, layer_end, component_list):
#     print()
#     # print('layer start', layer_start)
#     # print('layer end', layer_end)
#     for i in range(len(component_list)):
#         # print('Component',i,component_list[i])
#         component_list[i] = float(component_list[i])

#     context = zmq.Context()

#     socket = context.socket(zmq.REQ)
#     socket.connect("tcp://ip-172-31-24-131.us-east-2.compute.internal:5555")

#     socket

#     model = 'car'
#     seed = '0'

#     js = {
#         "model": model,
#         "seed": int(seed),
#         "slider_values": component_list
#     }

#     socket.send_json(json.dumps(js))

#     filename = socket.recv_string()
#     print(filename)

#     return filename


from PIL import Image
from pathlib import Path
from os import sys
import os.path
from time import sleep
import numpy as np
import base64
from io import BytesIO
sys.path.insert(1, '/Users/duongp/ganspace-online')
    
def sendDataToExporter(model_name, layer_start, layer_end, component_list, seed):
    print()
    print('model name', model_name)
    print('layer start', layer_start)
    print('layer end', layer_end)
    for i in range(len(component_list)):
        print('Component',i,component_list[i])
    print('seed', seed)
    
    value = float(component_list[0])*10

    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'static/media/Images','shoe_even.jpg')
    even = Image.open(path)
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'static/media/Images','shoe_odd.jpg')
    odd = Image.open(path)
    buffered = BytesIO()
    
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'static/media/Images','test.jpg')
    if (int(value) % 2 == 0):
        even.save(path)
        #even.save(buffered, format="JPEG")
    else:
        #odd.save(buffered, format="JPEG")
        odd.save(path)
    #img_str = base64.b64encode(buffered.getvalue())
    #return img_str
    #sleep(0.1)
    



def generate_new_seed():
    return np.random.randint(np.iinfo(np.int32).max - 1)


