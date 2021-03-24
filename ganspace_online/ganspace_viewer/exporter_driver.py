import sys
sys.path.insert(1, '/Users/duongp/ganspace-online') #TODO change this so that it works anywhere
from export import GanModel
#from django import forms

#car = GanModel(model='StyleGAN2', class_name='car', layer='style', n=1_000_000, b=10_000)

# class ganspace_state(forms.Form):
#     model_name = forms.CharField(label = 'model_name', max_length=50)
#     layer_start_slider_value = forms.IntegerField()
#     layer_end_slider_value = forms.IntegerField()


def sendDataToExporter(layer_start, layer_end, component_list):
    print()
    print('layer start', layer_start)
    print('layer end', layer_end)
    for i in range(len(component_list)):
        print('Component',i,component_list[i])
