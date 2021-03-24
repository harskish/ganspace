from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
from .models import *
from .exporter_driver import *

# Create your views here.
def home_view(request, *args, **kwargs):
    return render(request, "home.html", {})
from django.shortcuts import render

# Create your views here.

#Admin Views
def manage_users_view(request, *args, **kwargs):
    return render(request, "admin_manage_users.html", {})

def guides_view(request, *args, **kwargs):
    return render(request, "admin_guides.html", {})

def account_view(request, *args, **kwargs):
    return render(request, "admin_account.html", {})


def data_view(request, *args, **kwargs):
    return render(request, "admin_data.html", {})

def new_models_view(request, *args, **kwargs):
    return render(request, "admin_new_models.html", {})

def models_view(request ,*args, **kwargs):
    #model_list = kwargs.models
    model_list = ganspace_model.objects.values_list('className', flat = True)
    my_context = {
        "model_list" : model_list , #TODO: implement exporter
    }
    
    return render(request, "admin_models.html", my_context)

def user_gallery_view(request, *args, **kwargs):
    return render(request, "admin_user_gallery.html", {})

def ganspace_view(request, model_name):
    model = ganspace_model.objects.get(className = model_name)

    # seed = request.POST.get('seed')
    # layer_start = request.POST.get('layer_start')
    # print(seed, layer_start)

    #print(request.POST.get('layer_start'))

    if request.method == 'POST' and request.is_ajax():
        layer_start = request.POST.get('layer_start')
        layer_end = request.POST.get('layer_end')
        component_list = request.POST.getlist('component_sliders[]')
        sendDataToExporter(layer_start, layer_end, component_list)
        


    context = {
        "component_list": range(10),
    }
    return render(request,'admin_ganspace_viewer.html', context)

#User Views