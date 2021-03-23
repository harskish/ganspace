from django.shortcuts import render
from django.http import HttpResponse
from .models import *

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
    context = {
        "component_list": range(5),
    }
    return render(request,  'admin_ganspace_viewer.html', context)

#User Views