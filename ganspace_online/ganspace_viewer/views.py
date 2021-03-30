from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from .models import *
from .exporter_driver import *
from .forms import *

from django.contrib.auth import authenticate, login, logout
from django.contrib import messages


# Create your views here.
def home_view(request, *args, **kwargs):
    return render(request, "home.html", {})
from django.shortcuts import render

def login_view(request):
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username =username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.info(request, 'Username or Password is incorrect')

    context = {}
    return render(request, "login.html",context)

#TODO: implement this
def logoutUser(request):
    return redirect('login')

def register_view(request):
    form = CreateUserForm()

    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(request, 'Account created for ' + user)
            
            return redirect('login')


    context= { 'form': form}
    return render(request, "register.html",context)

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
        model_name = request.POST.get('model_name')
        layer_start = request.POST.get('layer_start')
        layer_end = request.POST.get('layer_end')
        component_list = request.POST.getlist('component_sliders[]')
        response_type = request.POST.get('type')
        seed = request.POST.get('seed')
        if (response_type == 'slider_value_update' or response_type == 'update_seed'):
            img_base64 = sendDataToExporter(model_name, layer_start, layer_end, component_list, seed)
            return JsonResponse({'base64_image': img_base64})
        elif (response_type == 'resample_latent'):
            seed = generate_new_seed()
            img_base64 = sendDataToExporter(model_name, layer_start, layer_end, component_list, seed)
            return JsonResponse({'seed': seed})

    #initialize 
    seed = generate_new_seed()
    context = {
        "component_list": range(10),
        "seed": seed,

    }
    return render(request,'admin_ganspace_viewer.html', context)
