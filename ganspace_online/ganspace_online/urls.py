"""ganspace_online URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

from ganspace_viewer.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home', home_view, name = 'home'), #only for debugging
    
    path('manage_users', manage_users_view, name ='manage_users'),
    path('guides', guides_view, name ='guides'),
    path('account', account_view, name ='account'),

    path('data', data_view, name='data'),
    path('new_models', new_models_view, name='new_models'),
    path('models', models_view, name='models'),
    path('user_gallery', user_gallery_view, name='user_gallery'),

    path('models/<str:model_name>', ganspace_view, name='m_name'),

    path('', login_view, name = 'login'),
    path('register', register_view, name = 'register')
]

urlpatterns += staticfiles_urlpatterns()
