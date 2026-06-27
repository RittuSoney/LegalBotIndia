from django.urls import path
from . import views

urlpatterns = [
    path('ask/', views.chat_api, name='chat_api'),       # The API (Backend)
    path('', views.chat_interface, name='chat_home'),    # The Frontend (HTML)
]