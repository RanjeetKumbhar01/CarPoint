from django.http import HttpResponse
from django.shortcuts import render

def index(request):
    return render(request,'index.html')

def buyingPrice(request):
    return HttpResponse("<a href='/aboutUs'>click Me</a>")

def sellingPrice(request):
    return HttpResponse("It Is selling Price")

def firstModel(request):
    return HttpResponse("It Is Model 1")

def secondModel(request):
    return HttpResponse("It Is Model 2")

def thiredModel(request):
    return HttpResponse("It Is Model 3")

def dataset1(request):
    return HttpResponse("It Is dataset 1")

def dataset2(request):
    return HttpResponse("It Is dataset 2")

def dataset3(request):
    return HttpResponse("It Is dataset 3")

def dataset4(request):
    return HttpResponse("It Is dataset 4")

def dataset5(request):
    return HttpResponse("It Is dataset 5")

def discover(requet):
    return HttpResponse("Discover")

def aboutUs(request):
    return HttpResponse("about us")



