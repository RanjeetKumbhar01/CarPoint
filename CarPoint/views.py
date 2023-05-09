from django.http import HttpResponse
from django.shortcuts import render,redirect


def index(request):
    return render(request,'index.html')

def buyingPrice(request):
    return render(request,'price.html')
def buy(request):
    if request.method == 'POST':
        tourque=request.POST.get('Torque')
        Power=request.POST.get('Power')
        Mileage=request.POST.get('Mileage')
        Manufacturer=request.POST.get('Manufacturer')
        Transmission=request.POST.get('Transmission')
        Fuel=request.POST.get('Fuel')
        
        param={'Torque':tourque,
               "Power":Power,
               'Mileage':Mileage,               
               'Manufacturer':Manufacturer,
               'Transmission':Transmission,
               'Fuel':Fuel,
               'Price':100,
               }
        return render(request,'price.html',param)
    elif request.method == 'GET':

        
        return render(request,'price_init.html')



def sellingPrice(request):
    if request.method == 'POST':
        Manufacturer=request.POST.get('Manufacturer')    
        Model=request.POST.get('Model')    
        Type=request.POST.get('Type') 
        Year=request.POST.get('Year') 
        Fuel=request.POST.get('Fuel') 
        Condition=request.POST.get('Condition') 
        Odometer=request.POST.get('Odometer') 
        context={ 'Manufacturer':Manufacturer,    
                'Model':Model,  
                'Type':Type,
                'Year':Year,
                "Fuel":Fuel,
                'Condition':Condition,
                 "Odometer":Odometer}
        return render(request,'sell.html',context)
           
        # return HttpResponse(Year)
    elif request.method == 'GET':
       context={ 'Manufacturer':"",    
                'Model':"" ,  
                'Type':"",
                'Year':"",
                "Fuel":"",
                'Condition':"",
                 "Odometer":""}
       return render(request,'sell.html',context)
    
    
    
    return render(request,'sell.html')
    

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



