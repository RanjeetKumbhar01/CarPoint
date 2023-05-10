from django.http import HttpResponse
from django.shortcuts import render,redirect
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np


def index(request):
    return render(request,'index.html')

def buyingPrice(request):
    return render(request,'price.html')

def getMakeList_ModelPrice():
    make = ['Audi', 'Bajaj', 'Bmw', 'Datsun', 'Dc', 'Fiat',
            'Force', 'Ford', 'Honda', 'Hyundai', 'Icml', 'Isuzu',
            'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'Land Rover Rover',
            'Lexus', 'Mahindra', 'Maruti Suzuki', 'Maruti Suzuki R', 'Mg',
            'Mini', 'Mitsubishi', 'Nissan', 'Premier', 'Renault',
            'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
    
    return make

def modelPrice(torque, power, mileage, manufacturer, fuel, transmission):
    """_summary_

    Args:
        torque (int): value of torque
        power (int): value of power
        mileage (int ): value of mileage
        manufacturer (string): name of manufacturer
        fuel (string): type of fuel
        transmission (string): type of transmission

    Returns:
        price: predicted price of Price Model
    """
    make = getMakeList_ModelPrice()

    print("Index of ",manufacturer," is ",make.index(manufacturer,0))
    # with open('Models\\New_rf_price_pred_90acc__all.pkl', 'rb') as f:
    #     model = pickle.load(f)s
    model = pickle.load(open('Models\\New_rf_price_pred_90acc__all.pkl', 'rb'))

    # When a user enters new input, transform the input using the same scaler
    # scaler = StandardScaler()

    # user_input = np.array([[torque, power, mileage, manufacturer, fuel, transmission]])
    # user_input_scaled = scaler.transform(user_input)

    # # Use the trained model to make predictions on the scaled user input
    # predicted_value = model.predict(user_input_scaled)

    return 0

def buy(request):
    if request.method == 'POST':
        torque=request.POST.get('Torque')
        power=request.POST.get('Power')
        mileage=request.POST.get('Mileage')
        manufacturer=request.POST.get('Manufacturer')
        transmission=request.POST.get('Transmission')
        fuel=request.POST.get('Fuel')
        
        price = modelPrice(torque, power, mileage, manufacturer, fuel, transmission)
        
        param={'Torque':torque,
               "Power":power,
               'Mileage':mileage,               
               'Manufacturer':manufacturer,
               'Transmission':transmission,
               'Fuel':fuel,
               'Price':price,
               }
        print(param)

        return render(request,'price.html',param)
    elif request.method == 'GET':

        
        return render(request,'price_init.html')



def sellingPrice(request):
    if request.method == 'POST':
        Manufacturer=request.POST.get('Manufacturer')    
        Model=request.POST.get('Model')    
        Type=request.POST.get('Type')         
        Year= (request.POST.get('Year')).split('-')[0]    #do int parsing here    
        Fuel=request.POST.get('Fuel') 
        Condition=request.POST.get('Condition') 
        Odometer= request.POST.get('Odometer')   #do int parsing here
        Price=100
        context={ 'Manufacturer':Manufacturer,    
                'Model':Model,  
                'Type':Type,
                'Year':request.POST.get('Year'),
                "Fuel":Fuel,
                'Condition':Condition,
                 "Odometer":Odometer,
                 "Price":Price}
        return render(request,'sell.html',context)
           
        # return HttpResponse(Year)
    elif request.method == 'GET':
       context={ 'Manufacturer':"",    
                'Model':"" ,  
                'Type':"",
                'Year':"",
                "Fuel":"",
                'Condition':"",
                 "Odometer":"",
                 "Price":""}
       return render(request,'sell.html',context)
    
    
    
    return render(request,'sell.html')
    

def firstModel(request):
       if request.method == 'POST':
        price=request.POST.get('Price')    
        fuel=request.POST.get('Fuel')    
        body=request.POST.get('Type') 
        seat= request.POST.get('Seat')   #do int
        mileage=request.POST.get('Mileage') #do int
        
        modal='BMW'
        context={ 
                  'Price':price,
                  'Fuel':fuel,
                  'Type':body,
                  'Seat':seat,
                  'Mileage':mileage,
                  'Modal':modal
        }
        return render(request,'model1.html',context)
           
        # return HttpResponse(Year)
       elif request.method == 'GET':
           context={ 'Price':'',
                  'Fuel':'',
                  'Type':'',
                  'Seat':'',
                  'Mileage':'',
                  'Modal':''
                  }
           return render(request,'model1.html',context)    
       return render(request,'model1.html')
    
    

def secondModel(request):
       if request.method == 'POST':
        price=request.POST.get('Price')    
        fuel=request.POST.get('Fuel')    
        body=request.POST.get('Type') 
        seat= request.POST.get('Seat')  
        mileage=request.POST.get('Mileage') 
        
        modal='BMW'
        context={ 
                  'Price':price,
                  'Fuel':fuel,
                  'Type':body,
                  'Seat':seat,
                  'Mileage':mileage,
                  'Modal':modal
        }
        return render(request,'model2.html',context)
           
        # return HttpResponse(Year)
       elif request.method == 'GET':
           context={ 'Manufacturer':"",    
                'Model':"" ,  
                'Type':"",
                'Year':"",
                "Fuel":"",
                'Condition':"",
                 "Odometer":"",
                  }
           return render(request,'model2.html',context)    
       return render(request,'model2.html')

def thiredModel(request):
     if request.method == 'POST':
        price=request.POST.get('Price')    
        fuel=request.POST.get('Fuel')         
        mileage=request.POST.get('Mileage')   #do int
        
        modal='BMW'
        context={ 
                  'Price':price,
                  'Fuel':fuel,                  
                  'Mileage':mileage,
                  'Modal':modal
        }
        return render(request,'model3.html',context)
           
        # return HttpResponse(Year)
     elif request.method == 'GET':
           context={ 'Price':'',
                  'Fuel':'',                  
                  'Mileage':'',
                  'Modal':''
                  }
           return render(request,'model3.html',context)    
     return render(request,'model3.html')

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



