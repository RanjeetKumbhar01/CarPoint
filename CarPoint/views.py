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



