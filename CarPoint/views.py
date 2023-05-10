from django.http import HttpResponse
from django.shortcuts import render, redirect
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import numpy as np
import locale  # to convert number to currency format
import pandas as pd
import tensorflow as tf


def index(request):
    return render(request, 'index.html')


def buyingPrice(request):
    return render(request, 'price.html')

# M O D E L -- P R I C E


def get_ModelPrice(list):
    if (list == 'make'):
        make = ['Audi', 'Bajaj', 'Bmw', 'Datsun', 'Dc', 'Fiat',
                'Force', 'Ford', 'Honda', 'Hyundai', 'Icml', 'Isuzu',
                'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'Land Rover Rover',
                'Lexus', 'Mahindra', 'Maruti Suzuki', 'Maruti Suzuki R', 'Mg',
                'Mini', 'Mitsubishi', 'Nissan', 'Premier', 'Renault',
                'Skoda', 'Tata', 'Toyota', 'Volkswagen', 'Volvo']
        return make
    elif (list == 'trans'):
        trans = ['Manual', 'Automatic', 'AMT', 'CVT', 'DCT']
        return trans
    else:
        # Fuel
        fuel = ['Petrol', 'CNG', 'Diesel',
                'CNG + Petrol', 'Electric', 'Hybrid']
        return fuel


def find_key(char_list, target_char):
    return [int(char == target_char) for char in char_list]
    # D E B U G G I N G   A R E A

    # result = []
    # for char in char_list:
    #     if char == target_char:
    #         result.append(1)
    #     else:
    #         result.append(0)
    # return result


def modelPrice(torque, power, mileage, manufacturer, fuel_type, transmission):
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
    make = get_ModelPrice('make')
    fuel = get_ModelPrice('fuel')
    trans = get_ModelPrice('trans')

    t_make = find_key(make, manufacturer)
    t_fuel = find_key(fuel, fuel_type)
    t_trans = find_key(trans, transmission)
    model = pickle.load(open('Models\\New_rf_price_pred_90acc__all.pkl', 'rb'))
    scaler = pickle.load(
        open('Models\\GEN car price\\new_standard_scaler_GEN_car_price.pkl', 'rb'))
    # D E B U G G I N G   A R E A

    # print("\nBefore scaling:\n")
    # param = {'Torque': type(torque),
    #              "Power": type(power),
    #              'Mileage': mileage,
    #              'Manufacturer': type(t_make),
    #              'Transmission': type(t_trans),
    #              'Fuel': type(t_fuel),
    #              'Price': 0,
    #              }
    # print(param)

    user_input = np.concatenate(
        ([torque, power, mileage], t_make, t_fuel, t_trans))

    # Reshape the input vector to have shape (1, n_features)
    user_input = user_input.reshape(1, -1)

    # Scale the input vector using the pre-trained scaler
    user_input_scaled = scaler.transform(user_input)
    print("\nafter scaling:\n")
    print(user_input_scaled)

    # # Use the trained model to make predictions on the scaled user input
    predicted_value = model.predict(user_input_scaled)
    print("predicted_value ", predicted_value)
    # Set the user's locale to the default system locale
    locale.setlocale(locale.LC_ALL, '')
    formatted_number = locale.currency(
        predicted_value[0].astype(int), grouping=True)
    if formatted_number.endswith('.00'):
        formatted_number = formatted_number[:-3]
    return formatted_number


def buy(request):
    if request.method == 'POST':
        torque = int(request.POST.get('Torque'))
        power = int(request.POST.get('Power'))
        mileage = int(request.POST.get('Mileage'))
        manufacturer = request.POST.get('Manufacturer')
        transmission = request.POST.get('Transmission')
        fuel = request.POST.get('Fuel')

        price = modelPrice(torque, power, mileage,
                           manufacturer, fuel, transmission)
        param = {'Torque': torque,
                 "Power": power,
                 'Mileage': mileage,
                 'Manufacturer': manufacturer,
                 'Transmission': transmission,
                 'Fuel': fuel,
                 'Price': price,
                 }

        return render(request, 'price.html', param)
    elif request.method == 'GET':

        return render(request, 'price_init.html')

# M O D E L -- R E - S E L L  P R I C E


def get_ModelReSellPrice(list):

    if (list == 'make'):
        make = ['jeep', 'bmw', 'dodge', 'chevrolet', 'ford', 'honda', 'toyota',
                'nissan', 'subaru', 'gmc', 'volkswagen', 'kia', 'acura', 'ram',
                'chrysler', 'hyundai', 'cadillac', 'volvo', 'mercedes-benz',
                'audi', 'infiniti', 'mazda', 'mini', 'buick', 'mitsubishi',
                'rover', 'pontiac', 'lincoln', 'lexus', 'fiat', 'jaguar',
                'mercury', 'saturn', 'tesla', 'harley-davidson', 'ferrari',
                'land rover', 'porche', 'alfa-romeo', 'morgan', 'aston-martin']
        return make
    elif (list == 'model'):
        df = pd.read_csv('Models\\resell\\model_unique_car_resell.csv')
        model = df.model_name.tolist()
        return model
    elif (list == 'cartype'):
        car_type = ['like new', 'excellent', 'fair', 'good', 'new', 'salvage']
        return car_type
    elif (list == 'codition'):
        condition = ['offroad', 'pickup', 'convertible', 'van', 'SUV', 'sedan', 'coupe',
                     'hatchback', 'mini-van', 'truck', 'wagon', 'other', 'bus']
        return condition
    else:
        # Fuel
        fuel = ['gas', 'electric', 'diesel', 'hybrid', 'other']
        return fuel


def find_key(char_list, target_char):
    return [int(char == target_char) for char in char_list]


def get_label(df, col, val):

    labelencoder = LabelEncoder().fit(df[col])
    return labelencoder.transform([val])


def modelresell(year, manufacturer, car_model, car_condition, fuel_type, odometer, cartype):

    df = pd.read_csv('Models\\new resell\\new_df_resell_required.csv')

    t_make = get_label(df, 'manufacturer', manufacturer)
    t_model = get_label(df, 'model', car_model)
    t_condition = get_label(df, 'condition', car_condition)
    t_fuel = get_label(df, 'fuel', fuel_type)
    t_type = get_label(df, 'type', cartype)

    model = pickle.load(open('Models\\new resell\\latest_xg_reg.pkl', 'rb'))
    scaler = pickle.load(
        open('Models\\new resell\\latest_standard_scaler_resell_price.pkl', 'rb'))
    labelencoder = pickle.load(
        open('Models\\new resell\\latest_standard_scaler_resell_price.pkl', 'rb'))

    # Transform a category into a numerical value

    # D E B U G G I N G   A R E A

    # print("\nBefore scaling:\n")

    cat_cols = [manufacturer, car_model, fuel_type, cartype, car_condition]

    t_make = labelencoder.transform(np.array(t_make).reshape(-1, 1))
    t_model = labelencoder.transform(np.array(t_model).reshape(-1, 1))
    t_condition = labelencoder.transform(np.array(t_condition).reshape(-1, 1))
    t_fuel = labelencoder.transform(np.array(t_fuel).reshape(-1, 1))
    t_type = labelencoder.transform(np.array(t_type).reshape(-1, 1))

    year = scaler.transform([[year]])
    odometer = scaler.transform([[odometer]])
    t_model = scaler.transform(t_model)

    user_input = np.concatenate(
        (t_make, t_model, t_fuel, t_type, t_condition, year, odometer))
    print("\nafter scaling:\n")
    print(type(user_input))
    print(user_input.shape)

    # # Use the trained model to make predictions on the scaled user input
    predicted_value = model.predict(user_input.reshape(1, -1))
    print("pred: ", predicted_value)
    print("pred log: ", np.power(2.71828, predicted_value))
    predicted_value = np.power(2.71828, predicted_value)
    locale.setlocale(locale.LC_ALL, '')

    # Define the currency symbol
    dollar = '$'

    # Use the locale.currency() function to format the predicted value as a dollar amount
    formatted_number = locale.currency(
        predicted_value[0].astype(int), symbol=dollar, grouping=True)

    # Remove the '.00' from the end of the formatted string, if it exists
    if formatted_number.endswith('.00'):
        formatted_number = formatted_number[:-3]
        return formatted_number

# @ajay


def sellingPrice(request):
    if request.method == 'POST':
        manufacturer = request.POST.get('Manufacturer')
        car_model = request.POST.get('Model')
        cartype = request.POST.get('Type')
        year = int((request.POST.get('Year')).split(
            '-')[0])  # do int parsing here
        fuel_type = request.POST.get('Fuel')
        car_condition = request.POST.get('Condition')
        odometer = int(request.POST.get('Odometer'))  # do int parsing here
        price = modelresell(int(year), manufacturer, car_model,
                            car_condition.lower(), fuel_type, int(odometer), cartype)

        context = {'Manufacturer': manufacturer,
                   'Model': car_model,
                   'Type': cartype,
                   #    'Year': year,
                   'Year': request.POST.get('Year'),
                   "Fuel": fuel_type,
                   'Condition': car_condition,
                   "Odometer": odometer,
                   "Price": "$ "+(price).split(' ')[1]}

        return render(request, 'sell.html', context)

        # return HttpResponse(Year)
    elif request.method == 'GET':
        context = {'Manufacturer': "",
                   'Model': "",
                   'Type': "",
                   'Year': "",
                   "Fuel": "",
                   'Condition': "",
                   "Odometer": ""}
        return render(request, 'sell.html', context)

    return render(request, 'sell.html')

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# M O D E L --  G E N  M O D E L


def get_genModel(list):
    if (list == 'type'):
        type = ['Hatchback', 'MPV', 'MUV', 'SUV', 'Sedan', 'Crossover', 'Coupe','Convertible', 'Sports', 'Pick-up']
        return type
    elif (list == 'model'):
        model = ['3-Series', '5-Series', '6-Series', 'A3', 'A3 Cabriolet', 'A4', 'A5', 'A6', 'Abarth Avventura', 'Abarth Punto', 'Accord Hybrid', 'Alto', 'Alto 800 Tour', 'Alto K10', 'Altroz', 'Alturas G4', 'Amaze', 'Ameo', 'Aspire', 'Aura', 'Avanti', 'Avventura', 'Baleno', 'Baleno Rs', 'Bolero', 'Bolero Power Plus', 'Bolt', 'Brv', 'Camry', 'Captur', 'Carnival', 'Celerio', 'Celerio Tour', 'Celerio X', 'Ciaz', 'City', 'Civic', 'Clubman', 'Compass', 'Compass Trailhawk', 'Convertible', 'Cooper 3 Door', 'Cooper 5 Door', 'Corolla Altis', 'Countryman', 'Cr-V', 'Creta', 'Discovery Sport', 'Dmax V-Cross', 'Duster', 'Dzire', 'Dzire Tour', 'E Verito', 'E2O Plus', 'Ecosport', 'Eeco', 'Elantra', 'Elite I20', 'Endeavour', 'Ertiga', 'Es', 'Etios Cross', 'Etios Liva', 'Extreme', 'F-Pace', 'Figo', 'Fortuner', 'Freestyle', 'Glanza', 'Go', 'Go+', 'Grand I10', 'Grand I10 Nios', 'Grand I10 Prime', 'Gurkha', 'Gypsy', 'Harrier', 'Hector', 'Hexa', 'I20 Active', 'Ignis', 'Innova Crysta', 'Jazz', 'John Cooper Works', 'Kicks', 'Kodiaq', 'Kodiaq Scout', 'Kona Electric', 'Kuv100 Nxt', 'Kwid', 'Linea', 'Linea Classic',
                 'Lodgy', 'Marazzo', 'Mercedes-Benz A-Class', 'Mercedes-Benz B-Class', 'Mercedes-Benz C-Class', 'Mercedes-Benz C-Class Cabriolet', 'Mercedes-Benz Cla-Class', 'Mercedes-Benz E-Class', 'Mercedes-Benz Gla-Class', 'Mercedes-Benz Glc', 'Micra', 'Micra Active', 'Monte Carlo', 'Mu-X', 'Nano Genx', 'Nexon', 'Nexon Ev', 'Nuvosport', 'Nx 300H', 'Octavia', 'Omni', 'Outlander', 'Pajero Sport', 'Passat', 'Platinum Etios', 'Polo', 'Prius', 'Punto Evo', 'Punto Evo Pure', 'Q3', 'Q5', 'Qute (Re60)', 'Range Evoque', 'Rapid', 'Redi-Go', 'Rio', 'S-Cross', 'S-Presso', 'S60', 'S60 Cross Country', 'S90', 'Safari Storme', 'Santro', 'Scorpio', 'Seltos', 'Sunny', 'Superb', 'Superb Sportline', 'Swift', 'Terrano', 'Thar', 'Tiago', 'Tiago Nrg', 'Tigor', 'Tigor Ev', 'Tiguan', 'Triber', 'Tucson', 'Tuv300', 'Tuv300 Plus', 'Urban Cross', 'V40', 'V40 Cross Country', 'Vento', 'Venue', 'Verito', 'Verito Vibe', 'Verna', 'Vitara Brezza', 'Wagon R', 'Winger', 'Wr-V', 'Wrangler', 'X1', 'X3', 'X4', 'Xc40', 'Xc60', 'Xcent', 'Xcent Prime', 'Xe', 'Xf', 'Xl6', 'Xuv300', 'Xuv500', 'Xylo', 'Yaris', 'Zest', 'Zs Ev']
        model = pd.Index(model)
        return model
    else:
        fuel = ['Petrol', 'CNG', 'Diesel','CNG + Petrol', 'Hybrid', 'Electric']
        return fuel


def get_key(char_list, target_char):
    # return [int(char == target_char) for char in char_list]
    # D E B U G G I N G   A R E A
    result = []
    for char in char_list:
        if char == target_char:
            result.append(1)
        else:
            result.append(0)
    return result


def genModel(price, seat, mileage, fuel_type, body_type):
    fuel = get_genModel('fuel')
    type = get_genModel('type')
    model_list = get_genModel('model')
    
    # for i in fuel:
    #     print(i)
    # for i in type:
    #     print(i)
    t_type = get_key(type, body_type)
    t_fuel = get_key(fuel, fuel_type)

    model = tf.keras.models.load_model(
        'Models\\gen model\\latest_better_65_without_trans_gen_model.h5')
    scaler = pickle.load(
        open('Models\\gen model\\standard_scaler_gen_model.pkl', 'rb'))
    # D E B U G G I N G   A R E A
    price = int(price)
    print("\nBefore scaling:\n")
    param = {
        'Price': (price),
        "seat": (seat),
        'Mileage': (mileage),
        't_type': len(t_type),
        't_fuel': len(t_fuel),
    }


    user_input = np.concatenate(
        ([price, seat, mileage], t_fuel, t_type))

    # Reshape the input vector to have shape (1, n_features)
    user_input = user_input.reshape(1, -1)

    # Scale the input vector using the pre-trained scaler
    user_input_scaled = scaler.transform(user_input)
    print("\nafter scaling:\n")
    print(user_input_scaled)

    # # Use the trained model to make predictions on the scaled user input
    predicted_value = model.predict(user_input_scaled)
    max_index = np.argmax(predicted_value, axis=1)
    print("predicted_value", model_list[max_index][0])
    print(len(model_list[max_index][0]))
    return model_list[max_index][0]


def firstModel(request):
    if request.method == 'POST':
<<<<<<< HEAD
        fuel = request.POST.get('Fuel')
        price = request.POST.get('Price')
        body = request.POST.get('Type')
        seat = int(request.POST.get('Seat'))
        mileage = int(request.POST.get('Mileage'))
        # price = int(price)
        # model = genModel(price, seat, mileage, fuel, body)
        model = 'BMW'
        print("-------------")
=======
        price=request.POST.get('Price')    
        fuel=request.POST.get('Fuel')    
        body=request.POST.get('Type') 
        seat= request.POST.get('Seat')   #do int
        mileage=request.POST.get('Mileage') #do int
        
        model='BMW'
>>>>>>> 7aa1fa2494c114a75e5801939ee1d217674a5f1c
        context={ 
                  'Price':price,
                  'Fuel':fuel,
                  'Type':body,
                  'Seat':seat,
                  'Mileage':mileage,
                  'Model':model
        }
        print(context)
        return render(request, 'model1.html', context)

        # return HttpResponse(Year)
    elif request.method == 'GET':
<<<<<<< HEAD
        context = {'Price': '',
                   'Fuel': '',
                   'Type': '',
                   'Seat': '',
                   'Mileage': '',
                   'Modal': ''
                   }
        return render(request, 'model1.html', context)
    return render(request, 'model1.html')

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
=======
           context={ 'Price':'',
                  'Fuel':'',
                  'Type':'',
                  'Seat':'',
                  'Mileage':'',
                  'Model':''
                  }
           return render(request,'model1.html',context)    
    return render(request,'model1.html')
>>>>>>> 7aa1fa2494c114a75e5801939ee1d217674a5f1c

def getBinary(data):
    if(data=='Yes'):
        return 1
    else:
        return 0
    
def secondModel(request):
    if request.method == 'POST':
<<<<<<< HEAD
        price = request.POST.get('Price')
        fuel = request.POST.get('Fuel')
        body = request.POST.get('Type')
        seat = request.POST.get('Seat')
        mileage = request.POST.get('Mileage')

        modal = 'BMW'
        context = {
            'Price': price,
            'Fuel': fuel,
            'Type': body,
            'Seat': seat,
            'Mileage': mileage,
            'Modal': modal
=======
        price=request.POST.get('Price')    
        airBag=request.POST.get('airBag')    
        engine=getBinary(request.POST.get('Engine'))         
        abc= getBinary(request.POST.get('ABC'))  
        ebd=getBinary(request.POST.get('EBD'))
        esc=getBinary(request.POST.get('ESC') )
        
        model='BMW'
        context={ 
                  'Price':price,
                  'airBag':airBag,
                  'Engine':request.POST.get('Engine'),
                  'ABC':request.POST.get('ABC'),
                  'EBD':request.POST.get('EBD'),
                  'ESC':request.POST.get('ESC') ,
                  'Model':model
>>>>>>> 7aa1fa2494c114a75e5801939ee1d217674a5f1c
        }
        return render(request, 'model2.html', context)

        # return HttpResponse(Year)
    elif request.method == 'GET':
<<<<<<< HEAD
        context = {'Manufacturer': "",
                   'Model': "",
                   'Type': "",
                   'Year': "",
                   "Fuel": "",
                   'Condition': "",
                   "Odometer": "",
                   }
        return render(request, 'model2.html', context)
    return render(request, 'model2.html')
=======
           context={ 'Price':'',
                  'airBag':'',
                  'Engine':'',
                  'ABC':'',
                  'EBD':'',
                  'ESC':'',
                  'Model':''
                  }
           return render(request,'model2.html',context)    
    return render(request,'model2.html')
>>>>>>> 7aa1fa2494c114a75e5801939ee1d217674a5f1c


def thiredModel(request):
    if request.method == 'POST':
<<<<<<< HEAD
        price = request.POST.get('Price')
        fuel = request.POST.get('Fuel')
        mileage = request.POST.get('Mileage')  # do int

        modal = 'BMW'
        context = {
            'Price': price,
            'Fuel': fuel,
            'Mileage': mileage,
            'Modal': modal
=======
        price=request.POST.get('Price')    
        fuel=request.POST.get('Fuel')         
        mileage=request.POST.get('Mileage')   #do int
        
        model='BMW'
        context={ 
                  'Price':price,
                  'Fuel':fuel,                  
                  'Mileage':mileage,
                  'Model':model
>>>>>>> 7aa1fa2494c114a75e5801939ee1d217674a5f1c
        }
        return render(request, 'model3.html', context)

        # return HttpResponse(Year)
    elif request.method == 'GET':
<<<<<<< HEAD
        context = {'Price': '',
                   'Fuel': '',
                   'Mileage': '',
                   'Modal': ''
                   }
        return render(request, 'model3.html', context)
    return render(request, 'model3.html')
=======
           context={ 'Price':'',
                  'Fuel':'',                  
                  'Mileage':'',
                  'Model':''
                  }
           return render(request,'model3.html',context)    
    return render(request,'model3.html')
>>>>>>> 7aa1fa2494c114a75e5801939ee1d217674a5f1c


def dataset1(request):
    return  render(request,'dataset1.html')


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
