from django.http import HttpResponse
from django.shortcuts import render, redirect
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import numpy as np
import locale  # to convert number to currency format
import pandas as pd


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
        fuel = ['Petrol', 'CNG', 'Diesel',
                'CNG + Petrol', 'Electric', 'Hybrid']
        return fuel


def find_key(char_list, target_char):
    return [int(char == target_char) for char in char_list]


def get_label(df, col, val):

    labelencoder = LabelEncoder().fit(df[col])
    return labelencoder.transform([val])


def modelresell(year, manufacturer, car_model, car_condition, fuel_type, odometer, cartype):

    df = pd.read_csv('Models\new resell\new_df_resell_required.csv')

    make = get_ModelReSellPrice('make')
    models = get_ModelReSellPrice('model')
    condition = get_ModelReSellPrice('condition')
    fuel = get_ModelReSellPrice('fuel')
    type = get_ModelReSellPrice('cartype')

    # manufacturer	model	fuel	type	condition	year	odometer

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

    print("\nBefore scaling:\n")
    param = {'year': year,
             "make": t_make,
             'condition': t_condition,
             'fuel': t_fuel,
             'odometer': odometer,
             'cartype': cartype,
             }
    print(param)

    cat_cols = [manufacturer, car_model, fuel_type, cartype, car_condition]
    # manufacturer	model	fuel	type	condition	year	odometer
    # user_input = np.concatenate(([torque, power, mileage], t_make, t_fuel, t_trans))

    odometer = labelencoder.transform(np.array(odometer).reshape(-1, 1))
    year = labelencoder.transform(np.array(year).reshape(-1, 1))
    t_model = labelencoder.transform(np.array(t_model).reshape(-1, 1))

    user_input = np.concatenate(
        (t_make, t_model, t_fuel, t_type, t_condition, [year, odometer]))

    # Reshape the input vector to have shape (1, n_features)
    user_input = user_input.reshape(1, -1)

    # Scale the input vector using the pre-trained scaler
    year = scaler.transform(year)
    odometer = scaler.transform(odometer)
    t_model = scaler.transform(t_model)

    user_input = np.concatenate(
        (t_make, t_model, t_fuel, t_type, t_condition, [year, odometer]))
    print("\nafter scaling:\n")
    print(user_input)

    # # Use the trained model to make predictions on the scaled user input
    predicted_value = model.predict(user_input)
    # Set the user's locale to the default system locale
    locale.setlocale(locale.LC_ALL, '')
    formatted_number = locale.currency(
        predicted_value[0].astype(int), grouping=True)
    if formatted_number.endswith('.00'):
        formatted_number = formatted_number[:-3]
    return formatted_number

# @ajay


def sellingPrice(request):
    if request.method == 'POST':
        Manufacturer = request.POST.get('Manufacturer')
        Model = request.POST.get('Model')
        Type = request.POST.get('Type')
        Year = request.POST.get('Year')
        Fuel = request.POST.get('Fuel')
        Condition = request.POST.get('Condition')
        Odometer = request.POST.get('Odometer')
        context = {'Manufacturer': Manufacturer,
                   'Model': Model,
                   'Type': Type,
                   'Year': Year,
                   "Fuel": Fuel,
                   'Condition': Condition,
                   "Odometer": Odometer}
        return render(request, 'sell.html', context)
        # price = modelresell(torque, power, mileage,
        #                     manufacturer, fuel, transmission)
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
# M O D E L -- R E - S E L L  P R I C E


'''
Price', 'Seating_Capacity', 'Mileage', 'Fuel_Type ','Body_Type_'
'''


def get_carModel_Gen(list):
    if (list == 'model'):
        model = ['Nano Genx', 'Redi-Go', 'Kwid', 'Eeco', 'Alto K10', 'Go',
                 'Celerio Tour', 'Santro', 'Tiago', 'Celerio X', 'Ignis', 'Triber',
                 'Rio', 'Etios Liva', 'Micra Active', 'Bolt', 'Xcent Prime',
                 'Dzire Tour', 'Elite I20', 'Aura', 'Polo', 'Dzire', 'Freestyle',
                 'Ameo', 'Aspire', 'Platinum Etios', 'Etios Cross', 'Verito Vibe',
                 'Urban Cross', 'Glanza', 'Avventura', 'Jazz', 'Compass Trailhawk',
                 'Mu-X', 'Alturas G4', 'Tiguan', 'Cr-V', 'Superb Sportline', 'A3',
                 'Mercedes-Benz B-Class', 'Mercedes-Benz Cla-Class', 'Kodiaq',
                 'Avanti', 'Q3', 'Cooper 5 Door', 'Convertible', 'Xc40', 'Clubman',
                 'A4', 'John Cooper Works', 'Xe', 'Xf', 'A3 Cabriolet', 'A6', 'X3',
                 'Discovery Sport', 'S90', 'Qute (Re60)', 'Alto', 'S-Presso',
                 'Celerio', 'Grand I10 Prime', 'Kuv100 Nxt', 'Swift', 'Altroz',
                 'Extreme', 'Tigor', 'Zest', 'Amaze', 'Gypsy', 'Venue', 'Nexon',
                 'Linea', 'Bolero Power Plus', 'Vitara Brezza', 'I20 Active',
                 'Ecosport', 'Duster', 'Verna', 'Xuv300', 'Lodgy', 'Vento',
                 'E2O Plus', 'Tigor Ev', 'Brv', 'Thar', 'Gurkha', 'Xl6',
                 'Abarth Avventura', 'Tuv300 Plus', 'Marazzo', 'Scorpio',
                 'Monte Carlo', 'Xuv500', 'E Verito', 'Hexa', 'Innova Crysta',
                 'Compass', 'Corolla Altis', 'Civic', 'Zs Ev', 'Carnival', 'Superb',
                 'V40', 'Fortuner', 'Endeavour', 'Cooper 3 Door', 'Kodiaq Scout',
                 'X1', 'S60', '3-Series', 'S60 Cross Country', 'Q5', 'Range Evoque',
                 'Mercedes-Benz E-Class', 'Xc60', 'X4', 'Wrangler',
                 'Mercedes-Benz C-Class Cabriolet', 'Omni', 'Go+', 'Punto Evo Pure',
                 'Figo', 'Baleno', 'Grand I10', 'Linea Classic', 'Sunny', 'Ertiga',
                 'Baleno Rs', 'Wr-V', 'Tuv300', 'S-Cross', 'Captur', 'Xylo',
                 'Seltos', 'Terrano', 'Safari Storme', 'Hector', 'Nexon Ev',
                 'Elantra', 'Tucson', 'Passat', 'Mercedes-Benz A-Class',
                 'V40 Cross Country', 'Countryman', 'Mercedes-Benz C-Class',
                 'Prius', 'Es', 'Nx 300H', 'F-Pace', 'Alto 800 Tour',
                 'Grand I10 Nios', 'Xcent', 'Micra', 'Bolero', 'Ciaz', 'Rapid',
                 'Abarth Punto', 'Creta', 'Harrier', 'Dmax V-Cross', 'Outlander',
                 'Mercedes-Benz Gla-Class', 'Accord Hybrid', '5-Series', '6-Series',
                 'Wagon R', 'Tiago Nrg', 'Nuvosport', 'Kicks', 'Winger',
                 'Kona Electric', 'Camry', 'A5', 'Punto Evo', 'Yaris', 'Octavia',
                 'Mercedes-Benz Glc', 'Verito', 'Pajero Sport', 'City']
        return model
    elif (list == 'type'):
        type = ['Hatchback', 'MPV', 'MUV', 'SUV', 'Sedan',
                'Crossover', 'Coupe', 'Convertible', 'Sports', 'Pick-up']
        return type
    else:
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


def modelPrice(Price, Seating_Capacity, Mileage, Fuel_Type, Body_Type):
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
    type = get_carModel_Gen('type')
    fuel = get_carModel_Gen('fuel')
    model = get_carModel_Gen('model')

    t_type = find_key(type, Body_Type)
    t_fuel = find_key(fuel, Fuel_Type)
    
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


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
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
