from flask import Flask,request,render_template
import numpy as np
import matplotlib.pyplot as plt
import pickle

dtr=pickle.load(open('dtr.pkl','rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        State_Name=request.form['State_Name']
        District_Name=request.form['District_Name']
        Crop_Year=request.form['Crop_Year']
        Season=request.form['Season']
        Area=request.form['Area']

        crops = ['Arecanut', 'Other Kharif pulses', 'Rice', 'Banana', 'Cashewnut',
        'Coconut ', 'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca',
        'Black pepper', 'Dry chillies', 'other oilseeds', 'Turmeric',
        'Maize', 'Moong(Green Gram)', 'Urad', 'Arhar/Tur', 'Groundnut',
        'Sunflower', 'Bajra', 'Castor seed', 'Cotton(lint)', 'Horse-gram',
        'Jowar', 'Korra', 'Ragi', 'Tobacco', 'Gram', 'Wheat', 'Masoor',
        'Sesamum', 'Linseed', 'Safflower', 'Onion', 'other misc. pulses',
        'Samai', 'Small millets', 'Coriander', 'Potato',
        'Other  Rabi pulses', 'Soyabean', 'Beans & Mutter(Vegetable)',
        'Bhindi', 'Brinjal', 'Citrus Fruit', 'Cucumber', 'Grapes', 'Mango',
        'Orange', 'other fibres', 'Other Fresh Fruits', 'Other Vegetables',
        'Papaya', 'Pome Fruit', 'Tomato', 'Mesta', 'Cowpea(Lobia)',
        'Lemon', 'Pome Granet', 'Sapota', 'Cabbage', 'Rapeseed &Mustard',
        'Peas  (vegetable)', 'Niger seed', 'Bottle Gourd', 'Varagu',
        'Garlic', 'Ginger', 'Oilseeds total', 'Pulses total', 'Jute',
        'Peas & beans (Pulses)', 'Blackgram', 'Paddy', 'Pineapple',
        'Barley', 'Sannhamp', 'Khesari', 'Guar seed', 'Moth',
        'Other Cereals & Millets', 'Cond-spcs other', 'Turnip', 'Carrot',
        'Redish', 'Arcanut (Processed)', 'Atcanut (Raw)',
        'Cashewnut Processed', 'Cashewnut Raw', 'Cardamom', 'Rubber',
        'Bitter Gourd', 'Drum Stick', 'Jack Fruit', 'Snak Guard', 'Tea',
        'Coffee', 'Cauliflower', 'Other Citrus Fruit', 'Water Melon',
        'Total foodgrain', 'Kapas', 'Colocosia', 'Lentil', 'Bean',
        'Jobster', 'Perilla', 'Rajmash Kholar', 'Ricebean (nagadal)',
        'Ash Gourd', 'Beet Root', 'Lab-Lab', 'Ribed Guard', 'Yam',
        'Pump Kin', 'Apple', 'Peach', 'Pear', 'Plums', 'Litchi', 'Ber',
        'Other Dry Fruit', 'Jute & mesta']

        predicted_values = {}
        for crop in crops:
           features = np.array([[State_Name, District_Name, Crop_Year, Season, crop, Area]])
           t_features = preprocessor.transform(features)
           p_v=(dtr.predict(t_features).reshape(1, -1))
           predicted_values[crop] = p_v.item()

        predicted_values = sorted(predicted_values.items(), key=lambda x: x[1], reverse=True)
        keys = predicted_values.keys()
        values = predicted_values.values()
        plt.xlabel('Crops')
        plt.ylabel('Production')
        plt.title('Crop-Yield-Prediction')
        plt.bar(keys, values)
        plt.savefig('C:/Users/hp/Desktop/python_here/graph_images')
        return render_template('index.html', predicted_values=predicted_values)


if __name__ == '__main__':
    app.run(debug=True)
