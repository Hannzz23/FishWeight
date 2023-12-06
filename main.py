import pickle
import streamlit as st
import pandas as pd
import os
import numpy as np
import altair as alt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset_ikan.csv")
#X = data.drop(['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width'], axis=1)
# X = data.drop(['Species'], axis=1)
# y = data['Weight']
# y = y.astype(int)
X = data[['Species', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]
y = data['Weight']
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

model_regresi = DecisionTreeRegressor()
model_regresi.fit(X_train, y_train)

filename = 'model_fish_weight123.sav'
pickle.dump(model_regresi,open(filename,'wb'))
model = pickle.load(open('model_fish_weight123.sav', 'rb'))


def get_svalue(Species):    
    if Species == 'Bream':
        return 1
    elif Species == 'Roach':
        return 2
    elif Species == 'Whitefish':
        return 3
    elif Species == 'Parkki':
        return 4
    elif Species == 'Perch':
        return 5
    elif Species == 'Pike':
        return 6
    elif Species == 'Smelt':
        return 7
    else:
        return Species
        
def load_data():
    data = pd.read_csv("Fish.csv")
    return data

def profil() :
    st.markdown("Semakin Berat Ikannya, Semakin banyak dagingnya, Semakin puas makannya")
    st.image('ikan.jpeg', use_column_width=True)

def Dataset() : 
    st.title("Dataset Fish Weight")
    # menampilkan dataframe
    fish = pd.read_csv("Fish.csv")
    st.dataframe(fish)

def Length() :
    st.write("Grafik Vertical Length")

    chart_Length1 = pd.DataFrame(data, columns=["Length1"])
    st.line_chart(chart_Length1)

    st.write("Grafik Horizontal Length")
    chart_Length2 = pd.DataFrame(data, columns=["Length2"])
    st.line_chart(chart_Length2)

    st.write("Grafik Cross Length")
    chart_Length3 = pd.DataFrame(data, columns=["Length3"])
    st.line_chart(chart_Length3)

# def Comparison() :
    # df = pd.read_csv("Fish.csv")

    # chart1 = pd.DataFrame(df,columns=["Height","Weight"])
    # chart2 = pd.DataFrame(df,columns=["Width","Weight"])

    # st.write("Perbandingan Height dan Weight")
    # plt.figure(figsize=(7,7))
    # plt.scatter(x="Height", y="Weight", data=df)
    # plt.xlabel('Fish Height')
    # plt.ylabel('Fish Weight')
    # plt.title('Height Vs. Weight')
    # st.scatter_chart(chart1)

    # st.write("Perbandingan Width dan Weight")
    # plt.figure(figsize=(7,7))
    # plt.scatter(x="Width", y="Weight", data=df)
    # plt.xlabel('Fish Width')
    # plt.ylabel('Fish Weight')
    # plt.title('Width Vs. Weight')
    # st.scatter_chart(chart2)

def widthcompare(df, width_threshold):
    filtered_df = df[df['Width'] >= width_threshold]

    plt.figure(figsize=(10, 6))
    plt.bar(filtered_df['Species'], filtered_df['Weight'], color='blue')

    plt.xlabel('Species')
    plt.ylabel('Weight')
    plt.title(f'Comparison of Weight for Fish with Width >= {width_threshold}')
    plt.xticks(rotation=45, ha='right')
    st.pyplot()

def heightcompare(df, height_threshold):
    filtered_df = df[df['Height'] >= height_threshold]

    plt.figure(figsize=(10, 6))
    plt.bar(filtered_df['Species'], filtered_df['Weight'], color='blue')

    plt.xlabel('Species')
    plt.ylabel('Weight')
    plt.title(f'Comparison of Weight for Fish with Height >= {height_threshold}')
    plt.xticks(rotation=45, ha='right')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Contoh dataset ikan


# Contoh dataset ikan
  


def Prediksi() :

    Species = st.radio('Pick The Species Fish',['Bream','Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt'])
    Length1 = st.number_input('Vertical Length', 0, 1000)
    Length2 = st.number_input('Diagonal Length', 0, 1000)
    Length3 = st.number_input('Cross Length', 0, 1000)
    Height = st.number_input('Height', 0, 1000)
    Width = st.number_input('Width', 0, 1000)
    Species = get_svalue(Species)

    if st.button('Predict Your Fish Weight'):
        weight_prediction = model.predict([[Species,Length1, Length2, Length3, Height, Width]])
        weight_prediction_str = np.array(weight_prediction)
        weight_prediction_float = float(weight_prediction_str[0])
        weight_prediction_formatted = "{:,.2f}".format(weight_prediction_float)
        st.markdown(f'Berat Ikan Tersebut : {weight_prediction_formatted} gram')


def main():

    st.sidebar.title("Menu")
    menu = st.sidebar.selectbox("Pilih Menu", ["Deskripsi", "Dataset", "Length of Fish", "Weight vs Width","Weight vs Height", "Prediksi"])


    if menu == "Deskripsi":
        st.title ("Deskripsi Ikan") 
        profil()

    elif menu == "Dataset":
        st.title("Tampilan Dataset")
        
        Dataset()

    elif menu == "Length of Fish":
        st.title("Tampilan Grafik Length of Fish")
        data = load_data()
        Length()

    elif menu == "Weight vs Width":
        data = load_data()

        df = pd.DataFrame(data)

        # Menampilkan input threshold lebar di Streamlit
        width_threshold = st.slider('Select Width Threshold', min_value=df['Width'].min(), max_value=df['Width'].max(), value=df['Width'].min())

        # Menampilkan dataframe di Streamlit
        st.title('Fish Dataset Comparison')
        st.dataframe(df)

        # Menampilkan bar chart di Streamlit berdasarkan threshold lebar
        st.title(f'Comparison of Weight for Fish with Width >= {width_threshold}')
        widthcompare(df, width_threshold)

    elif menu == "Weight vs Height":
        data = load_data()

        df = pd.DataFrame(data)

        # Menampilkan input threshold lebar di Streamlit
        height_threshold = st.slider('Select height Threshold', min_value=df['Height'].min(), max_value=df['Height'].max(), value=df['Height'].min())

        # Menampilkan dataframe di Streamlit
        st.title('Fish Dataset Comparison')
        st.dataframe(df)

        # Menampilkan bar chart di Streamlit berdasarkan threshold lebar
        st.title(f'Comparison of Weight for Fish with Height >= {height_threshold}')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        heightcompare(df, height_threshold)

    elif menu == "Prediksi" :
        st.title("Prediksikan Berat Ikanmu!!!")
        Prediksi()


if __name__ == "__main__":
    main()
