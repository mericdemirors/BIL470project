import streamlit as st
import pandas as pd
import pickle

st.write("""# Simple Movie Prediction App\nThis app predicts the movie revenue!""")
st.sidebar.header('Movie information')

# Taking input ---------------------------------------------------------------------------------------------------
def user_input_features():
    input_Year= st.sidebar.selectbox('Year', range(1990, 2021))
    input_Rating = st.sidebar.slider('Rating', min_value=float(0.0), value=float(5.0), max_value=float(10.0), step=0.1)
    input_1 = st.sidebar.number_input('Number of votes as 1', value=3000, format="%i")
    input_2 = st.sidebar.number_input('Number of votes as 2', value=3000, format="%i")
    input_3 = st.sidebar.number_input('Number of votes as 3', value=3000, format="%i")
    input_4 = st.sidebar.number_input('Number of votes as 4', value=3000, format="%i")
    input_5 = st.sidebar.number_input('Number of votes as 5', value=3000, format="%i")
    input_6 = st.sidebar.number_input('Number of votes as 6', value=3000, format="%i")
    input_7 = st.sidebar.number_input('Number of votes as 7', value=3000, format="%i")
    input_8 = st.sidebar.number_input('Number of votes as 8', value=3000, format="%i")
    input_9 = st.sidebar.number_input('Number of votes as 9', value=3000, format="%i")
    input_10 = st.sidebar.number_input('Number of votes as 10', value=3000, format="%i")
    data = {
        'Year' : input_Year,
        'Rating' : input_Rating,
        'Votes' : input_1 + input_2 + input_3 + input_4 + input_5 + input_6 + input_7 + input_8 + input_9 + input_10,
        '1': input_1, '2': input_2, '3': input_3, '4': input_4, '5': input_5, '6': input_6, '7': input_7, '8': input_8, '9': input_9, '10': input_10}
    return pd.DataFrame(data, index=[0])

raw_input = user_input_features()

st.subheader('Inputted Movie informations')
st.write(raw_input)
# Taking input ---------------------------------------------------------------------------------------------------


# Scaling input ---------------------------------------------------------------------------------------------------
def scale_raw_input(data):
    import numpy as np
    year_revenue_dict = {1990: 0.7658344741262136, 1991: 0.6158904723529411, 1992: 0.5810284048958334, 1993: 0.5947457455973276, 1994: 0.5941740310769231, 1995: 0.6217016949917985, 1996: 0.6502561518881119, 1997: 0.6338443119205298, 1998: 0.8677550960544218, 1999: 0.6733860998742138, 2000: 0.7771750025433526, 2001: 0.7466757578888888, 2002: 0.708450799753397, 2003: 0.7759085865470852, 2004: 0.8238424626760563, 2005: 0.782264222, 2006: 0.7498834795081967, 2007: 0.6502655863192183, 2008: 0.7055149379672131, 2009: 0.8754953023278688, 2010: 0.6809290582777777, 2011: 0.9059954949253732, 2012: 0.811775069737705, 2013: 0.8438092751023868, 2014: 0.8119720837606839, 2015: 0.8807708197784809, 2016: 0.82369238359375, 2017: 1.0752321504301077, 2018: 0.8333133838255032, 2019: 0.9456890671153846, 2020: 0.3045369520779221, 2021: 0.9344049581962025, 2022: 0.9695750887288135}
    data['Year'] = data['Year'].map(year_revenue_dict)
    data["Rating"] = data["Rating"]/10
    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']] = np.log2(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])

    min_max_scaler = pickle.load(open('MinMaxScaler.pickle', 'rb'))
    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]= min_max_scaler.transform(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])

    return data

scaled_input = scale_raw_input(raw_input)

#st.subheader('Scaled inputs')
#st.write(scaled_input)
# Scaling input ---------------------------------------------------------------------------------------------------




# Predicting  ---------------------------------------------------------------------------------------------------
try:
    LR = pickle.load(open('LR.pickle', 'rb')) # good at middle part
except:
    pass

try:
    SVR_grid = pickle.load(open('SVR_grid.pickle', 'rb'))
except:
    pass
SVR = SVR_grid.best_estimator_ # good around high part

try:
    RFR_grid = pickle.load(open('RFR_grid.pickle', 'rb'))
except:
    pass
RFR = RFR_grid.best_estimator_ # good around high part

# CSV UPLOAD EDİP ONU PREDİCT EDELİM Mİ?
# PREDİCTİON'U 2 ÜZERİ DİYE YAZ

prediction = LR.predict(scaled_input)

st.subheader('Prediction')
st.write(prediction)
# Predicting  ---------------------------------------------------------------------------------------------------