from sklearn.linear_model import LinearRegression
from pylab import * 
from pandas import *
from sklearn.metrics import root_mean_squared_error as rmse ,r2_score
from sklearn.model_selection import train_test_split as tts 
import streamlit as st 
 
# The air quality data is imported 
air_quality=read_csv("final_dataset.csv")

# Air Quality data is printed here
print(air_quality)
print("Air Qualiry Shape :",air_quality.shape)

# lets find when was the AQI max and When it was Minimum AQI And Print the data 

print("The Data when AQI was MAX and MIN :\n",air_quality.iloc[[air_quality["AQI"].idxmax(),air_quality["AQI"].idxmin()]])

# Find the relationship between the AQI and Other factors

print(air_quality.drop(["Date","Days","Month","Year","Holidays_Count"],axis=1).corr())


# Creating the arrays for the columns 

PM2=air_quality["PM2.5"].values
AQI=air_quality["AQI"].values
PM10=air_quality["PM10"].values
NO2=air_quality["NO2"].values
SO2=air_quality["SO2"].values
CO=air_quality["CO"].values
Ozone=air_quality['Ozone'].values


# Plotting the scatter Plot 
gs=GridSpec(3,3)
fig=figure(figsize=(6,6))

fig.add_subplot(gs[0,:1])
ylabel("PM2")
scatter(PM2,AQI)

fig.add_subplot(gs[1,:1])
ylabel("PM10")
scatter(PM10,AQI,color="k",alpha=0.7)

fig.add_subplot(gs[2,:1])
ylabel("NO2")
scatter(NO2,AQI,color="cyan")

fig.add_subplot(gs[0,1:])
ylabel("SO2")
scatter(SO2,AQI)

fig.add_subplot(gs[1,1:])
ylabel("CO")
scatter(CO,AQI)

fig.add_subplot(gs[2,1:])
ylabel("Ozone")
scatter(Ozone,AQI)
tight_layout()
show()

model=LinearRegression()

# Creating Series

pm2=PM2.reshape(1461,1)
print(pm2.shape)

pm10=air_quality["PM10"]
so2=air_quality["SO2"]
no2=air_quality["NO2"]
co=air_quality["CO"]
ozone=air_quality['Ozone']

# This is the model working code and the training of the model is done using this code 
input_test,input_train,targets_test,targets_train=tts(air_quality[["PM2.5","PM10","CO"]],air_quality["AQI"],test_size=0.2)
model.fit(input_train,targets_train)

# This is the Prediction code 
st.title("AQI Predictor ")
st.dataframe(air_quality,use_container_width=True)
pm=st.number_input('Enter the data of PM2.5 : ')
pm2=st.number_input("Enter the data of PM10:")
co=st.number_input("Enter the data of CO :")
pre_data=[[pm,pm2,co]]
predictions=model.predict(pre_data)
if st.button("Predict"):
    st.write("The predicted AQI for today is ",predictions[0])
st.pyplot(fig)

# predict_df=DataFrame({"Predictions":predictions,"Observations":targets_test})
# print(predict_df)

# This is the Root Mean Square Error of the model 
# This one is function that i created
# def rmse(x,y):
#     rms=mean(sqrt(square(x-y)))
#     return rms
# loss=rmse(targets_test,predictions)
# print("Loss :",loss)

# # This one is a inbuild function created by sklearn 
# print("Loss :",rmse(targets_test,predictions))


# # This is where we predict the values 
# print('Predicted AQI :',predictions[0])


# # To check the R2 score 
# print("R2 square score :",r2_score(targets_test,predictions))

# # To check the values of the score and check the linear relationship they have 

# print("The Slope :",model.coef_)

# # To check the intercept of the model 

# print("The Intercept :",model.intercept_)

