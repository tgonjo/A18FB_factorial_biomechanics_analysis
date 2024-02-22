import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import math
from sklearn.linear_model import LinearRegression
from io import BytesIO


## Functions ##

def fill_data_gaps(data, max_consecutive_gaps=3):
        # Iterate over each column in the DataFrame
        for column in data.columns:
            # Check if the column contains numerical data
            if pd.api.types.is_numeric_dtype(data[column]):
                # Check for gaps in the data column
                mask = data[column].isnull()
                consecutive_gaps = mask.astype(int).groupby(mask.ne(mask.shift()).cumsum()).cumsum()

                # Check if there are more than four consecutive gaps
                if consecutive_gaps.max() > max_consecutive_gaps:
                    # Create a simple Tkinter GUI to display the error message
                    st.error(f"Gap check 1: Error! More than {max_consecutive_gaps} consecutive gaps detected in column {column}. Unable to fill. Please go back to Kinovea/Tracker and track the missing points for {column}")
                    st.subheader("Ignore any error messages appear below. Refresh this page and try again once you fix the issue described in 'Gap check 1'.")                          
                    st.stop()
                # Apply linear interpolation to fill gaps up to three rows in sequence
                data[column] = data[column].interpolate(method='linear', limit_area='inside')
                
        st.success('Gap check 1: Success! Less than 4 consecutive gaps in your data')
                
def check_and_continue():
    # Check for NaN values in the first and last rows
    if df_raw.iloc[0].isna().any() or df_raw.iloc[-1].isna().any():
        
        st.error('Gap check 2: Error! There are some blank cells in the first or last row of your Excel data')
    
        st.subheader("Ignore any error messages appear below. Refresh this page and try again once you fix the issue described in 'Gap check 2'.")                           
        st.stop()
    else:
        st.success('Gap check 2: Success! No blank cells in the first and last data rows')
        
def apply_low_pass_filter(data, cutoff_freq, fs):
    # Calculate the filter coefficients
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

#Generate a residual analysis function

def residual_analysis(data_column,time_column):
    
    sample_rate = 1 / (time_column[1] - time_column[0])
    
    smooth_data = pd.DataFrame(index=np.arange(len(data_column)), columns=np.arange(100))
    MS = pd.DataFrame(index=np.arange(1), columns=np.arange(100))
    RMS = pd.DataFrame(index=np.arange(1), columns=np.arange(100))


    # Calculate the corresponding frequencies
    frequencies = np.fft.fftfreq(len(data_column), d=1/sample_rate)

    # Calculate the indices for the one-sided spectrum
    half_length = len(data_column) // 2


    freq = frequencies[:half_length]

    # Generate time indices for the original data
    time_indices = np.linspace(0, 1, len(freq))

    # Generate time indices for the expanded data
    expanded_time_indices = np.linspace(0, 1, 101)

    # Perform linear interpolation to expand the data
    freq_int = interp1d(time_indices, freq, kind='linear')(expanded_time_indices)

    #Smooth the data using each frequency and calculate RMS
    for j in np.arange(100)+1:
        
        smooth_data.iloc[:,j-1] = apply_low_pass_filter(data_column, freq_int[j], sample_rate)
        
        
        MSdat = smooth_data.iloc[:,j-1] - data_column.reset_index().iloc[:,1]
        MS.iloc[0,j-1] = MSdat.mean()**2
        
        RMS.iloc[0,j-1] = math.sqrt(MS.iloc[0,j-1])

    # Establish a linear regression line for the residual analysis
    freq_data = RMS.iloc[0, -50:].reset_index().iloc[:,1]
    xx = np.arange(freq_data.shape[0]).astype(float)
    x = pd.Series(xx+50)

    x_reshaped = x.values.reshape(-1, 1)
    y_reshaped = freq_data.values.reshape(-1, 1)

    regression_model = LinearRegression()
    regression_model.fit(x_reshaped, y_reshaped)

    intercept = regression_model.intercept_[0]

    ind = min(range(len(RMS.iloc[0, :])), key=lambda k: abs(RMS.iloc[0, k]-intercept))
    

    #Smoothed output data
    outdat = smooth_data.iloc[:,ind]
    best_freq = freq_int[ind]
    
    return outdat, best_freq

        

## Main script

image = Image.open('test_image.png')
st.image(image,width=800)
st.title('CoM & Net Force calculator for A18FB Assessment 1')
st.caption('This web app was designed to calculate the 2D centre of mass (CoM) location and net force based on 2D coordinates of six joints (Ankle, Knee, Hip, Shoulder, Elbow, and Wrist).')

st.subheader('What does this web app do?')
st.text('1. It smoothes your data to minimise the noise in your joint coordinates.')
st.text('2. Based on the smoothed data, it calculates the CoM coordinates and net force.')
st.text('3. It exports the output data (smoothed coordinates, CoM coordinates, and net force) as an Excel file.')

st.text('')
st.subheader('Before you start, make sure you have:')
st.text('1. digitised all six joints and exported the coordinate data.')
st.text('2. converted the unit if necessary (e.g. milli-second to second, cm to m, etc.')
st.text('3. checked there are no blank cells at the first and last rows.')
st.text('4. checked there are no more than 3 consecutive blank cells at each column.')
st.text('5. pasted your time and coordinate data onto the template Excel file and saved it.')

deleva_url = "https://www.sciencedirect.com/science/article/pii/0021929095001786?via%3Dihub"

template_columns = ['Time', 'WristX', 'WristY', 'ElbowX', 'ElbowY', 'ShoulderX',
       'ShoulderY', 'HipX', 'HipY', 'KneeX', 'KneeY', 'AnkleX', 'AnkleY']

temp_df=pd.DataFrame(columns=template_columns)

temp_df.to_excel(buf := BytesIO(), index=False)

st.text('')
st.download_button(
    "Download the template Excel file here",
    buf.getvalue(),
    "Coordinates_template.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
st.text('IMPORTANT!!: Do not change the header of each column (e.g. Shoulder X, HipY, etc.) - the app will not work if you change them')
st.text('')

st.subheader("Input participant's information and upload your Excel file")

with st.form(key='mass_form'):
    
    entered_mass = st.number_input("Participant's body mass (kg).")
           
    selected_sex = st.radio("Participant's sex", ("Male", "Female", "Prefer not to say"))
    st.write("The sex information is necessary to select the CoM calculation method proposed by De Leva (1996) [link to the article](%s). If you select 'Prefer not to say', the average of the male and female models will be used for CoM calculation." % deleva_url)
        
    submit1 = st.form_submit_button("Confirm participant's mass and sex")
    
    if submit1:
        st.text(f'Body mass: {entered_mass} kg')
        st.text(f'Sex: {selected_sex}')
        
uploaded_file = st.file_uploader('Upload your Excel file', type='xlsx', accept_multiple_files=False, key ='xlsx')

if selected_sex is None:
    st.error("Participant's sex is not confirmed")         
    st.stop()
    
elif entered_mass == 0:
    st.error("Input participant's mass and sex (don't forget to click the confirm button)")         
    st.stop()
    
elif uploaded_file is None:
    st.error('Upload your Excel file')         
    st.stop()

with st.form(key='upload_form'):

    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file)
        fig,ax = plt.subplots()
        ax.plot(df_raw['Time']-df_raw['Time'].iloc[0], df_raw['HipY'])
        ax.set_title('This is your Hip(Y) time-series plot. Do the time and displacement values look like seconds and metres?')
        ax.set_xlabel('Time')
        ax.set_ylabel('Vertical Displacement')
        st.pyplot(fig)
        st.text('If either the time or displacement unit seems wrong, go back to your Excel file and fix it')
        
            
        df_freq = pd.DataFrame(columns = df_raw.columns[1:],index=range(1))
        
        fill_data_gaps(df_raw, max_consecutive_gaps=3)
        
        check_and_continue()
        
        st.text("If the two gap check functions are successful and there are no mistakes in the units, click 'Proceed'")
        
        submit2 = st.form_submit_button('Proceed (it might take a bit to start processing data - be patient!)')
        
progress_placeholder = st.empty()
        
if submit2:
    
    df = df_raw.copy()

    for column in df_raw.columns[1:]:
        
        df[column], df_freq[column][0] = residual_analysis(df_raw[column],df_raw['Time'])


    BSP_M = [['lower_arm',1.62, 45.74],['upper_arm',2.71,57.72],['trunk',43.46,43.10],['thigh',14.16,40.95],['shank',4.33,43.95]]

    BSP_F = [['lower_arm',1.38, 45.59],['upper_arm',2.55,57.54],['trunk',42.57,37.82],['thigh',14.78,36.12],['shank',4.81,43.52]]

    df_bsp_m = pd.DataFrame(BSP_M, columns=['Segment', 'Percent_mass','CoM_location'])
    df_bsp_f = pd.DataFrame(BSP_F, columns=['Segment', 'Percent_mass','CoM_location'])
    df_bsp_m.set_index('Segment', inplace=True)
    df_bsp_f.set_index('Segment', inplace=True)

    df_bsp_m = df_bsp_m/100
    df_bsp_f = df_bsp_f/100

    df_bsp_pns = (df_bsp_m + df_bsp_f)/2

    # df_com = pd.DataFrame(columns =['CoM_X', 'CoM_Y'], index = range(len(df)))

    if selected_sex == 'Male':
        com_la_x = df['ElbowX']+(df['WristX']-df['ElbowX'])*df_bsp_m.loc['lower_arm']['CoM_location']
        com_ua_x = df['ShoulderX']+(df['ElbowX']-df['ShoulderX'])*df_bsp_m.loc['upper_arm']['CoM_location']
        com_trunk_x = df['ShoulderX']+(df['HipX']-df['ShoulderX'])*df_bsp_m.loc['trunk']['CoM_location']
        com_thigh_x = df['HipX']+(df['KneeX']-df['HipX'])*df_bsp_m.loc['thigh']['CoM_location']
        com_shank_x = df['KneeX']+(df['AnkleX']-df['KneeX'])*df_bsp_m.loc['shank']['CoM_location']
        
        com_la_y = df['ElbowY']+(df['WristY']-df['ElbowY'])*df_bsp_m.loc['lower_arm']['CoM_location']
        com_ua_y = df['ShoulderY']+(df['ElbowY']-df['ShoulderY'])*df_bsp_m.loc['upper_arm']['CoM_location']
        com_trunk_y = df['ShoulderY']+(df['HipY']-df['ShoulderY'])*df_bsp_m.loc['trunk']['CoM_location']
        com_thigh_y = df['HipY']+(df['KneeY']-df['HipY'])*df_bsp_m.loc['thigh']['CoM_location']
        com_shank_y = df['KneeY']+(df['AnkleY']-df['KneeY'])*df_bsp_m.loc['shank']['CoM_location']
        
        com_wb_x = (com_la_x*df_bsp_m['Percent_mass']['lower_arm']+com_ua_x*df_bsp_m['Percent_mass']['upper_arm'] \
        + com_trunk_x*df_bsp_m['Percent_mass']['trunk']+com_thigh_x*df_bsp_m['Percent_mass']['thigh']+com_shank_x*df_bsp_m['Percent_mass']['shank'])\
        / df_bsp_m['Percent_mass'].sum()
        
        com_wb_y = (com_la_y*df_bsp_m['Percent_mass']['lower_arm']+com_ua_y*df_bsp_m['Percent_mass']['upper_arm'] \
        + com_trunk_y*df_bsp_m['Percent_mass']['trunk']+com_thigh_y*df_bsp_m['Percent_mass']['thigh']+com_shank_y*df_bsp_m['Percent_mass']['shank'])\
        / df_bsp_m['Percent_mass'].sum()
        
    elif selected_sex == 'Female':
        com_la_x = df['ElbowX']+(df['WristX']-df['ElbowX'])*df_bsp_f.loc['lower_arm']['CoM_location']
        com_ua_x = df['ShoulderX']+(df['ElbowX']-df['ShoulderX'])*df_bsp_f.loc['upper_arm']['CoM_location']
        com_trunk_x = df['ShoulderX']+(df['HipX']-df['ShoulderX'])*df_bsp_f.loc['trunk']['CoM_location']
        com_thigh_x = df['HipX']+(df['KneeX']-df['HipX'])*df_bsp_f.loc['thigh']['CoM_location']
        com_shank_x = df['KneeX']+(df['AnkleX']-df['KneeX'])*df_bsp_f.loc['shank']['CoM_location']
        
        com_la_y = df['ElbowY']+(df['WristY']-df['ElbowY'])*df_bsp_f.loc['lower_arm']['CoM_location']
        com_ua_y = df['ShoulderY']+(df['ElbowY']-df['ShoulderY'])*df_bsp_f.loc['upper_arm']['CoM_location']
        com_trunk_y = df['ShoulderY']+(df['HipY']-df['ShoulderY'])*df_bsp_f.loc['trunk']['CoM_location']
        com_thigh_y = df['HipY']+(df['KneeY']-df['HipY'])*df_bsp_f.loc['thigh']['CoM_location']
        com_shank_y = df['KneeY']+(df['AnkleY']-df['KneeY'])*df_bsp_f.loc['shank']['CoM_location']
        
        com_wb_x = (com_la_x*df_bsp_f['Percent_mass']['lower_arm']+com_ua_x*df_bsp_f['Percent_mass']['upper_arm'] \
        + com_trunk_x*df_bsp_f['Percent_mass']['trunk']+com_thigh_x*df_bsp_f['Percent_mass']['thigh']+com_shank_x*df_bsp_f['Percent_mass']['shank'])\
        / df_bsp_f['Percent_mass'].sum()
        
        com_wb_y = (com_la_y*df_bsp_f['Percent_mass']['lower_arm']+com_ua_y*df_bsp_f['Percent_mass']['upper_arm'] \
        + com_trunk_y*df_bsp_f['Percent_mass']['trunk']+com_thigh_y*df_bsp_f['Percent_mass']['thigh']+com_shank_y*df_bsp_f['Percent_mass']['shank'])\
        / df_bsp_f['Percent_mass'].sum()
        
    else:
        com_la_x = df['ElbowX']+(df['WristX']-df['ElbowX'])*df_bsp_pns.loc['lower_arm']['CoM_location']
        com_ua_x = df['ShoulderX']+(df['ElbowX']-df['ShoulderX'])*df_bsp_pns.loc['upper_arm']['CoM_location']
        com_trunk_x = df['ShoulderX']+(df['HipX']-df['ShoulderX'])*df_bsp_pns.loc['trunk']['CoM_location']
        com_thigh_x = df['HipX']+(df['KneeX']-df['HipX'])*df_bsp_pns.loc['thigh']['CoM_location']
        com_shank_x = df['KneeX']+(df['AnkleX']-df['KneeX'])*df_bsp_pns.loc['shank']['CoM_location']
        
        com_la_y = df['ElbowY']+(df['WristY']-df['ElbowY'])*df_bsp_pns.loc['lower_arm']['CoM_location']
        com_ua_y = df['ShoulderY']+(df['ElbowY']-df['ShoulderY'])*df_bsp_pns.loc['upper_arm']['CoM_location']
        com_trunk_y = df['ShoulderY']+(df['HipY']-df['ShoulderY'])*df_bsp_pns.loc['trunk']['CoM_location']
        com_thigh_y = df['HipY']+(df['KneeY']-df['HipY'])*df_bsp_pns.loc['thigh']['CoM_location']
        com_shank_y = df['KneeY']+(df['AnkleY']-df['KneeY'])*df_bsp_pns.loc['shank']['CoM_location']
        
        com_wb_x = (com_la_x*df_bsp_pns['Percent_mass']['lower_arm']+com_ua_x*df_bsp_pns['Percent_mass']['upper_arm'] \
        + com_trunk_x*df_bsp_pns['Percent_mass']['trunk']+com_thigh_x*df_bsp_pns['Percent_mass']['thigh']+com_shank_x*df_bsp_pns['Percent_mass']['shank'])\
        / df_bsp_pns['Percent_mass'].sum()
        
        com_wb_y = (com_la_y*df_bsp_pns['Percent_mass']['lower_arm']+com_ua_y*df_bsp_pns['Percent_mass']['upper_arm'] \
        + com_trunk_y*df_bsp_pns['Percent_mass']['trunk']+com_thigh_y*df_bsp_pns['Percent_mass']['thigh']+com_shank_y*df_bsp_pns['Percent_mass']['shank'])\
        / df_bsp_pns['Percent_mass'].sum()

    df['CoM_X'] = com_wb_x
    df['CoM_Y'] = com_wb_y

    com_wb_x_v = com_wb_x.copy()
    com_wb_y_v = com_wb_y.copy()
    com_wb_x_a = com_wb_x.copy()
    com_wb_y_a = com_wb_y.copy()
    
    for i in range(len(com_wb_x)-2):
        
        com_wb_x_v.iloc[i+1] = (com_wb_x.iloc[i+2]-com_wb_x.iloc[i])/(df['Time'].iloc[i+2]-df['Time'].iloc[i])
        com_wb_y_v.iloc[i+1] = (com_wb_y.iloc[i+2]-com_wb_y.iloc[i])/(df['Time'].iloc[i+2]-df['Time'].iloc[i])

    com_wb_x_v.iloc[0] = (com_wb_x_v.iloc[1] + com_wb_x_v.iloc[2])-com_wb_x_v.iloc[3]
    com_wb_x_v.iloc[-1] = (com_wb_x_v.iloc[-2] + com_wb_x_v.iloc[-3])-com_wb_x_v.iloc[-4]

    com_wb_y_v.iloc[0] = (com_wb_y_v.iloc[1] + com_wb_y_v.iloc[2])-com_wb_y_v.iloc[3]
    com_wb_y_v.iloc[-1] = (com_wb_y_v.iloc[-2] + com_wb_y_v.iloc[-3])-com_wb_y_v.iloc[-4]


    for i in range(len(com_wb_x)-2):
        
        com_wb_x_a.iloc[i+1] = (com_wb_x_v.iloc[i+2]-com_wb_x_v.iloc[i])/(df['Time'].iloc[i+2]-df['Time'].iloc[i])
        com_wb_y_a.iloc[i+1] = (com_wb_y_v.iloc[i+2]-com_wb_y_v.iloc[i])/(df['Time'].iloc[i+2]-df['Time'].iloc[i])

    com_wb_x_a.iloc[0] = (com_wb_x_a.iloc[1] + com_wb_x_a.iloc[2])-com_wb_x_a.iloc[3]
    com_wb_x_a.iloc[-1] = (com_wb_x_a.iloc[-2] + com_wb_x_a.iloc[-3])-com_wb_x_a.iloc[-4]

    com_wb_y_a.iloc[0] = (com_wb_y_a.iloc[1] + com_wb_y_a.iloc[2])-com_wb_y_a.iloc[3]
    com_wb_y_a.iloc[-1] = (com_wb_y_a.iloc[-2] + com_wb_y_a.iloc[-3])-com_wb_y_a.iloc[-4]

    net_force_x = com_wb_x_a*entered_mass
    net_force_y = com_wb_y_a*entered_mass

    df[' '] = pd.Series([None] * len(net_force_x ))
    df['Net_Force_X'] = net_force_x
    df['Net_Force_Y'] = net_force_y
            
    progress_placeholder.write("Processing data...")
    # Your existing code for data processing and calculations
    
    df_len = len(df)

    df_int = math.floor(df_len/9)

    fig2, axs = plt.subplots(1, 10, figsize=(15, 3))

    X = df[['WristX','ElbowX','ShoulderX','HipX','KneeX','AnkleX']]
    Y = df[['WristY','ElbowY','ShoulderY','HipY','KneeY','AnkleY']]


    xmin = (X.min()-0.5).min()
    xmax = (X.max()+0.5).max()

    ymin = (Y.min()-0.05).min()
    ymax = (Y.max()+0.1).max()

    for num, i in enumerate(range(0, df_len,df_int)):
        
        axs[num].plot(X.iloc[i],Y.iloc[i])
        axs[num].set_xlim(left=xmin, right=xmax)
        axs[num].set_ylim(bottom=ymin, top=ymax)
        axs[num].set_title(f'Frame {i}')
        
        CM_x = df['CoM_X'].iloc[i] 
        CM_y = df['CoM_Y'].iloc[i] 
        
        axs[num].scatter(CM_x, CM_y, color='red', marker='o')
        
        if i>0:
            axs[num].set_yticks([])
        
    st.pyplot(fig2)
    st.text('Stick figures of the processed jump movement (the red dots show the location of CoM)')

    df.to_excel(buf := BytesIO(), index=False)
    
    # Display the download button after processing is complete
    st.success("Processing complete!")
    st.text('Does the series of stick figures look like a jump movement? If yes:')
    st.download_button(
        "Download your output file",
        buf.getvalue(),
        "outdat_with_com_nf.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.text('If not, check and fix your Excel file, reload this pafe, and run the analysis again')
    
    progress_placeholder.empty()
        
            
        
        
