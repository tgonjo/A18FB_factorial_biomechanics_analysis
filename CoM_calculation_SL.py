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
import json


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
                    st.subheader("Ignore any error messages below. Refresh this page and try again once you have fixed the issue described in 'Gap check 1' above.")                          
                    st.stop()
                # Apply linear interpolation to fill gaps up to three rows in sequence
                data[column] = data[column].interpolate(method='linear', limit_area='inside')
                
        st.success('Gap check 1: Success! Less than 4 consecutive gaps in your data')
                
def dot_product_angle(df, i, ref_x, ref_y, point1_x, point1_y, point2_x, point2_y):
    v1_x = df.loc[i,point1_x] - df.loc[i,ref_x]
    v2_x = df.loc[i,point2_x] - df.loc[i,ref_x]
    v1_y = df.loc[i,point1_y] - df.loc[i,ref_y]
    v2_y = df.loc[i,point2_y] - df.loc[i,ref_y]
    
    joint_cos = (v1_x*v2_x + v1_y*v2_y)/(math.sqrt(v1_x**2+v1_y**2)*math.sqrt(v2_x**2+v2_y**2))
    
    joint_angle = math.acos(joint_cos)
    joint_angle_degrees = math.degrees(joint_angle)
    
    return joint_angle_degrees
        
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
st.title('Data converter for A18FB Assessment 1')
st.caption('This web app was designed to convert, smooth, and export data obtained on Factorial Biomechanics.')

st.subheader('What does this web app do?')
st.text('1. It smoothes your data to minimise the noise in your joint and centre of mass (CoM) coordinates and joint angles.')
st.text('2. It calculates the net force acting on the body based on the CoM information.')
st.text('3. It exports selected output data (smoothed coordinates, joint angles, and net force) as an Excel file.')

st.text('')
st.subheader('Before you start, make sure you have:')
st.text('1. calibrated and digitised your video data on the Factorial Biomechanics page;')
st.text('2. sense-checked your data (e.g. was the calibration data succesfully applied?)')
st.text('3. downloaded your data as json file (not only video file)')


st.subheader("Input participant's information and upload your json file")

with st.form(key='mass_form'):
    
    entered_mass = st.number_input("Participant's body mass (kg).")
           
    submit1 = st.form_submit_button("Confirm participant's mass")
    
    if submit1:
        st.text(f'Body mass: {entered_mass} kg')
       
        
uploaded_file = st.file_uploader('Upload your json file', type='json', accept_multiple_files=False, key ='json')

    
if entered_mass == 0:
    st.error("Input participant's mass (don't forget to click the confirm button)")         
    st.stop()
    
elif uploaded_file is None:
    st.error('Upload your json file')         
    st.stop()


progress_placeholder = st.empty()
        
if uploaded_file:
    
    with open(path + file_name, 'r') as file:
        data = json.load(file)

    frames = len(data)

    linear_df = pd.DataFrame(index=range(frames), columns=range(69))
    angular_df = pd.DataFrame(index=range(frames), columns=range(11))
    fps = 180

    for i in range(frames):
        
        for j in range(len(data[i]['keypoints2D'])):
            segment = data[i]['keypoints2D'][j]['name']
            
            if i == 0:
                linear_df = linear_df.rename(columns={j*2+1: segment+'_X (m)'})
                linear_df = linear_df.rename(columns={j*2+2: segment+'_Y (m)'})
                
            linear_df.iloc[i,j*2+1] = data[i]['keypoints2D'][j]['realX']
            linear_df.iloc[i,j*2+2] = data[i]['keypoints2D'][j]['realY']
            
        linear_df.iloc[i,67] = data[i]['com2D']['realX']
        linear_df.iloc[i,68] = data[i]['com2D']['realY']
        
        for k,joint in enumerate(data[i]['angles2D']):
            if i == 0:
                angular_df = angular_df.rename(columns={k+1: joint})
        
        angular_df.loc[i,'rightElbowAngle'] = dot_product_angle(linear_df,i,'right_elbow_X (m)','right_elbow_Y (m)',
                                            'right_wrist_X (m)','right_wrist_Y (m)','right_shoulder_X (m)','right_shoulder_Y (m)')
        
        angular_df.loc[i,'leftElbowAngle'] = dot_product_angle(linear_df,i,'left_elbow_X (m)','left_elbow_Y (m)',
                                            'left_wrist_X (m)','left_wrist_Y (m)','left_shoulder_X (m)','left_shoulder_Y (m)')
            
        angular_df.loc[i,'rightKneeAngle'] = dot_product_angle(linear_df,i,'right_knee_X (m)','right_knee_Y (m)',
                                            'right_ankle_X (m)','right_ankle_Y (m)','right_hip_X (m)','right_hip_Y (m)')
        
        angular_df.loc[i,'leftKneeAngle'] = dot_product_angle(linear_df,i,'left_knee_X (m)','left_knee_Y (m)',
                                            'left_ankle_X (m)','left_ankle_Y (m)','left_hip_X (m)','left_hip_Y (m)')
        
        angular_df.loc[i,'rightShoulderAngle'] = dot_product_angle(linear_df,i,'right_shoulder_X (m)','right_shoulder_Y (m)',
                                            'right_hip_X (m)','right_hip_Y (m)','right_elbow_X (m)','right_elbow_Y (m)')
        
        angular_df.loc[i,'leftShoulderAngle'] = dot_product_angle(linear_df,i,'left_shoulder_X (m)','left_shoulder_Y (m)',
                                            'left_hip_X (m)','left_hip_Y (m)','left_elbow_X (m)','left_elbow_Y (m)')
        
        angular_df.loc[i,'rightHipAngle'] = dot_product_angle(linear_df,i,'right_hip_X (m)','right_hip_Y (m)',
                                            'right_shoulder_X (m)','right_shoulder_Y (m)','right_knee_X (m)','right_knee_Y (m)')
        
        angular_df.loc[i,'leftHipAngle'] = dot_product_angle(linear_df,i,'left_hip_X (m)','left_hip_Y (m)',
                                            'left_shoulder_X (m)','left_shoulder_Y (m)','left_knee_X (m)','left_knee_Y (m)')
        
        angular_df.loc[i,'rightAnkleAngle'] = dot_product_angle(linear_df,i,'right_ankle_X (m)','right_ankle_Y (m)',
                                            'right_knee_X (m)','right_knee_Y (m)','right_foot_index_X (m)','right_foot_index_Y (m)')
        
        angular_df.loc[i,'leftAnkleAngle'] = dot_product_angle(linear_df,i,'left_ankle_X (m)','left_ankle_Y (m)',
                                            'left_knee_X (m)','left_knee_Y (m)','left_foot_index_X (m)','left_foot_index_Y (m)')
        
        
        if i == 0:       
            linear_df.iloc[i,0] = 0
            angular_df.iloc[i,0] = 0
        else:
            linear_df.iloc[i,0] = linear_df.iloc[i-1,0]+1/fps
            angular_df.iloc[i,0] = angular_df.iloc[i-1,0]+1/fps
            
    linear_df = linear_df.rename(columns={0: 'Time (s)'})
    angular_df = angular_df.rename(columns={0: 'Time (s)'})
    linear_df = linear_df.rename(columns={67: 'CoM_X (m)'})
    linear_df = linear_df.rename(columns={68: 'CoM_Y (m)'})


    linear_df.iloc[:,1:] = linear_df.iloc[:,1:]*-1

    ref_X = linear_df.iloc[0,61]
    ref_Y = linear_df.iloc[0,62]

    for col in linear_df.columns:
        if '_X' in col:
            linear_df[col] = linear_df[col]-ref_X
            
        elif '_Y' in col:
            linear_df[col] = linear_df[col]-ref_Y

    linear_df = linear_df.astype(int)
    angular_df = angular_df.astype(int)

    com_wb_x = linear_df['CoM_X (m)']
    com_wb_y = linear_df['CoM_Y (m)']

    com_wb_x_v = com_wb_x.copy()
    com_wb_y_v = com_wb_y.copy()
    com_wb_x_a = com_wb_x.copy()
    com_wb_y_a = com_wb_y.copy()
    
    for i in range(len(com_wb_x)-2):
        
        com_wb_x_v.iloc[i+1] = (com_wb_x.iloc[i+2]-com_wb_x.iloc[i])/(linear_df['Time'].iloc[i+2]-linear_df['Time'].iloc[i])
        com_wb_y_v.iloc[i+1] = (com_wb_y.iloc[i+2]-com_wb_y.iloc[i])/(linear_df['Time'].iloc[i+2]-linear_df['Time'].iloc[i])

    com_wb_x_v.iloc[0] = (com_wb_x_v.iloc[1] + com_wb_x_v.iloc[2])-com_wb_x_v.iloc[3]
    com_wb_x_v.iloc[-1] = (com_wb_x_v.iloc[-2] + com_wb_x_v.iloc[-3])-com_wb_x_v.iloc[-4]

    com_wb_y_v.iloc[0] = (com_wb_y_v.iloc[1] + com_wb_y_v.iloc[2])-com_wb_y_v.iloc[3]
    com_wb_y_v.iloc[-1] = (com_wb_y_v.iloc[-2] + com_wb_y_v.iloc[-3])-com_wb_y_v.iloc[-4]


    for i in range(len(com_wb_x)-2):
        
        com_wb_x_a.iloc[i+1] = (com_wb_x_v.iloc[i+2]-com_wb_x_v.iloc[i])/(linear_df['Time'].iloc[i+2]-linear_df['Time'].iloc[i])
        com_wb_y_a.iloc[i+1] = (com_wb_y_v.iloc[i+2]-com_wb_y_v.iloc[i])/(linear_df['Time'].iloc[i+2]-linear_df['Time'].iloc[i])

    com_wb_x_a.iloc[0] = (com_wb_x_a.iloc[1] + com_wb_x_a.iloc[2])-com_wb_x_a.iloc[3]
    com_wb_x_a.iloc[-1] = (com_wb_x_a.iloc[-2] + com_wb_x_a.iloc[-3])-com_wb_x_a.iloc[-4]

    com_wb_y_a.iloc[0] = (com_wb_y_a.iloc[1] + com_wb_y_a.iloc[2])-com_wb_y_a.iloc[3]
    com_wb_y_a.iloc[-1] = (com_wb_y_a.iloc[-2] + com_wb_y_a.iloc[-3])-com_wb_y_a.iloc[-4]

    net_force_x = com_wb_x_a*entered_mass
    net_force_y = com_wb_y_a*entered_mass

    linear_df[' '] = pd.Series([None] * len(net_force_x ))
    linear_df['Net_Force_X'] = net_force_x
    linear_df['Net_Force_Y'] = net_force_y
            
    progress_placeholder.write("Processing data...")
    # Your existing code for data processing and calculations
    
    df_len = len(linear_df)

    df_int = math.floor(df_len/9)

    fig2, axs = plt.subplots(1, 10, figsize=(15, 3))

    X = linear_df[['WristX','ElbowX','ShoulderX','HipX','KneeX','AnkleX']]
    Y = linear_df[['WristY','ElbowY','ShoulderY','HipY','KneeY','AnkleY']]


    xmin = (X.min()-0.5).min()
    xmax = (X.max()+0.5).max()

    ymin = (Y.min()-0.05).min()
    ymax = (Y.max()+0.1).max()

    for num, i in enumerate(range(0, df_len,df_int)):
        
        axs[num].plot(X.iloc[i],Y.iloc[i])
        axs[num].set_xlim(left=xmin, right=xmax)
        axs[num].set_ylim(bottom=ymin, top=ymax)
        axs[num].set_title(f'Frame {i}')
        axs[num].set_xticks([])
        
        CM_x = linear_df['CoM_X'].iloc[i] 
        CM_y = linear_df['CoM_Y'].iloc[i] 
        
        axs[num].scatter(CM_x, CM_y, color='red', marker='o')
        
        if i>0:
            axs[num].set_yticks([])

        if i ==0:
            axs[num].set_ylabel('Y Displacement (m)')
        
    st.pyplot(fig2)
    st.text('Stick figures of the processed jump movement (the red dots show the location of CoM)')

    linear_df.to_excel(buf := BytesIO(), index=False)
    
    # Display the download button after processing is complete
    st.success("Processing complete!")
    st.text('Does the series of stick figures look like a jump movement? If yes:')
    st.download_button(
        "Download your output file",
        buf.getvalue(),
        "outdat_with_com_nf.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.text('If not, check and fix your Excel file, reload this page, and run the analysis again')
    
    progress_placeholder.empty()
        
            
        
        
