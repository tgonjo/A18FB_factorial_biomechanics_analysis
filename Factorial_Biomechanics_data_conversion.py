import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import math
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

deleva_url = "https://www.sciencedirect.com/science/article/pii/0021929095001786?via%3Dihub"

st.subheader("1. Input participant's information and upload your json file")

with st.form(key='mass_form'):
    
    entered_fps = st.number_input("Frame rate of your video data (frames per second).")
    entered_mass = st.number_input("Participant's body mass (kg).")
    selected_sex = st.radio("Participant's sex", ("Male", "Female", "Prefer not to say"))
    st.write("The sex information is necessary to select the CoM calculation method proposed by De Leva (1996) [link to the article](%s). If you select 'Prefer not to say', the average of the male and female models will be used for CoM calculation." % deleva_url)
           
    submit1 = st.form_submit_button("Confirm the frame rate and participant's mass and sex")
    
    if submit1:
        st.text(f'Frame Rate: {entered_fps} frames per second')
        st.text(f'Body mass: {entered_mass} kg')


st.markdown('<p class="big-font">2. Upload your JSON file generated by Factorial Biomechanics</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type='json', accept_multiple_files=False, key='json')
    
if entered_mass == 0:
    st.error("Input participant's mass (don't forget to click the confirm button)")         
    st.stop()
    
elif uploaded_file is None:
    st.error('Upload your json file')         
    st.stop()


progress_placeholder = st.empty()
        
if uploaded_file:
    
    raw_data = uploaded_file.getvalue()
        
    data = json.loads(raw_data.decode("utf-8"))

 
    frames = len(data)

    linear_df_raw = pd.DataFrame(index=range(frames), columns=range(69))
    angular_df_raw = pd.DataFrame(index=range(frames), columns=range(11))

 
    for i in range(frames):
        
        for j in range(len(data[i]['keypoints2D'])):
            segment = data[i]['keypoints2D'][j]['name']
            
            if i == 0:
                linear_df_raw = linear_df_raw.rename(columns={j*2+1: segment+'_X (m)'})
                linear_df_raw = linear_df_raw.rename(columns={j*2+2: segment+'_Y (m)'})
                
            linear_df_raw.iloc[i,j*2+1] = data[i]['keypoints2D'][j]['realX']
            linear_df_raw.iloc[i,j*2+2] = data[i]['keypoints2D'][j]['realY']
            
        linear_df_raw.iloc[i,67] = data[i]['com2D']['realX']
        linear_df_raw.iloc[i,68] = data[i]['com2D']['realY']
        
        for k,joint in enumerate(data[i]['angles2D']):
            if i == 0:
                angular_df_raw = angular_df_raw.rename(columns={k+1: joint})
        
        angular_df_raw.loc[i,'rightElbowAngle'] = dot_product_angle(linear_df_raw,i,'right_elbow_X (m)','right_elbow_Y (m)',
                                            'right_wrist_X (m)','right_wrist_Y (m)','right_shoulder_X (m)','right_shoulder_Y (m)')
        
        angular_df_raw.loc[i,'leftElbowAngle'] = dot_product_angle(linear_df_raw,i,'left_elbow_X (m)','left_elbow_Y (m)',
                                            'left_wrist_X (m)','left_wrist_Y (m)','left_shoulder_X (m)','left_shoulder_Y (m)')
            
        angular_df_raw.loc[i,'rightKneeAngle'] = dot_product_angle(linear_df_raw,i,'right_knee_X (m)','right_knee_Y (m)',
                                            'right_ankle_X (m)','right_ankle_Y (m)','right_hip_X (m)','right_hip_Y (m)')
        
        angular_df_raw.loc[i,'leftKneeAngle'] = dot_product_angle(linear_df_raw,i,'left_knee_X (m)','left_knee_Y (m)',
                                            'left_ankle_X (m)','left_ankle_Y (m)','left_hip_X (m)','left_hip_Y (m)')
        
        angular_df_raw.loc[i,'rightShoulderAngle'] = dot_product_angle(linear_df_raw,i,'right_shoulder_X (m)','right_shoulder_Y (m)',
                                            'right_hip_X (m)','right_hip_Y (m)','right_elbow_X (m)','right_elbow_Y (m)')
        
        angular_df_raw.loc[i,'leftShoulderAngle'] = dot_product_angle(linear_df_raw,i,'left_shoulder_X (m)','left_shoulder_Y (m)',
                                            'left_hip_X (m)','left_hip_Y (m)','left_elbow_X (m)','left_elbow_Y (m)')
        
        angular_df_raw.loc[i,'rightHipAngle'] = dot_product_angle(linear_df_raw,i,'right_hip_X (m)','right_hip_Y (m)',
                                            'right_shoulder_X (m)','right_shoulder_Y (m)','right_knee_X (m)','right_knee_Y (m)')
        
        angular_df_raw.loc[i,'leftHipAngle'] = dot_product_angle(linear_df_raw,i,'left_hip_X (m)','left_hip_Y (m)',
                                            'left_shoulder_X (m)','left_shoulder_Y (m)','left_knee_X (m)','left_knee_Y (m)')
        
        angular_df_raw.loc[i,'rightAnkleAngle'] = dot_product_angle(linear_df_raw,i,'right_ankle_X (m)','right_ankle_Y (m)',
                                            'right_knee_X (m)','right_knee_Y (m)','right_foot_index_X (m)','right_foot_index_Y (m)')
        
        angular_df_raw.loc[i,'leftAnkleAngle'] = dot_product_angle(linear_df_raw,i,'left_ankle_X (m)','left_ankle_Y (m)',
                                            'left_knee_X (m)','left_knee_Y (m)','left_foot_index_X (m)','left_foot_index_Y (m)')
        
        
        linear_df_raw.iloc[i,0] = data[i]['timestamp']
        angular_df_raw.iloc[i,0] = data[i]['timestamp']
            
    linear_df_raw = linear_df_raw.rename(columns={0: 'Time (s)'})
    angular_df_raw = angular_df_raw.rename(columns={0: 'Time (s)'})
    linear_df_raw = linear_df_raw.rename(columns={67: 'CoM_X (m)'})
    linear_df_raw = linear_df_raw.rename(columns={68: 'CoM_Y (m)'})

    for col in linear_df_raw.columns:
        linear_df_raw[col] = pd.to_numeric(linear_df_raw[col], errors='coerce')
    
    for col in angular_df_raw.columns:
        angular_df_raw[col] = pd.to_numeric(angular_df_raw[col], errors='coerce')

    beg_time = linear_df_raw.iloc[0,0]
    end_time = linear_df_raw.iloc[-1,0]

    duration = end_time-beg_time

    time_calibration = entered_fps/50
    
    fps = frames/(duration/time_calibration)

    linear_df_raw['Time (s)'] = linear_df_raw['Time (s)']/time_calibration
    angular_df_raw['Time (s)'] = angular_df_raw['Time (s)']/time_calibration

    time_new = np.arange(linear_df_raw['Time (s)'].min(), linear_df_raw['Time (s)'].max(), round(1 / fps,4))

    linear_df_raw_rs = pd.DataFrame({'Time (s)': time_new})
    angular_df_raw_rs = pd.DataFrame({'Time (s)': time_new})

    for col in linear_df_raw.columns[1:]:
        linear_df_raw_rs[col] = np.interp(time_new, linear_df_raw['Time (s)'], linear_df_raw[col])

    for col in angular_df_raw.columns[1:]:
        angular_df_raw_rs[col] = np.interp(time_new, angular_df_raw['Time (s)'], angular_df_raw[col])


    if fps < entered_fps/3:
        st.error(f'Your raw data file contains {round(entered_fps/fps)} times the smaller number of data compared with your original video. It is likely that the raw data file exported from Factorial Biomechanics is missing too much information. Please try the tracking process on Factorial Biomechanics again. If the issue persists, please try another computer to obtain the tracking data on Factorial Biomechanics.') 
        st.stop()

    st.markdown(
    f"""
    <div style="background-color: #e6ffe6; padding: 10px; border-radius: 5px;">
        <p style="font-size:20px; color: green; font-weight: bold; margin: 0;">
            ✅ Sampling frequency of the output data: {round(fps,2)} Hz ({round(fps,2)} fps)
        </p>
        <p style="font-size:20px; color: red; font-style: italic; margin: 0;">
                  (Note down this information and report it in your method section)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

 
    linear_df_raw_rs.iloc[:,1:] = linear_df_raw_rs.iloc[:,1:]*-1

    ref_X = linear_df_raw_rs.iloc[0,61]
    ref_Y = linear_df_raw_rs.iloc[0,62]

    for col in linear_df_raw_rs.columns:
        if '_X' in col:
            linear_df_raw_rs[col] = linear_df_raw_rs[col]-ref_X
            
        elif '_Y' in col:
            linear_df_raw_rs[col] = linear_df_raw_rs[col]-ref_Y

    # linear_df = linear_df.astype(int)
    # angular_df = angular_df.astype(int)

    linear_df = linear_df_raw_rs.copy()
    angular_df = angular_df_raw_rs.copy()

    linear_cols = list(linear_df.columns[1:])
    angular_cols = list(angular_df.columns[1:])

    for linear_col in linear_cols:
        linear_df[linear_col] = apply_low_pass_filter(linear_df_raw_rs[linear_col], 4,fps)

    for angular_col in angular_cols:
        angular_df[angular_col] = apply_low_pass_filter(angular_df_raw_rs[angular_col], 4,fps)

    df_columns = ['Time', 'HeadX','HeadY','FingerX','FingerY','WristX', 'WristY', 'ElbowX', 'ElbowY', 'ShoulderX',
       'ShoulderY', 'HipX', 'HipY', 'KneeX', 'KneeY', 'AnkleX', 'AnkleY','ToeX','ToeY']

    df = pd.DataFrame(index=range(len(linear_df_raw_rs)), columns=df_columns)

    
    df.iloc[:,1] = (linear_df.iloc[:,15] + linear_df.iloc[:,17])/2
    df.iloc[:,2] = (linear_df.iloc[:,16] + linear_df.iloc[:,18])/2
    df.iloc[:,3] = (linear_df.iloc[:,39] + linear_df.iloc[:,41])/2
    df.iloc[:,4] = (linear_df.iloc[:,40] + linear_df.iloc[:,42])/2
    df.iloc[:,5] = (linear_df.iloc[:,31] + linear_df.iloc[:,33])/2
    df.iloc[:,6] = (linear_df.iloc[:,32] + linear_df.iloc[:,34])/2
    df.iloc[:,7] = (linear_df.iloc[:,27] + linear_df.iloc[:,29])/2
    df.iloc[:,8] = (linear_df.iloc[:,28] + linear_df.iloc[:,30])/2
    df.iloc[:,9] = (linear_df.iloc[:,23] + linear_df.iloc[:,25])/2
    df.iloc[:,10] = (linear_df.iloc[:,24] + linear_df.iloc[:,26])/2
    df.iloc[:,11] = (linear_df.iloc[:,47] + linear_df.iloc[:,49])/2
    df.iloc[:,12] = (linear_df.iloc[:,48] + linear_df.iloc[:,50])/2
    df.iloc[:,13] = (linear_df.iloc[:,51] + linear_df.iloc[:,53])/2
    df.iloc[:,14] = (linear_df.iloc[:,52] + linear_df.iloc[:,54])/2
    df.iloc[:,15] = (linear_df.iloc[:,55] + linear_df.iloc[:,57])/2
    df.iloc[:,16] = (linear_df.iloc[:,56] + linear_df.iloc[:,58])/2
    df.iloc[:,17] = (linear_df.iloc[:,63] + linear_df.iloc[:,65])/2
    df.iloc[:,18] = (linear_df.iloc[:,64] + linear_df.iloc[:,66])/2


    df['Time'] = linear_df['Time (s)']

    BSP_M = [['hand',0.61, 79.0],['lower_arm',1.62, 45.74],['upper_arm',2.71,57.72],['trunk',43.46,43.10],['thigh',14.16,40.95],['shank',4.33,43.95],['foot',1.37,44.15]]

    BSP_F = [['hand',0.56, 74.74],['lower_arm',1.38, 45.59],['upper_arm',2.55,57.54],['trunk',42.57,37.82],['thigh',14.78,36.12],['shank',4.81,43.52],['foot',1.29,40.14]]

    df_bsp_m = pd.DataFrame(BSP_M, columns=['Segment', 'Percent_mass','CoM_location'])
    df_bsp_f = pd.DataFrame(BSP_F, columns=['Segment', 'Percent_mass','CoM_location'])
    df_bsp_m.set_index('Segment', inplace=True)
    df_bsp_f.set_index('Segment', inplace=True)

    df_bsp_m = df_bsp_m/100
    df_bsp_f = df_bsp_f/100

    df_bsp_pns = (df_bsp_m + df_bsp_f)/2

    if selected_sex == 'Male':
        com_hand_x = df['WristX']+(df['FingerX']-df['WristX'])*df_bsp_m.loc['hand']['CoM_location']
        com_la_x = df['ElbowX']+(df['WristX']-df['ElbowX'])*df_bsp_m.loc['lower_arm']['CoM_location']
        com_ua_x = df['ShoulderX']+(df['ElbowX']-df['ShoulderX'])*df_bsp_m.loc['upper_arm']['CoM_location']
        com_trunk_x = df['ShoulderX']+(df['HipX']-df['ShoulderX'])*df_bsp_m.loc['trunk']['CoM_location']
        com_thigh_x = df['HipX']+(df['KneeX']-df['HipX'])*df_bsp_m.loc['thigh']['CoM_location']
        com_shank_x = df['KneeX']+(df['AnkleX']-df['KneeX'])*df_bsp_m.loc['shank']['CoM_location']
        com_foot_x = df['AnkleX']+(df['ToeX']-df['AnkleX'])*df_bsp_m.loc['foot']['CoM_location']
        
        com_hand_y = df['WristY']+(df['FingerY']-df['WristY'])*df_bsp_m.loc['hand']['CoM_location']
        com_la_y = df['ElbowY']+(df['WristY']-df['ElbowY'])*df_bsp_m.loc['lower_arm']['CoM_location']
        com_ua_y = df['ShoulderY']+(df['ElbowY']-df['ShoulderY'])*df_bsp_m.loc['upper_arm']['CoM_location']
        com_trunk_y = df['ShoulderY']+(df['HipY']-df['ShoulderY'])*df_bsp_m.loc['trunk']['CoM_location']
        com_thigh_y = df['HipY']+(df['KneeY']-df['HipY'])*df_bsp_m.loc['thigh']['CoM_location']
        com_shank_y = df['KneeY']+(df['AnkleY']-df['KneeY'])*df_bsp_m.loc['shank']['CoM_location']
        com_foot_y = df['AnkleY']+(df['ToeY']-df['AnkleY'])*df_bsp_m.loc['foot']['CoM_location']
        
        com_wb_x = (df['HeadX']*6.94/100 + com_hand_x*df_bsp_m['Percent_mass']['hand']+ com_la_x*df_bsp_m['Percent_mass']['lower_arm']+com_ua_x*df_bsp_m['Percent_mass']['upper_arm'] \
        + com_trunk_x*df_bsp_m['Percent_mass']['trunk']+com_thigh_x*df_bsp_m['Percent_mass']['thigh']+com_shank_x*df_bsp_m['Percent_mass']['shank']+com_foot_x*df_bsp_m['Percent_mass']['foot'])\
        / df_bsp_m['Percent_mass'].sum()
        
        com_wb_y = (df['HeadY']*6.94/100 + com_hand_y*df_bsp_m['Percent_mass']['hand']+com_la_y*df_bsp_m['Percent_mass']['lower_arm']+com_ua_y*df_bsp_m['Percent_mass']['upper_arm'] \
        + com_trunk_y*df_bsp_m['Percent_mass']['trunk']+com_thigh_y*df_bsp_m['Percent_mass']['thigh']+com_shank_y*df_bsp_m['Percent_mass']['shank']+com_foot_y*df_bsp_m['Percent_mass']['foot'])\
        / df_bsp_m['Percent_mass'].sum()
        
    elif selected_sex == 'Female':
        com_hand_x = df['WristX']+(df['FingerX']-df['WristX'])*df_bsp_f.loc['hand']['CoM_location']
        com_la_x = df['ElbowX']+(df['WristX']-df['ElbowX'])*df_bsp_f.loc['lower_arm']['CoM_location']
        com_ua_x = df['ShoulderX']+(df['ElbowX']-df['ShoulderX'])*df_bsp_f.loc['upper_arm']['CoM_location']
        com_trunk_x = df['ShoulderX']+(df['HipX']-df['ShoulderX'])*df_bsp_f.loc['trunk']['CoM_location']
        com_thigh_x = df['HipX']+(df['KneeX']-df['HipX'])*df_bsp_f.loc['thigh']['CoM_location']
        com_shank_x = df['KneeX']+(df['AnkleX']-df['KneeX'])*df_bsp_f.loc['shank']['CoM_location']
        com_foot_x = df['AnkleX']+(df['ToeX']-df['AnkleX'])*df_bsp_f.loc['foot']['CoM_location']
        
        com_hand_y = df['WristY']+(df['FingerY']-df['WristY'])*df_bsp_f.loc['hand']['CoM_location']
        com_la_y = df['ElbowY']+(df['WristY']-df['ElbowY'])*df_bsp_f.loc['lower_arm']['CoM_location']
        com_ua_y = df['ShoulderY']+(df['ElbowY']-df['ShoulderY'])*df_bsp_f.loc['upper_arm']['CoM_location']
        com_trunk_y = df['ShoulderY']+(df['HipY']-df['ShoulderY'])*df_bsp_f.loc['trunk']['CoM_location']
        com_thigh_y = df['HipY']+(df['KneeY']-df['HipY'])*df_bsp_f.loc['thigh']['CoM_location']
        com_shank_y = df['KneeY']+(df['AnkleY']-df['KneeY'])*df_bsp_f.loc['shank']['CoM_location']
        com_foot_y = df['AnkleY']+(df['ToeY']-df['AnkleY'])*df_bsp_f.loc['foot']['CoM_location']
        
        com_wb_x = (df['HeadX']*6.68/100 + com_hand_x*df_bsp_f['Percent_mass']['hand']+com_la_x*df_bsp_f['Percent_mass']['lower_arm']+com_ua_x*df_bsp_f['Percent_mass']['upper_arm'] \
        + com_trunk_x*df_bsp_f['Percent_mass']['trunk']+com_thigh_x*df_bsp_f['Percent_mass']['thigh']+com_shank_x*df_bsp_f['Percent_mass']['shank']+com_foot_x*df_bsp_f['Percent_mass']['foot'])\
        / df_bsp_f['Percent_mass'].sum()
        
        com_wb_y = (df['HeadY']*6.68/100 + com_hand_y*df_bsp_f['Percent_mass']['hand']+com_la_y*df_bsp_f['Percent_mass']['lower_arm']+com_ua_y*df_bsp_f['Percent_mass']['upper_arm'] \
        + com_trunk_y*df_bsp_f['Percent_mass']['trunk']+com_thigh_y*df_bsp_f['Percent_mass']['thigh']+com_shank_y*df_bsp_f['Percent_mass']['shank']+com_foot_y*df_bsp_f['Percent_mass']['foot'])\
        / df_bsp_f['Percent_mass'].sum()
        
    else:
        com_hand_x = df['WristX']+(df['FingerX']-df['WristX'])*df_bsp_pns.loc['hand']['CoM_location']
        com_la_x = df['ElbowX']+(df['WristX']-df['ElbowX'])*df_bsp_pns.loc['lower_arm']['CoM_location']
        com_ua_x = df['ShoulderX']+(df['ElbowX']-df['ShoulderX'])*df_bsp_pns.loc['upper_arm']['CoM_location']
        com_trunk_x = df['ShoulderX']+(df['HipX']-df['ShoulderX'])*df_bsp_pns.loc['trunk']['CoM_location']
        com_thigh_x = df['HipX']+(df['KneeX']-df['HipX'])*df_bsp_pns.loc['thigh']['CoM_location']
        com_shank_x = df['KneeX']+(df['AnkleX']-df['KneeX'])*df_bsp_pns.loc['shank']['CoM_location']
        com_foot_x = df['AnkleX']+(df['ToeX']-df['AnkleX'])*df_bsp_pns.loc['foot']['CoM_location']
        
        com_hand_y = df['WristY']+(df['FingerY']-df['WristY'])*df_bsp_pns.loc['hand']['CoM_location']
        com_la_y = df['ElbowY']+(df['WristY']-df['ElbowY'])*df_bsp_pns.loc['lower_arm']['CoM_location']
        com_ua_y = df['ShoulderY']+(df['ElbowY']-df['ShoulderY'])*df_bsp_pns.loc['upper_arm']['CoM_location']
        com_trunk_y = df['ShoulderY']+(df['HipY']-df['ShoulderY'])*df_bsp_pns.loc['trunk']['CoM_location']
        com_thigh_y = df['HipY']+(df['KneeY']-df['HipY'])*df_bsp_pns.loc['thigh']['CoM_location']
        com_shank_y = df['KneeY']+(df['AnkleY']-df['KneeY'])*df_bsp_pns.loc['shank']['CoM_location']
        com_foot_y = df['AnkleY']+(df['ToeY']-df['AnkleY'])*df_bsp_pns.loc['foot']['CoM_location']
        
        com_wb_x = (df['HeadX']*6.81/100 + com_hand_x*df_bsp_pns['Percent_mass']['hand']+com_la_x*df_bsp_pns['Percent_mass']['lower_arm']+com_ua_x*df_bsp_pns['Percent_mass']['upper_arm'] \
        + com_trunk_x*df_bsp_pns['Percent_mass']['trunk']+com_thigh_x*df_bsp_pns['Percent_mass']['thigh']+com_shank_x*df_bsp_pns['Percent_mass']['shank']+com_foot_x*df_bsp_pns['Percent_mass']['foot'])\
        / df_bsp_pns['Percent_mass'].sum()
        
        com_wb_y = (df['HeadY']*6.81/100 + com_hand_y*df_bsp_pns['Percent_mass']['hand']+com_la_y*df_bsp_pns['Percent_mass']['lower_arm']+com_ua_y*df_bsp_pns['Percent_mass']['upper_arm'] \
        + com_trunk_y*df_bsp_pns['Percent_mass']['trunk']+com_thigh_y*df_bsp_pns['Percent_mass']['thigh']+com_shank_y*df_bsp_pns['Percent_mass']['shank']+com_foot_y*df_bsp_pns['Percent_mass']['foot'])\
        / df_bsp_pns['Percent_mass'].sum()

    linear_df['CoM_X (m)'] = com_wb_x
    linear_df['CoM_Y (m)'] = com_wb_y

    com_wb_x_v = com_wb_x.copy()
    com_wb_y_v = com_wb_y.copy()
    com_wb_x_a = com_wb_x.copy()
    com_wb_y_a = com_wb_y.copy()
    
    for i in range(len(com_wb_x)-2):
        
        com_wb_x_v.iloc[i+1] = (com_wb_x.iloc[i+2]-com_wb_x.iloc[i])/(linear_df['Time (s)'].iloc[i+2]-linear_df['Time (s)'].iloc[i])
        com_wb_y_v.iloc[i+1] = (com_wb_y.iloc[i+2]-com_wb_y.iloc[i])/(linear_df['Time (s)'].iloc[i+2]-linear_df['Time (s)'].iloc[i])

    com_wb_x_v.iloc[0] = (com_wb_x_v.iloc[1] + com_wb_x_v.iloc[2])-com_wb_x_v.iloc[3]
    com_wb_x_v.iloc[-1] = (com_wb_x_v.iloc[-2] + com_wb_x_v.iloc[-3])-com_wb_x_v.iloc[-4]

    com_wb_y_v.iloc[0] = (com_wb_y_v.iloc[1] + com_wb_y_v.iloc[2])-com_wb_y_v.iloc[3]
    com_wb_y_v.iloc[-1] = (com_wb_y_v.iloc[-2] + com_wb_y_v.iloc[-3])-com_wb_y_v.iloc[-4]


    for i in range(len(com_wb_x)-2):
        
        com_wb_x_a.iloc[i+1] = (com_wb_x_v.iloc[i+2]-com_wb_x_v.iloc[i])/(linear_df['Time (s)'].iloc[i+2]-linear_df['Time (s)'].iloc[i])
        com_wb_y_a.iloc[i+1] = (com_wb_y_v.iloc[i+2]-com_wb_y_v.iloc[i])/(linear_df['Time (s)'].iloc[i+2]-linear_df['Time (s)'].iloc[i])

    com_wb_x_a.iloc[0] = (com_wb_x_a.iloc[1] + com_wb_x_a.iloc[2])-com_wb_x_a.iloc[3]
    com_wb_x_a.iloc[-1] = (com_wb_x_a.iloc[-2] + com_wb_x_a.iloc[-3])-com_wb_x_a.iloc[-4]

    com_wb_y_a.iloc[0] = (com_wb_y_a.iloc[1] + com_wb_y_a.iloc[2])-com_wb_y_a.iloc[3]
    com_wb_y_a.iloc[-1] = (com_wb_y_a.iloc[-2] + com_wb_y_a.iloc[-3])-com_wb_y_a.iloc[-4]

    net_force_x = com_wb_x_a*entered_mass
    net_force_y = com_wb_y_a*entered_mass

    linear_df['Net_Force_X (N)'] = net_force_x
    linear_df['Net_Force_Y (N)'] = net_force_y

            
    progress_placeholder.write("Processing data...")
    # Your existing code for data processing and calculations
    
    df_len = len(linear_df)

    df_int = math.floor(df_len/11)

    fig2, axs = plt.subplots(3, 4, figsize=(15, 10))

    LX = linear_df[['left_index_X (m)','left_wrist_X (m)','left_elbow_X (m)','left_shoulder_X (m)','left_hip_X (m)','left_knee_X (m)','left_ankle_X (m)','left_heel_X (m)','left_foot_index_X (m)']]
    LY = linear_df[['left_index_Y (m)','left_wrist_Y (m)','left_elbow_Y (m)','left_shoulder_Y (m)','left_hip_Y (m)','left_knee_Y (m)','left_ankle_Y (m)','left_heel_Y (m)','left_foot_index_Y (m)']]

    RX = linear_df[['right_index_X (m)','right_wrist_X (m)','right_elbow_X (m)','right_shoulder_X (m)','right_hip_X (m)','right_knee_X (m)','right_ankle_X (m)','right_heel_X (m)','right_foot_index_X (m)']]
    RY = linear_df[['right_index_Y (m)','right_wrist_Y (m)','right_elbow_Y (m)','right_shoulder_Y (m)','right_hip_Y (m)','right_knee_Y (m)','right_ankle_Y (m)','right_heel_Y (m)','right_foot_index_Y (m)']]

    ShoulderX = linear_df[['right_shoulder_X (m)','left_shoulder_X (m)']]
    ShoulderY = linear_df[['right_shoulder_Y (m)','left_shoulder_Y (m)']]

    HipX = linear_df[['right_hip_X (m)','left_hip_X (m)']]
    HipY = linear_df[['right_hip_Y (m)','left_hip_Y (m)']]

    xmin = (LX.min()-0.5).min()
    xmax = (LX.max()+0.5).max()

    ymin = (LY.min()-0.05).min()
    ymax = (LY.max()+0.6).max()

    for num, i in enumerate(range(0, df_len,df_int)):
        if 0 <= num <= 3:
            fig_row = 0
            fig_col = num

        elif 4 <= num <= 7:
            fig_row = 1
            fig_col = num-4

        elif 8 <= num <= 11:
            fig_row = 2
            fig_col = num-8


        fig_time = linear_df.iloc[i,0]
        fig_time = round(fig_time,2)

        axs[fig_row, fig_col].plot(LX.iloc[i],LY.iloc[i])
        axs[fig_row, fig_col].plot(RX.iloc[i],RY.iloc[i])
        axs[fig_row, fig_col].plot(ShoulderX.iloc[i],ShoulderY.iloc[i])
        axs[fig_row, fig_col].plot(HipX.iloc[i],HipY.iloc[i])

        axs[fig_row, fig_col].set_xlim(left=xmin, right=xmax)
        axs[fig_row, fig_col].set_ylim(bottom=ymin, top=ymax)
        axs[fig_row, fig_col].set_title(f'Time: {fig_time} s')
        axs[fig_row, fig_col].set_xticks([])
        
        CM_x = linear_df['CoM_X (m)'].iloc[i] 
        CM_y = linear_df['CoM_Y (m)'].iloc[i] 

        Head_x = (linear_df.loc[i,'left_ear_X (m)']+linear_df.loc[i,'right_ear_X (m)'])/2
        Head_y = (linear_df.loc[i,'left_ear_Y (m)']+linear_df.loc[i,'right_ear_Y (m)'])/2
        
        axs[fig_row, fig_col].scatter(CM_x, CM_y, color='red', marker='o')
        axs[fig_row, fig_col].scatter(Head_x, Head_y, marker='o', s=100)
        
        if i ==0 | i ==4 | i==8:
            axs[fig_row, fig_col].set_ylabel('Y Displacement (m)')
        else:
            axs[fig_row, fig_col].set_yticks([])
        
    st.pyplot(fig2)
    st.text('Stick figures of the processed jump movement (the red dots show the location of CoM)')

    # Display the download button after processing is complete
    st.success("Processing complete!")
    st.write('Does the series of stick figures look like a jump movement? If yes, select the data you want to export and download the output file')
    st.write('If not, check and fix your tracking file on Factorial Biomechanics, and run the analysis again.')


    st.markdown('<p class="big-font">3. Select the side of the body</p>', unsafe_allow_html=True)
    selected_side = st.radio("For which side of the joints do you want to export the data?", ("left", "right", "both left and right"))

    output_joints = ['ear', 'shoulder','elbow','wrist','finger','hip','knee','ankle','toe','Centre of Mass','Net Force']

   # Set the number of columns per row
    num_cols_per_row = 3  # Adjust this based on your needs
    columns = list(output_joints)

    # Create checkboxes in a grid layout
    selected_joints = []
    st.write("### 4. Select linear displacement and kinetic (force) data you want to export:")

    # Arrange checkboxes in a table-like format
    cols = st.columns(num_cols_per_row)  # Create multiple columns
    for index, col in enumerate(columns):
        with cols[index % num_cols_per_row]:  # Place in respective column
            if st.checkbox(col, key=col):
                selected_joints.append(col)
    
    all_columns = list(linear_df.columns[1:])
    selected_columns = ['Time (s)']

    for joint in selected_joints:
        if selected_side == 'both left and right':
            for column in all_columns:
                if joint in column:
                    selected_columns.append(column)
                elif 'CoM' in column and joint == 'Centre of Mass':
                    selected_columns.append(column)
                elif 'Force' in  column and joint == 'Net Force':
                    selected_columns.append(column)
        
        else:
           for column in all_columns:
               if 'CoM' in column and joint == 'Centre of Mass':
                    selected_columns.append(column)
               elif 'Force' in  column and joint == 'Net Force':
                    selected_columns.append(column)
               elif joint in column and selected_side in column:
                    selected_columns.append(column)


    # Create new DataFrame based on selected columns
    if selected_columns:
        new_df = linear_df[selected_columns]


    output_angles = ['Elbow','Shoulder','Hip','Knee','Ankle']

    ang_columns = list(output_angles)

    # Create checkboxes in a grid layout
    selected_angles = []
    st.write("### 5. Select angular displacement (angle) data you want to export:")

    # Arrange checkboxes in a table-like format
    cols = st.columns(num_cols_per_row)  # Create multiple columns
    for index, col in enumerate(ang_columns):
        with cols[index % num_cols_per_row]:  # Place in respective column
            if st.checkbox(col, key=col):
                selected_angles.append(col)

    all_ang_columns = list(angular_df.columns[1:])
    print(all_ang_columns)
    selected_ang_columns = ['Time (s)']

    for angle in selected_angles:
        if selected_side == 'both left and right':
            for column in all_ang_columns:
                if angle in column:
                    selected_ang_columns.append(column)
        else:
            for column in all_ang_columns:
                if selected_side in column and angle in column:
                    selected_ang_columns.append(column)
                

    
    if selected_ang_columns:
        new_ang_df = angular_df[selected_ang_columns]

    # Create a BytesIO buffer
    buf = BytesIO()

    # Reset buffer position before writing the second DataFrame
    buf.seek(0)

    # Save new_ang_df to Excel in the same buffer
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        new_df.to_excel(writer, sheet_name="Linear data", index=False)
        new_ang_df.to_excel(writer, sheet_name="Angular data", index=False)

    # Reset buffer position before downloading
    buf.seek(0)
    
    st.write("### 6. Choose the file name and download:")
    # User inputs the file name
    file_name = st.text_input("Enter file name (without extension):", "output_data")

    # Ensure the user provides a valid name
    full_file_name = f"{file_name}.xlsx" if file_name.strip() else "output_data.xlsx"

    # Download button
    st.download_button(
        label="Download your output file",
        data=buf.getvalue(),
        file_name=full_file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    
    progress_placeholder.empty()
        
            
        
        
