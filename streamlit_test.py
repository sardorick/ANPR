from hashlib import new
from cv2 import bitwise_and
from imageio import save
import streamlit as st
import numpy as np
import cv2
import easyocr
import imutils
from PIL import Image
from streamlit_option_menu import option_menu
import csv
import uuid
import pandas as pd

# helper function to save files to a csv file
def save_results(text, csv_filename, path):
    img_name = f'{uuid.uuid1()}.jpg'
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])

with st.sidebar:
    menu = option_menu(None, ['Home'], icons=['house'])

if menu == 'Home':
    st.title('Automatic License Plate Recognition app')
    st.markdown('_Created by: Sardorbek Zokirov_')
    image_1 =  Image.open("tested.jpg")
    st.image(image_1)
    st.markdown('This is a basic example of how the ALPR system can work in recognizing images.')
    st.markdown('The application is still in early stages of the development. Going forward, the system will be able to recognize car number plates in real-time and track them.')
    st.markdown('In the application below, we can upload an image and get the label of the license plate.')
    st.markdown('Please upload an image with car license plate:')
    with st.form(key='car'):
        uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'png', 'jpeg'])
        submit_button = st.form_submit_button(label='Get the license plate')
    if uploaded_image != None:

        if submit_button:
            img = Image.open(uploaded_image)
            img = img.save('img.jpg')
            # converted_image = Image.fromarray(uploaded_image)
            img = cv2.imread('img.jpg')        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(bfilter, 30, 90)
            keycontours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # st.write(keycontours)
            contours = imutils.grab_contours(keycontours)
            # st.write(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            # st.write(contours)
            location = None
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break
            # create a mask
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)
            new_image = bitwise_and(img, img, mask=mask)
            (x,y) = np.where(mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2+1, y1:y2+1]
            # easyOCR reader
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image)
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=result[0][-2], org=(approx[0][0][0]+50, approx[1][0][1]+50), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
            res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            st.image(res)
            # save results to a csv file
            save_results(result[0][-2], 'results.csv', 'ANPR')
            st.dataframe(pd.read_csv('results.csv', names=['Image name', 'License plate'], header=None))
    else:
        st.markdown('Please upload an image!')
