  
import os
import sys
import streamlit as st
from extract import fetch_invoice_details, draw_image 
import os
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
import base64

st.sidebar.title("Invoice Extraction using BERT")

def main():
    st.sidebar.subheader('Please upload an Invoice')
    image_file_uploaded = st.sidebar.file_uploader('Upload an image', type = 'tif')
    if image_file_uploaded:
        image = Image.open(image_file_uploaded)
        imageFilename = os.path.join("/mnt/d/Data_Science_Work/hello-vue3/uploads/", image_file_uploaded.name)
        image.save(imageFilename)
        """"""
        print(f"Uploaded Image is {imageFilename}")
        dest_image_file = imageFilename.replace(".png","_result.png")
        result = None
        destination = Path(imageFilename)
        result = fetch_invoice_details(imageFilename)
        df_result = pd.DataFrame(result)
        image = draw_image(imageFilename, df_result)
        result = [{key:val for key, val in item.items() if key != 'boxes'} for item in result]
        image.save(dest_image_file)      
        with st.beta_expander('Predicted Image', expanded = True):
            st.image(image, use_column_width = True)        
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.subheader('Prediction') 
        st.dataframe(pd.DataFrame(result))

if __name__ == "__main__":
    main()