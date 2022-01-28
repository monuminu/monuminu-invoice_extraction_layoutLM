from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse 

from extract import fetch_invoice_details, draw_image 
import os
import shutil
from pathlib import Path
from PIL import Image
import pandas as pd
app = FastAPI()
import base64


origins = ["*"]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    imageFilename = os.path.join("/mnt/d/Data_Science_Work/hello-vue3/uploads/", file.filename)
    dest_image_file = imageFilename.replace(".tif","_result.png")
    result = None
    destination = Path(imageFilename)
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    result = fetch_invoice_details(imageFilename)
    df_result = pd.DataFrame(result)
    image = draw_image(imageFilename, df_result)
    result = [{key:val for key, val in item.items() if key != 'boxes'} for item in result]
    image.save(dest_image_file)      
    print(f"{file.filename} ran successfully !")
    with open(dest_image_file, "rb") as image_file:
        encoded_image_string = base64.b64encode(image_file.read())
    return {"image" : encoded_image_string, "value": result}