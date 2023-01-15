import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import UploadFile

from predict_like import predict_like

router = APIRouter(prefix="/effects")


@router.post("/like")
async def like(file: UploadFile = File(...), type_predict: str = Form("all")):
    if type_predict not in ['all', 'clear']:
        raise ValueError(f'type_predict must be all or clear. Get {type_predict}')
    img = Image.open(io.BytesIO(file.file.read()))
    img = np.array(img)
    return predict_like(img, type_predict)


@router.post("/like_directory")
async def like_on_directory(dir_path: str = Form(...), type_predict: str = Form("all")):
    if type_predict not in ['all', 'clear']:
        raise ValueError(f'type_predict must be all or clear. Get {type_predict}')
    results = []
    dir_path = Path(dir_path)
    for img_name in os.listdir(dir_path):
        img = Image.open(dir_path / img_name)
        img = np.array(img)
        results.append((img_name, predict_like(img, type_predict)))
    return results
