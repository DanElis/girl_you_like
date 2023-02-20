import io
import os
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import UploadFile

from parse_path import convert_u
from predict_like import predict_like, face_rating, is_it_like

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
    if is_win_path(dir_path):
        dir_path = convert_u(dir_path)
    dir_path = Path(dir_path)
    for img_name in os.listdir(dir_path):
        print(img_name)
        if img_name.split('.')[-1].lower() not in ['.jpeg', '.png', 'jpg']:
            continue

        img = Image.open(dir_path / img_name)
        img = np.array(img)
        results.append((img_name, predict_like(img, type_predict)))
    return results


@router.post("/like_profile")
async def like_profile(dir_path: str = Form(...), type_predict: str = Form("all")):
    if type_predict not in ['all', 'clear']:
        raise ValueError(f'type_predict must be all or clear. Get {type_predict}')
    ratings = []
    if is_win_path(dir_path):
        dir_path = convert_u(dir_path)
    dir_path = Path(dir_path)
    for img_name in os.listdir(dir_path):
        if img_name.split('.')[-1].lower() not in ['.jpeg', '.png', 'jpg']:
            continue

        img = Image.open(dir_path / img_name)
        img = np.array(img)
        rating = face_rating(img, type_predict)
        if rating == -1:
            continue
        ratings.append(rating)
    return is_it_like(np.mean(ratings))


def is_win_path(path):
    if 'C:\\' in path:
        return True
    return False
