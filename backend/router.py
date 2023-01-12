import io

from PIL import Image
from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import UploadFile

from predict_like import predict_like

router = APIRouter(prefix="/effects")


@router.post("/like")
async def apply_effect(file: UploadFile = File(...), type_predict: str = Form("all")):
    if type_predict not in ['all', 'clear']:
        raise ValueError(f'type_predict must be all or clear. Get {type_predict}')
    img = Image.open(io.BytesIO(file.file.read()))
    return predict_like(img, type_predict)
