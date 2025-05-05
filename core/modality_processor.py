from typing import List, Dict
import pytesseract
from PIL import Image #opens and manipulates Images
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from config.settings import Config

class ModalityDetector:
    @staticmethod
    def detect_modalities(document: dict) -> List[str]:
        modalities = ["text"]
        
        if "image" in document:
            modalities.append("image")
        if "table" in document:
            modalities.append("table")
        if "math" in document:
            modalities.append("math")
            
        return modalities