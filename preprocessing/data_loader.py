import os
import cv2
import numpy as np
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Union

from database.aviation_db import AviationDatabase

logger = logging.getLogger(__name__)

class ChartPreprocessor:
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.target_size = self.config.get('target_size', (1024, 1024))
        self.db = AviationDatabase()
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"找不到图像: {image_path}")
            
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def preprocess_chart(self, image: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        edges = cv2.Canny(gray, 100, 200)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = 100
        max_area = image.shape[0] * image.shape[1] / 10
        filtered_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'edges': edges,
            'contours': filtered_contours,
            'thresholded': thresh
        }
    
    def extract_text(self, image: np.ndarray) -> List[Dict]:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            logger.warning("未安装pytesseract，文本提取功能已禁用")
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        data = pytesseract.image_to_data(binary, output_type=Output.DICT)
        
        text_regions = []
        for i in range(len(data['text'])):
            if data['conf'][i] > 60 and len(data['text'][i].strip()) > 2:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text_regions.append({
                    'text': data['text'][i],
                    'confidence': data['conf'][i],
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                })
        
        return text_regions
    
    def detect_runway_patterns(self, image: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        line_groups = []
        for i, line1 in enumerate(lines):
            x1, y1, x2, y2 = line1[0]
            angle1 = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            found_group = False
            for group in line_groups:
                group_angle = group['angle']
                if abs(angle1 - group_angle) < 10 or abs(abs(angle1 - group_angle) - 180) < 10:
                    group['lines'].append(line1[0])
                    found_group = True
                    break
            
            if not found_group:
                line_groups.append({
                    'angle': angle1,
                    'lines': [line1[0]]
                })
        
        runway_candidates = [g for g in line_groups if len(g['lines']) >= 2]
        
        runways = []
        for i, candidate in enumerate(runway_candidates):
            lines = np.array(candidate['lines'])
            x_min = np.min(lines[:, [0, 2]])
            x_max = np.max(lines[:, [0, 2]])
            y_min = np.min(lines[:, [1, 3]])
            y_max = np.max(lines[:, [1, 3]])
            
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = max(width, height) / max(1, min(width, height))
            
            if aspect_ratio > 3:
                runways.append({
                    'id': i,
                    'x_min': int(x_min),
                    'y_min': int(y_min),
                    'x_max': int(x_max),
                    'y_max': int(y_max),
                    'angle': candidate['angle'],
                    'aspect_ratio': aspect_ratio
                })
        
        return runways
    
    def find_potential_landmarks(self, image: np.ndarray, airport_icao: str = None) -> List[Dict]:
        landmarks = []
        
        features = self.extract_features(image)
        text_regions = self.extract_text(image)
        runways = self.detect_runway_patterns(image)
        
        if features['keypoints'] is not None:
            for i, kp in enumerate(features['keypoints']):
                landmarks.append({
                    'type': 'keypoint',
                    'id': f'kp_{i}',
                    'x': kp.pt[0],
                    'y': kp.pt[1],
                    'size': kp.size,
                    'angle': kp.angle,
                    'response': kp.response
                })
        
        for i, text in enumerate(text_regions):
            landmarks.append({
                'type': 'text',
                'id': f'text_{i}',
                'text': text['text'],
                'x': text['x'] + text['width'] / 2,
                'y': text['y'] + text['height'] / 2,
                'width': text['width'],
                'height': text['height'],
                'confidence': text['confidence']
            })
        
        for i, runway in enumerate(runways):
            center_x = (runway['x_min'] + runway['x_max']) / 2
            center_y = (runway['y_min'] + runway['y_max']) / 2
            landmarks.append({
                'type': 'runway',
                'id': f'runway_{i}',
                'x': center_x,
                'y': center_y, 
                'width': runway['x_max'] - runway['x_min'],
                'height': runway['y_max'] - runway['y_min'],
                'angle': runway['angle'],
                'aspect_ratio': runway['aspect_ratio']
            })
        
        if airport_icao:
            self._match_landmarks_with_database(landmarks, airport_icao)
        
        return landmarks
    
    def _match_landmarks_with_database(self, landmarks: List[Dict], airport_icao: str) -> None:
        airport_df = self.db.get_airport(airport_icao)
        if airport_df.empty:
            logger.warning(f"数据库中找不到机场 {airport_icao}")
            return
        
        airport = airport_df.iloc[0]
        
        runways_df = self.db.get_runways(airport_icao)
        navaids_df = self.db.get_navaids_near_airport(airport_icao, radius_nm=30)
        
        for landmark in landmarks:
            if landmark['type'] == 'text':
                text = landmark['text'].strip()
                
                for _, runway in runways_df.iterrows():
                    designation = runway['designation']
                    designations = designation.split('/')
                    
                    for d in designations:
                        norm_d = d.replace('0', '') if d.startswith('0') else d
                        
                        if text == d or text == norm_d:
                            landmark['matched_feature'] = {
                                'type': 'runway',
                                'designation': d,
                                'latitude_start': runway['latitude_start'],
                                'longitude_start': runway['longitude_start'],
                                'latitude_end': runway['latitude_end'],
                                'longitude_end': runway['longitude_end']
                            }
                            break
                
                if 'matched_feature' not in landmark:
                    for _, navaid in navaids_df.iterrows():
                        if text == navaid['ident']:
                            landmark['matched_feature'] = {
                                'type': 'navaid',
                                'ident': navaid['ident'],
                                'name': navaid['name'],
                                'navaid_type': navaid['type'],
                                'latitude': navaid['latitude'],
                                'longitude': navaid['longitude']
                            }
                            break
                            
        for landmark in landmarks:
            if landmark['type'] == 'runway':
                landmark_angle = landmark['angle'] % 180
                
                best_match = None
                min_angle_diff = float('inf')
                
                for _, runway in runways_df.iterrows():
                    lat1, lon1 = runway['latitude_start'], runway['longitude_start']
                    lat2, lon2 = runway['latitude_end'], runway['longitude_end']
                    
                    y = np.sin(lon2-lon1) * np.cos(lat2)
                    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2-lon1)
                    bearing = (np.arctan2(y, x) * 180 / np.pi) % 360
                    
                    runway_angle = bearing % 180
                    
                    angle_diff = min(abs(landmark_angle - runway_angle), 
                                    abs(landmark_angle - (runway_angle + 180) % 180))
                    
                    if angle_diff < min_angle_diff and angle_diff < 30:
                        min_angle_diff = angle_diff
                        best_match = runway
                
                if best_match is not None:
                    landmark['matched_feature'] = {
                        'type': 'runway',
                        'designation': best_match['designation'],
                        'latitude_start': best_match['latitude_start'],
                        'longitude_start': best_match['longitude_start'],
                        'latitude_end': best_match['latitude_end'],
                        'longitude_end': best_match['longitude_end']
                    }


class ChartDataset(Dataset):
    
    def __init__(self, 
                 db: AviationDatabase,
                 chart_paths: List[str] = None,
                 airport_icao: str = None,
                 chart_type: str = None,
                 transform=None,
                 target_size=(1024, 1024)):
        self.db = db
        self.transform = transform or transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_size = target_size
        self.preprocessor = ChartPreprocessor({'target_size': target_size})
        
        if chart_paths:
            self.charts = [{'file_path': p} for p in chart_paths]
        else:
            charts_df = self.db.get_charts(airport_icao, chart_type)
            self.charts = charts_df.to_dict('records')
        
        logger.info(f"加载了 {len(self.charts)} 张航图")
    
    def __len__(self):
        return len(self.charts)
    
    def __getitem__(self, idx):
        chart = self.charts[idx]
        chart_path = chart['file_path']
        
        try:
            img = Image.open(chart_path).convert('RGB')
            
            tensor_img = self.transform(img)
            
            metadata = {k: v for k, v in chart.items() if k != 'file_path'}
            
            if 'id' in chart and chart.get('calibrated', 0) == 1:
                calibration = self.db.get_calibration(chart['id'])
                if calibration:
                    metadata['calibration'] = calibration
            
            if 'airport_icao' in chart:
                np_img = np.array(img)
                landmarks = self.preprocessor.find_potential_landmarks(np_img, chart['airport_icao'])
                metadata['landmarks'] = landmarks
                
            return {
                'image': tensor_img,
                'chart_path': chart_path,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"加载航图 {chart_path} 时出错: {e}")
            dummy_tensor = torch.zeros((3, *self.target_size))
            return {
                'image': dummy_tensor,
                'chart_path': chart_path,
                'metadata': {'error': str(e)}
            }


def get_chart_dataloader(
    db: AviationDatabase,
    chart_paths: List[str] = None,
    airport_icao: str = None,
    chart_type: str = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    target_size: Tuple[int, int] = (1024, 1024)
) -> DataLoader:
    dataset = ChartDataset(
        db=db,
        chart_paths=chart_paths,
        airport_icao=airport_icao,
        chart_type=chart_type,
        target_size=target_size
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
