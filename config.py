import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

NUM_CLASSES = 8 
NUM_POINTS = 2048
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

CLASSES = ['beam', 'column', 'floor', 'wall', 'roof', 'foundation', 'pipe', 'other']

MODEL_TYPE = 'Combined'

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
NUM_WORKERS = 4
