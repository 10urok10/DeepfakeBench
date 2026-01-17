import os
# --- KRİTİK DÜZELTME: BU SATIR EN ÜSTTE OLMALI ---
# OpenMP kütüphane çakışmasını (libiomp5md.dll hatasını) engeller.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# -------------------------------------------------

import sys
import torch
import yaml
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gradio as gr  # Arayüz kütüphanesi

# 1. YOL VE IMPORT AYARLARI
current_dir = os.getcwd()
training_path = os.path.join(current_dir, 'training')
sys.path.append(training_path)

# 2. GEREKSİZ IMPORT HATALARINI ENGELLEME
try:
    import training.dataset.fwa_blend as fwa
    fwa.face_detector = None
    fwa.face_predictor = None
except ImportError:
    pass # Eğer import edilemezse sorun yok, zaten kullanmayacağız

# 3. MODELİ İÇE AKTARMA (UCFDetector)
try:
    from detectors.ucf_detector import UCFDetector
except ImportError:
    try:
        sys.path.append(os.path.join(training_path, 'detectors'))
        from ucf_detector import UCFDetector
    except ImportError:
        print("HATA: Model dosyası (ucf_detector.py) bulunamadı.")
        exit()

# --- MODEL YÜKLEME FONKSİYONU ---
def load_my_model():
    # Dosya yolları
    config_path = 'my_test_config.yaml'
    checkpoint_path = './training/checkpoints/ckpt_best.pth'
    
    if not os.path.exists(config_path):
        print(f"HATA: Config bulunamadı: {config_path}")
        return None
    
    # Config yükle
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Checkpoint yolunu ayarla
    config['checkpoint_path'] = checkpoint_path

    print(">>> Model yükleniyor, lütfen bekleyin...")
    try:
        model = UCFDetector(config)
        
        # Ağırlıkları yükle
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        print(">>> Model ve Ağırlıklar Başarıyla Yüklendi!")
        return model
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None

# Global Model (Uygulama açılınca bir kere yüklenir)
global_model = load_my_model()

# --- GÖRÜNTÜ İŞLEME ---
def preprocess_frame(frame, image_size=256):
    transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    augmented = transform(image=frame)
    return augmented['image'].unsqueeze(0)

# --- ANALİZ FONKSİYONU ---
def analyze_video(video_path):
    if global_model is None:
        return {"Model Yüklenemedi": 0.0}
    
    if video_path is None:
        return {"Lütfen Video Seçin": 0.0}
    
    cap = cv2.VideoCapture(video_path)
    preds = []
    frame_count = 0
    
    # Kullanıcıya bilgi ver (Gradio loglarında görünür)
    print(f"Analiz başlıyor: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Hızlandırma: Her 30 karede bir analiz
        if frame_count % 30 == 0:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess_frame(frame_rgb)
                
                # Dummy label
                dummy_label = torch.tensor([0]).long()
                data_dict = {'image': input_tensor, 'label': dummy_label}
                
                with torch.no_grad():
                    output_dict = global_model(data_dict, inference=True)
                    # Çıktıdan olasılığı al
                    if isinstance(output_dict, dict) and 'cls' in output_dict:
                        logits = output_dict['cls']
                    else:
                        logits = output_dict # Bazı durumlarda direkt tensor dönebilir

                    prob = torch.softmax(logits, dim=1)
                    fake_prob = prob[0][1].item()
                    preds.append(fake_prob)
            except Exception as e:
                # print(f"Kare hatası: {e}")
                pass
        frame_count += 1
    
    cap.release()
    
    if len(preds) == 0:
        return {"Video Okunamadı": 0.0}
        
    avg_fake_prob = sum(preds) / len(preds)
    avg_real_prob = 1.0 - avg_fake_prob
    
    print(f"Analiz Bitti. Fake Skoru: {avg_fake_prob:.4f}")
    
    # Sözlük döndür: {Etiket: Olasılık}
    return {"FAKE (Sahte)": avg_fake_prob, "REAL (Gerçek)": avg_real_prob}

# --- ARAYÜZ ---
interface = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Videoyu Buraya Sürükleyin"),
    outputs=gr.Label(num_top_classes=2, label="Sonuç"),
    title="Deepfake Tespit Arayüzü",
    description="DeepfakeBench (UCF Modeli) ile geliştirilmiştir.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch(inbrowser=True)