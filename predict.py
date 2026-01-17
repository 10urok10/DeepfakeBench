import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import yaml
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------------------------------------------------
# 1. YOL VE IMPORT AYARLARI
# -------------------------------------------------------------------
# Python'ın 'training' klasörünü ana dizin gibi görmesini sağlıyoruz.
# Böylece 'from detectors...' veya 'from networks...' komutları çalışacak.
current_dir = os.getcwd()
training_path = os.path.join(current_dir, 'training')
sys.path.append(training_path)

try:
    # Sizin paylaştığınız dosya yolundan UCFDetector'ü çağırıyoruz
    from detectors.ucf_detector import UCFDetector
    print(">>> 'UCFDetector' sınıfı başarıyla bulundu.")
except ImportError as e:
    # Alternatif yol denemesi (Klasör yapısına göre değişebilir)
    try:
        sys.path.append(os.path.join(training_path, 'detectors'))
        from ucf_detector import UCFDetector
        print(">>> 'UCFDetector' alternatif yoldan bulundu.")
    except ImportError:
        print(f"\nKRİTİK HATA: UCFDetector import edilemedi.\nDetay: {e}")
        print("Lütfen terminalin 'DeepfakeBench' ana klasöründe olduğundan emin olun.")
        exit()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_model(config, checkpoint_path):
    print(f"Model Başlatılıyor: UCFDetector")
    
    # 1. Modeli Başlat
    # Config dosyasını modele yediriyoruz
    try:
        model = UCFDetector(config)
    except Exception as e:
        print(f"Model başlatma hatası: {e}")
        print("Config dosyasındaki parametreler modelle uyuşmuyor olabilir.")
        exit()
    
    # 2. Ağırlıkları Yükle
    print(f"Ağırlıklar Yükleniyor: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {checkpoint_path}")

    # CPU'ya map ederek yüklüyoruz
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # İsim temizliği (module. öneklerini kaldır)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    # Modele yükle (strict=False önemli, bazı loss katmanları checkpointte olmayabilir)
    model.load_state_dict(new_state_dict, strict=False)
    
    model.eval()
    return model

def preprocess_frame(frame, image_size=256):
    # Görüntü Ön İşleme (Resize + Normalize)
    transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    augmented = transform(image=frame)
    return augmented['image'].unsqueeze(0) # Batch boyutu ekle (1, 3, 256, 256)

def predict_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {video_path}")
        return 0.5

    preds = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Hızlandırma: Her 30 karede bir analiz (Saniyede ~1 kare)
        if frame_count % 30 == 0:
            try:
                # OpenCV BGR okur, RGB'ye çevir
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = preprocess_frame(frame_rgb)
                
                # --- KRİTİK KISIM ---
                # Model 'inference=True' modunda bile olsa bir 'label' bekliyor.
                # O yüzden sahte bir etiket (0) oluşturup veriye ekliyoruz.
                dummy_label = torch.tensor([0]).long() 
                data_dict = {'image': input_tensor, 'label': dummy_label}
                
                with torch.no_grad():
                    # Modeli inference modunda çağır
                    output_dict = model(data_dict, inference=True)
                    
                    # Çıktıyı al ('cls' anahtarı logits değerini tutar)
                    logits = output_dict['cls']
                    
                    # Softmax ile olasılığa çevir
                    prob = torch.softmax(logits, dim=1)
                    
                    # 1. index 'FAKE' sınıfıdır (Genellikle 0: Real, 1: Fake)
                    fake_prob = prob[0][1].item() 
                    preds.append(fake_prob)
                    
            except Exception as e:
                # print(f"Hata (Kare {frame_count}): {e}") 
                pass
        
        frame_count += 1
    
    cap.release()
    
    if len(preds) == 0:
        return 0.5 # Okunamazsa nötr dön
        
    # Videonun genel skoru (Karelerin ortalaması)
    avg_fake_prob = sum(preds) / len(preds)
    return avg_fake_prob

def main():
    # --- AYARLAR ---
    config_file = 'my_test_config.yaml'
    # Config dosyasında checkpoint_path yoksa burası kullanılır
    default_ckpt = './training/checkpoints/ckpt_best.pth'
    test_folder = r"C:\Users\10ur\Desktop\final\DeepfakeBench\training\my_dataset\test"
    # ----------------
    
    # 1. Config Yükle
    if not os.path.exists(config_file):
        print(f"HATA: Config dosyası bulunamadı: {config_file}")
        return
    config = load_config(config_file)
    
    # Checkpoint yolunu ayarla
    if 'checkpoint_path' not in config:
        config['checkpoint_path'] = default_ckpt

    # 2. Modeli Hazırla
    try:
        model = prepare_model(config, config['checkpoint_path'])
        print("\n>>> Model başarıyla yüklendi! Test başlıyor...\n")
    except Exception as e:
        print(f"\nKRİTİK HATA: Model yüklenirken sorun oluştu.\nDetay: {e}")
        return

    # 3. Videoları Test Et
    if not os.path.exists(test_folder):
        print(f"Klasör bulunamadı: {test_folder}")
        return

    video_files = [f for f in os.listdir(test_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if len(video_files) == 0:
        print("Klasörde video dosyası bulunamadı.")
        return

    print("-" * 60)
    print(f"{'VIDEO ADI':<35} | {'SONUÇ':<10} | {'FAKE SKORU'}")
    print("-" * 60)

    for vid in video_files:
        path = os.path.join(test_folder, vid)
        try:
            score = predict_video(model, path)
            
            # Eşik değeri 0.5 (Üstü Fake, Altı Real)
            if score > 0.5:
                label = "FAKE"
            else:
                label = "REAL"
            
            print(f"{vid:<35} | {label:<10} | %{score*100:.2f}")
            
        except Exception as e:
            print(f"{vid:<35} | HATA       | {e}")

if __name__ == '__main__':
    main()