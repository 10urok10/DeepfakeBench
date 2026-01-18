## ⚖️ Lisans ve Referanslar (License & Acknowledgements)

Bu proje, **[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)** altyapısı kullanılarak geliştirilmiştir. 

Orijinal Proje:
> Yan, Z., et al. "DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection."

Bu çalışma eğitim amaçlıdır ve orijinal projenin lisans kurallarına tabidir.


# 🕵️ Deepfake Tespit Sistemi (UCF Model - CPU Optimize)

Bu proje, **DeepfakeBench** altyapısını kullanarak geliştirilmiş, **UCF (Uncovering Common Features)** modelini temel alan bir deepfake tespit sistemidir.

Proje, özellikle **NVIDIA ekran kartı olmayan (veya uyumsuz olan)** bilgisayarlarda **CPU üzerinde** çalışacak şekilde optimize edilmiştir. Kullanıcı dostu bir arayüz (Gradio) ve komut satırı test aracı içerir.

---

## ⚠️ Kritik Ön Bilgilendirme (Versiyon Uyumluluğu)

Bu proje kütüphane versiyonlarına karşı çok hassastır. Rastgele `pip install` yapmak projenin çalışmasını bozar. Lütfen aşağıdaki versiyon uyarılarını dikkate alın:

* **Python Sürümü:** Kesinlikle **Python 3.8** kullanılmalıdır. (3.9 veya 3.10 ile `dlib` ve `numpy` uyumsuzlukları yaşanabilir).
* **Dlib Kütüphanesi:** Windows üzerinde doğrudan `pip install dlib` komutu genellikle hata verir (C++ derleyicisi ister). Bu yüzden `Conda` üzerinden kurulacaktır.
* **Numpy & Scipy:** `numpy` sürümü 1.21.5'ten yüksek olursa `scipy` ile çakışma yaşanır ve proje açılmaz.

---

## 🛠️ Adım Adım Kurulum Rehberi

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları sırasıyla uygulayın.

### 1. Gereksinimler
* **Anaconda** veya **Miniconda** (Sanal ortam yönetimi için şart).
* **Git** (Bazı kütüphanelerin GitHub'dan çekilmesi için).

### 2. Sanal Ortamın Oluşturulması
Anaconda Prompt (veya terminalinizi) açın ve temiz bir ortam kurun:

Powershell değil cmd kullanın.

# Reponun çekilmesi.
git clone https://github.com/10urok10/DeepfakeBench

# 1. Python 3.8 tabanlı ortamı oluşturulması
conda create -n DeepfakeProje python=3.8

# 2. Ortamın aktif edilmesi
conda activate DeepfakeProje

# 3. PyTorch (CPU Sürümü) kuralulması
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Conda üzerinden dlib kurulması
conda install -c conda-forge dlib

# 5. Diğer kütüphanelerin kurulması
pip install -r requirements.txt

# 6. Dosyaların uygun yere konulması
Hazır xception mimari dosyası
DeepfakeBench\training\pretrained\xception-b5690688.pth
xception-b5690688.pth dosyayı bu uzantıya yerleştirin.

Model dosyası
training\checkpoints\ckpt_best.pth
checkpoints klasörü oluşturun ve içine indireceğiniz ckpt_best.pth dosyasını koyun.

xception-b5690688.pth: 
https://drive.google.com/file/d/19YwmzGBBdJ0P7e_AhVJ_7oklN0zbm789/view?usp=sharing

ckpt_best.pth
https://drive.google.com/file/d/1njZPtGH12WrBZdGa9etGNxASCGxWRoKW/view?usp=sharing


# 7. Çalıştırma
Arayüz ile çalıştırmak için 
python app.py

Birden fazla video için 
training\my_datasets\test klasörü içine videolarını koyun
python predict.py
