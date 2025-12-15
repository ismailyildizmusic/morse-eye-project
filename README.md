# 👁️ MORSE-EYE: Göz Hareketleri ile Mors Kodu İletişim Sistemi

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![TÜBİTAK](https://img.shields.io/badge/Proje-TÜBİTAK%202204--A-green)
![License](https://img.shields.io/badge/Lisans-MIT-yellow)

> **"Gözlerinizle konuşun, sınırları kaldırın."**

Bu proje, **TÜBİTAK 2204-A Lise Öğrencileri Araştırma Projeleri Yarışması** kapsamında geliştirilmiştir. Konuşma ve hareket yeteneğini kaybetmiş (ALS, Felç, Serebral Palsi vb.) bireylerin, sadece göz hareketlerini kullanarak Mors alfabesi üzerinden iletişim kurmalarını sağlayan yapay zeka destekli bir web uygulamasıdır.

---

## 🎯 Projenin Amacı

Dünya genelinde milyonlarca insan nörolojik hastalıklar nedeniyle iletişim kurma yetisini kaybetmektedir. Mevcut göz takip sistemlerinin yüksek maliyeti (40.000 TL - 80.000 TL), bu teknolojiye erişimi kısıtlamaktadır.

**MORSE-EYE Projesinin Hedefleri:**
1.  **Erişilebilirlik:** Düşük maliyetli ve ekstra donanım gerektirmeyen (sadece web kamera) bir çözüm sunmak.
2.  **Yapay Zeka:** Yerli ve özgün bir algoritma ile göz hareketlerini %95+ doğrulukla tespit etmek.
3.  **İletişim:** Göz kırpma ve bakış yönleri ile Mors kodunu metne, metni ise sese çevirmek.

---

## 🚀 Özellikler

* **Gerçek Zamanlı Göz Takibi:** MediaPipe Face Mesh teknolojisi ile hassas iris takibi.
* **Mors Kodu Entegrasyonu:**
    * ⬅️ **Sola Bakış:** Nokta ( • )
    * ➡️ **Sağa Bakış:** Çizgi ( − )
* **Akıllı Kontrol Sistemi:**
    * 😉 **2 Kez Kırpma:** Harfi Onayla
    * 😌 **3 Kez Kırpma:** Boşluk Ekle
    * 😑 **Uzun Kırpma / Aşağı Bakış:** Mesajı Sesli Oku (TTS)
* **Web Tabanlı Arayüz:** Streamlit sayesinde kurulum gerektirmeden tarayıcıda çalışır.
* **Sesli Geri Bildirim:** Yazılan mesajları sesli olarak okur (Text-to-Speech).

---

## 🛠️ Kurulum ve Çalıştırma

Bu projeyi kendi bilgisayarınızda veya Streamlit Cloud üzerinde çalıştırmak için aşağıdaki adımları izleyin.

### 1. Gereksinimler
Projenin çalışması için `requirements.txt` dosyasındaki kütüphanelerin yüklü olması gerekir:
* opencv-python
* mediapipe
* numpy
* streamlit
* pyttsx3 (veya gTTS)

### 2. Yerel Kurulum (Local)
Terminal veya Komut İstemcisi'ni açın ve şu komutları girin:

```bash
# Depoyu klonlayın
git clone [https://github.com/kullaniciadiniz/morse-eye-project.git](https://github.com/kullaniciadiniz/morse-eye-project.git)

# Klasöre gidin
cd morse-eye-project

# Kütüphaneleri yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
streamlit run app.py
