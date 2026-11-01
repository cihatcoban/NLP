# Türkçe Word2Vec Eğitimi ve Konotasyon (Connotation) Analiz Motoru

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Keystone%20Framework-orange.svg)](docs/ARCHITECTURE.md)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Bu proje, Türkçe Vikipedi dökümü (Wikipedia dump) üzerinde **Word2Vec** kelime gömme (word embedding) modelinin eğitilmesi ve elde edilen anlamsal komşulukların **NRC-VAD** ve **MEmoLon** konotasyon sözlükleri (Valence, Arousal, Dominance ve temel duygular: Neşe, Öfke, Üzüntü, Korku, İğrenme) üzerinden çok boyutlu analizini gerçekleştiren **üretim seviyesinde (production-grade)** bir Doğal Dil İşleme (NLP) kütüphanesidir.

---

## 🏗️ Mimari ve Dizin Yapısı (`src-layout`)

Modern Python paketleme standartlarına (PEP 517 / PEP 621) uygun `src-layout` dosyalama yapısı:

```text
nlp-connotation/
├── configs/
│   └── default_config.yaml         # Merkezi ve yapılandırılabilir YAML ayarları
├── data/                           # Veri dizinleri (.gitkeep ile korunur)
│   ├── raw/                        # Ham Vikipedi dökümleri ve sözlük dosyaları
│   ├── processed/                  # Temizlenmiş ve tokenize edilmiş metin dökümleri
│   ├── lexicons/                   # Birleştirilmiş konotasyon sözlükleri (my_Lexicon.csv)
│   ├── models/                     # Eğitilmiş ikili (binary) Word2Vec modelleri
│   └── results/                    # CSV özet ve detay raporları
├── docs/
│   ├── ARCHITECTURE.md             # Katmanlı mimari ve Mimari Karar Kayıtları (ADR)
│   └── CONSTRAINTS.md              # Sistem sınırları, bellek ve performans kısıtları
├── src/
│   └── nlp_connotation/
│       ├── core/                   # Yapılandırma, loglama, şemalar ve özel istisnalar
│       ├── data/                   # Türkçe tokenizasyon ve bellek-dostu akış (streaming)
│       ├── lexicons/               # Vektörel sözlük birleştirme ve O(1) indeksli arama
│       ├── models/                 # Model adaptörleri (Adapter Pattern) ve Word2Vec eğitici
│       ├── analysis/               # Benzerlik analizi ve top-k konotasyon ortalama hesaplama
│       └── cli.py                  # Click tabanlı komut satırı arayüzü (CLI)
├── tests/                          # Kapsamlı birim (unit) testleri
├── Makefile                        # Görev otomasyonu (test, lint, format, evaluate)
├── pyproject.toml                  # Modern paketleme ve bağımlılık tanımları
├── requirements.txt                # Üretim ortamı bağımlılıkları
├── requirements-dev.txt            # Geliştirme ve test bağımlılıkları
└── README.md                       # Detaylı dokümantasyon
```

---

## ⚡ Kıdemli Refaktoring Analizi (Öncesi vs. Sonrası)

| Alan | Öğrenci Kodu (Eski Hali) | Kıdemli / Keystone Mimarisi (Yeni Hali) |
| :--- | :--- | :--- |
| **Dosya Yolları** | Sabitlenmiş Windows yolları (`C:\Users\cihat\...`) | `pathlib.Path` ve YAML/CLI parametrik yapılandırması |
| **Sözlük Birleştirme** | $O(N \cdot M)$ `df.iterrows()` iç içe döngüsü (~15 dakika) | $O(N)$ Hash-join (`pd.merge`) ($<100\text{ ms}$) |
| **Bellek Yönetimi** | `list(LineSentence(...))` ile GB'larca veriyi RAM'e yükleme ($OOM$) | `CorpusSentenceIterator` ile satır satır akış ($O(1)$ RAM) |
| **Türkçe Karakter** | Temel replace haritası (noktalı/noktasız I hataları) | Unicode NFC normalizasyonu ve hatasız Türkçe küçültme |
| **Hata Yönetimi** | Tanımsız fonksiyonlar (`get_word_freq`), sessiz çökmeler | Hiyerarşik `NLPConnotationError` ve bağlamsal loglama |
| **Model Erişimi** | Farklı modellere dağınık kod blokları | `BaseModelAdapter` ile ortak sözlük ve benzerlik arayüzü |
| **Eksik Kelime Hesabı** | Sözlükte olmayan kelimelerin göz ardı edilmesinde belirsizlik | `requested_k` vs `found_k` açık raporlaması ve doğru ortalama |
| **Test & Kalite** | Sıfır test, sıfır tip tanımı | Kapsamlı birim testleri, type hinting, linter ve Makefile |

---

## 🚀 Hızlı Başlangıç

### 1. Ortam Kurulumu

Python 3.9+ yüklü bir sanal ortam oluşturun ve bağımlılıkları yükleyin:

```bash
# Sanal ortam oluşturma ve etkinleştirme
python3 -m venv .venv
source .venv/bin/activate  # Windows için: .venv\Scripts\activate

# Projeyi düzenlenebilir (editable) modda kurma
pip install -e ".[dev,training]"
```

### 2. Testleri Çalıştırma

Tüm birim testlerini tek bir komutla çalıştırın:

```bash
make test
# veya: PYTHONPATH=src python3 -m unittest discover tests -v
```

---

## 🛠️ Komut Satırı Arayüzü (CLI) Kullanımı

Paket, `nlp-connotation` komutu veya `python -m nlp_connotation.cli` ile doğrudan çalıştırılabilir.

### 1. Aşama: Vikipedi Dökümünü Çıkarma ve Tokenize Etme

Vikipedi `.xml.bz2` sıkıştırılmış dosyasını bellek dostu bir şekilde işleyip eğitilmeye hazır metin haline getirir:

```bash
nlp-connotation preprocess \
    --dump-path data/raw/trwiki-latest-pages-articles.xml.bz2 \
    --output-path data/processed/wiki.tr.txt
```

### 2. Aşama: Word2Vec Modelini Eğitme

Metin korpusu üzerinde optimize edilmiş parametrelerle Türkçe Word2Vec modelini eğitir:

```bash
nlp-connotation train \
    --corpus data/processed/wiki.tr.txt \
    --output data/models/turkish_word2vec.bin \
    --vector-size 300 \
    --window 5 \
    --epochs 5
```

### 3. Aşama: Konotasyon Sözlüklerini Birleştirme

**NRC-VAD** (Valence, Arousal, Dominance) ile **MEmoLon** (Joy, Anger, Sadness, Fear, Disgust) veri setlerini yüksek hızlı vektörel birleştirmeyle tek dosyada toplar:

```bash
nlp-connotation merge-lexicons \
    --nrc-path data/raw/Turkish-NRC-VAD-Lexicon.csv \
    --memolon-path data/raw/MTL_grouped/en.tsv \
    --output data/lexicons/my_Lexicon.csv
```

### 4. Aşama: Top-K Benzerlik ve Konotasyon Analizini Çalıştırma

Seçilen modeller ve anahtar kelimeler (`spor`, `magazin`, `siyaset`, `ekonomi`) için en yakın $k \in \{1, 3, 10\}$ kelimeleri çıkarır, sözlükte karşılığı bulunan kelimelerin duygusal ve VAD ortalamalarını hesaplar:

```bash
nlp-connotation evaluate \
    --lexicon data/lexicons/my_Lexicon.csv \
    --output-dir data/results
```

**Oluşturulan Çıktı Dosyaları:**
- `data/results/connotation_summary_report.csv`: Model, kelime ve $k$ seviyesi bazında özet metrikler (`requested_k`, `found_k`, `mean_valence`, `mean_arousal`, `mean_dominance`, duygu ortalamaları).
- `data/results/neighbor_details_report.csv`: Tespit edilen her bir komşu kelimenin benzerlik skoru, sözlükte bulunma durumu ve tekil metrikleri.

---

## 🧪 Analiz ve Eksik Kelime (Missing Token) Yaklaşımı

Ödev şartnamesinde belirtildiği üzere:
> *"Eğer sözlükten ilgili kelimeye erişemiyorsanız, o kelimenin yokluğunu göz ardı edin. Örneğin, top 3 kelime arıyorsanız ve bir kelimeyi sözlükte bulamadığınız için çıkarıyorsanız, top kelime sayısını 2 olarak yazdırın."*

`ConnotationAnalyzer` motorumuz:
1. Modelden talep edilen ilk $k$ komşuyu alır (`requested_k`).
2. Her bir komşuyu $O(1)$ sözlük indeksinde arar.
3. Sözlükte bulunamayan kelimeleri yapay sıfır puanlarla ortalamayı bozmadan eler.
4. Yalnızca sözlükte var olan kelimelerin aritmetik ortalamasını hesaplar ve sözlükte bulunan kelime adedini (`found_k`) şeffaf biçimde raporlar.

---

## 📖 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında lisanslanmıştır.
