# Türkçe Word2Vec Modeli ve Konotasyon Sözlükleri Üzerine Çalışma

Bu projede, Türkçe Word2Vec modelini geliştireceğiz ve konotasyon sözlüklerini kullanarak kelime benzerliklerini analiz edeceğiz.

## 1. Aşama: Word2Vec Modelinin Geliştirilmesi
İlk olarak, Gensim kütüphanesi [Google](https://radimrehurek.com/gensim/) kullanılarak  Wikipedia dump  [Google]( https://dumps.wikimedia.org/trwiki/) üzerinde Türkçe Word2Vec modelini geliştireceğiz.
Bu işlem için bu [Google](https://github.com/akoksal/Turkish-Word2Vec/wiki/)  adımları takip edeceğiz. 
Haydi Wikipedia dump da aldığımız veriyi kullanarak modelimizi eğitmeden önce verimizi türkçe dilyapısa uyacak bir hale gelmesi için verimizle biraz ilgilenelim.
\`\`\`python
    from gensim.corpora import WikiCorpus
    from gensim.models import Word2Vec
    from multiprocessing import cpu_count
    from gensim import utils
    import logging
    import os
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="gensim.utils")
    
    # Türkçe tokenleştirme fonksiyonu
    def tokenize_tr(content, token_min_len=2, token_max_len=50, lower=True):
        if lower:
            lowerMap = {ord(u'A'): u'a', ord(u'B'): u'b', ord(u'C'): u'c', ord(u'Ç'): u'ç', ord(u'D'): u'd',
                        ord(u'E'): u'e', ord(u'F'): u'f', ord(u'G'): u'g', ord(u'Ğ'): u'ğ', ord(u'H'): u'h',
                        ord(u'I'): u'ı', ord(u'İ'): u'i', ord(u'J'): u'j', ord(u'K'): u'k', ord(u'L'): u'l',
                        ord(u'M'): u'm', ord(u'N'): u'n', ord(u'O'): u'o', ord(u'Ö'): u'ö', ord(u'P'): u'p',
                        ord(u'R'): u'r', ord(u'S'): u's', ord(u'Ş'): u'ş', ord(u'T'): u't', ord(u'U'): u'u',
                        ord(u'Ü'): u'ü', ord(u'V'): u'v', ord(u'Y'): u'y', ord(u'Z'): u'z'}
            content = content.translate(lowerMap)
        return [
            utils.to_unicode(token) for token in utils.tokenize(content, lower=False, errors='ignore')
            if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
        ]
    
    if __name__ == '__main__':
        # Dosya yolları
        inputFile = "dosya_yolu\\trwiki-20240101-pages-articles .xml.bz2"
        outputFile = "dosya_youlu\\wiki.tr.txt"
    
        # Wikipedia dump dosyasını işleme ve temizlenmiş metni çıkış dosyasına yazma
        wiki = WikiCorpus(inputFile, tokenizer_func=tokenize_tr)
        logging.info("Wikipedia dump is opened.")
        
        # Output dosyasının bulunduğu dizini oluşturalım
        output_directory = os.path.dirname(outputFile)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
        output = open(outputFile, "w", encoding="utf-8")
        logging.info("Output file is created.")
    
        i = 0
        for text in wiki.get_texts():
            output.write(" ".join(text) + "\n")
            i += 1
            if i % 10000 == 0:
                logging.info("Saved " + str(i) + " articles.")
    
        output.close()
\`\`\`
Kodummuzda görüldüğü üzere türkçe tokenize etme işlemi yaptıktan sonra Wikipedia dump dosyasını işleme ve temizlenmiş metni çıkış dosyasına yazma işlemi yapıyor 
ve ardından "wiki.tr.txt" adında bir dosyaya hazır veriyi kayıt ediyoruz.
hazırlanan veriyi artık modelimizde kullanmaya hazırız.
\`\`\`python
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# wiki.tr.txt çıktı dosyasından satırları okuma
sentences = list(LineSentence('dosya_yolu\\wiki.tr.txt'))

# Word2Vec modelini eğitelim
model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=5, workers=4)

# Eğitilen modeli kaydedelim
model.wv.save_word2vec_format("dosya_yolu\\turkish_word2vec_model.model", binary=True)
\`\`\`

modelimizi eğitiğimize göre şimdi ufak bir test yapalım. Verilen kelimelere en çok benzeyen ilk 1, ilk 3 ve ilk 10 kelimeyi bu fonksiyon ile elde etmeye çalışalım.

\`\`\`python
from gensim.models import Word2Vec
import csv

model_path = "dosya_yolu\\turkish_word2vec_model.model"
turkish_model = Word2Vec.load(model_path)

# İlgili kelimeler
keywords = ['spor', 'magazin', 'siyaset', 'ekonomi']

# Sonuçları saklamak için bir liste
results = []

# Her kelime için topn=1, topn=3 ve topn=10 durumlarını inceleyelim
for keyword in keywords:
    # Topn=1
    top1_similar = turkish_model.wv.most_similar(keyword, topn=1)
    results.append((keyword, 1, top1_similar[0][0]))

    # Topn=3
    top3_similar = turkish_model.wv.most_similar(keyword, topn=3)
    results.append((keyword, 3, ', '.join([word[0] for word in top3_similar])))

    # Topn=10
    top10_similar = turkish_model.wv.most_similar(keyword, topn=10)
    results.append((keyword, 10, ', '.join([word[0] for word in top10_similar])))

# Sonuçları CSV dosyasına yazma
output_file = "dosya_yolu\\similar_words_results.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Keyword', 'TopN', 'Similar Words']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for result in results:
        writer.writerow({'Keyword': result[0], 'TopN': result[1], 'Similar Words': result[2]})

print(f"Sonuçlar CSV dosyasına {output_file} olarak kaydedildi.")
\`\`\`

elde etiğimiz sonuçları görüntülemek için [Google](https://github.com/cihatcoban/NLP/blob/main/similar_words_results.csv) dan görüntüleyebilirsiniz.

Ayrıca, 2018'de eğitilmiş başka bir Türkçe modeli [Google](https://drive.google.com/drive/folders/1IBMTAGtZ4DakSCyAoA4j7Ch0Ft1aFoww)de kullanacağız.
Ek olarak, dört İngilizce Word embedding modeline ihtiyacımız olacak bu modellere [Google](https://radimrehurek.com/gensim/models/word2vec.html) üzerinden ulaşabilirsiniz:
fasttext-wiki-news-300, word2vec-google-news-300, glove-wiki-gigaword-300, glove-twitter-200 modellerini indirme işlemi yapabiliriz.
NOT:Bu modellerin boyutlarını, kendi bilgisayarınızda veya Google Colab'da çalıştırabilirlik durumuna göre ayarlayabilirsiniz.
Ayrıca [Google](https://github.com/piskvorky/gensim-data) adresini kontrol etmenizi de tavsiye ederim.
Bu aşamadan sonra, verilen kelimelere en çok benzeyen ilk 1, ilk 3 ve ilk 10 kelimeyi bu fonksiyon ile elde etmeye çalışacağız.

\`\`\`python
import gensim.downloader as api
import os
from gensim.models import KeyedVectors
import pandas as pd
from tqdm import tqdm

my_turkish_model = KeyedVectors.load_word2vec_format('turkish_word2vec_model.model', binary=True)
print(my_turkish_model.most_similar(positive=["abaküs","bilgisayar"],negative=[])[:3])
print(my_turkish_model.doesnt_match(["elma","ev", "konak", "apartman"]))
#print(word_vectors.n_similarity(['', ''], ['', ""])
other_turkish_model = KeyedVectors.load_word2vec_format('trmodel', binary=True)
print(other_turkish_model.most_similar(positive=["abaküs","bilgisayar"],negative=[])[:3])
print(other_turkish_model.doesnt_match(["elma","ev", "konak", "apartman"]))
turkish_models_dict = {
    "turkish_model_1":my_turkish_model,
    "turkish_model_2":other_turkish_model,
}

google_news_300                 = None
glove_twitter_200               = None
glove_wiki_gigaword_300         = None
fasttext_wiki_news_subwords_300 = None

if not os.path.exists("word2vec-google-news-300.model"):
    google_news_300=api.load("word2vec-google-news-300")
    google_news_300.save_word2vec_format("word2vec-google-news-300.model", binary=True)
    print("word2vec-google-news-300.model created")
else:
    google_news_300 = KeyedVectors.load_word2vec_format('word2vec-google-news-300.model', binary=True)
    print("word2vec-google-news-300.model loaded")

if not os.path.exists("glove-twitter-200.model"):
    glove_twitter_200=api.load("glove-twitter-200")
    glove_twitter_200.save_word2vec_format("glove-twitter-200.model", binary=True)
    print("glove-twitter-200.model created")
else: 
    glove_twitter_200 = KeyedVectors.load_word2vec_format('glove-twitter-200.model', binary=True)
    print("glove-twitter-200.model loaded")

if not os.path.exists("glove-wiki-gigaword-300.model"):
    glove_wiki_gigaword_300=api.load("glove-wiki-gigaword-300")
    glove_wiki_gigaword_300.save_word2vec_format("glove-wiki-gigaword-300.model", binary=True)
else: 
    glove_wiki_gigaword_300 = KeyedVectors.load_word2vec_format('glove-wiki-gigaword-300.model', binary=True)
    print("glove-wiki-gigaword-300.model loaded")

if not os.path.exists("fasttext-wiki-news-subwords-300.model"):
    fasttext_wiki_news_subwords_300=api.load("fasttext-wiki-news-subwords-300")
    fasttext_wiki_news_subwords_300.save_word2vec_format("fasttext-wiki-news-subwords-300.model", binary=True)
else:
    fasttext_wiki_news_subwords_300 = KeyedVectors.load_word2vec_format('fasttext-wiki-news-subwords-300.model', binary=True)
    print("fasttext-wiki-news-subwords-300.model loaded")


english_models_dict = {
    "word2vec-google-news-300"        : google_news_300,
    "glove-twitter-200"               : glove_twitter_200,
    "glove-wiki-gigaword-300"         : glove_wiki_gigaword_300,
    "fasttext-wiki-news-subwords-300" : fasttext_wiki_news_subwords_300,
}

print("Loaded all models.")
\`\`\`
 Bu kod ile ihtiyacımız olan tüm modellerin kurulum işlemleri yapıyor ve modeleri türkçe ve ingilizce olmak üzere iki farklı sözlük yapısı altında topluyoruz.

 Şimdi sıra geldi test işlemi yapmamız için kullanacağımız kelimeleri modeller bazında topn1 ,topn3, topn10 şeklinde vektörel değerlerini bastıralım.
 
\`\`\`python
test_words = {
    "sport": {
      "Türkçe": "spor",
      "İngilizce": "sport",
    },
    "economy": {
      "Türkçe": "ekonomi",
      "İngilizce": "economy",
    },
    "magazine": {
      "Türkçe": "dergi",
      "İngilizce": "magazine",
    },
    "politics": {
      "Türkçe": "siyaset",
      "İngilizce": "politics",
    }
  }

k_list = [1, 3, 10]
  
import pandas as pd

if not os.path.exists("results.csv"):
    print("Creating results.csv")
    df_results = pd.DataFrame(columns=["Model", "Word", "List of k"])
    
    for k in k_list:
      print("k = ", k)
      for model_name in english_models_dict:
        print(model_name)
        model = english_models_dict[model_name]
        for word in test_words:
          list_of_k=model.most_similar(positive=[test_words[word]["İngilizce"]],topn=k)
          df_results.loc[len(df_results)] = [model_name, word, list_of_k]
          print(word)
          print(list_of_k)
          print("---------------------------------------------------")

    for k in k_list:
      print("k = ", k)
      for model_name in turkish_models_dict:
        print(model_name)
        model = turkish_models_dict[model_name]
        for word in test_words:
          list_of_k=model.most_similar(positive=[test_words[word]["Türkçe"]],topn=k)
          df_results.loc[len(df_results)] = [model_name, word, list_of_k]
          print(word)
          print(list_of_k)
          print("---------------------------------------------------")
    
    df_results.to_csv("results.csv", index=False)
else:
    print("results.csv already exists")
    pass

df_results = pd.read_csv("results.csv")
\`\`\`
Çıktılara göz gezdirmel için [Google](https://github.com/cihatcoban/NLP/blob/main/results.csv) adresini kontrol edebilirsiniz.

## 2. Aşama: Konotasyon Sözlüklerinin Kullanılması

İkinci aşamada, iki konotasyon sözlüğü seçeceğiz: NRC-VAD ve MEmoLon. İlk sözlük, valence, arousal ve dominance değerlerini içerirken, ikinci sözlük çeşitli duyguları işaret eden ek özelliklere sahiptir.

Her bir Word2Vec modeli için top-k kelimelerin bu leksikal boyutlarının ortalamalarını bulan bir tablo oluşturacağız. Test kelimeleri 'spor', 'magazin', 'siyaset', 'ekonomi' olacaktır. Bu işlem için bu fonksiyonları kullanabilir veya sözlüğü kendiniz ayrıştırabilirsiniz.

Eğer sözlükten ilgili kelimeye erişemiyorsanız, o kelimenin yokluğunu göz ardı edin. Örneğin, top 3 kelime arıyorsanız ve bir kelimeyi sözlükte bulamadığınız için çıkarıyorsanız, top kelime sayısını 2 olarak yazdırın.

Gerekli olan dökümanlara şu addresslerden ulaşabilirsiniz: [Google](https://saifmohammad.com/WebPages/nrc-vad.html) , [Google](https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip) , [Google](https://github.com/JULIELab/MEmoLon) ,[Google](https://zenodo.org/record/3756607/files/MTL_grouped.zip?download=1) , [Google](https://shifterator.readthedocs.io/en/latest/cookbook/sentiment_analysis.htm)

Gerekli dosyaları da indirdiğimize göre artık kelimelerin dilde gelen duygusal değerlerinin  karşılığını ölçmeye çalışacağız.
ilk olarak elimizde bulunan lexiconları birleştirelim ve verimizi tek çatı altında csv şeklinde kullanalım.
\`\`\`python
import os
from collections import Counter
from itertools import chain
import nltk
from nltk.corpus import stopwords
import pandas as pd
from tqdm import tqdm  # tqdm'yi doğru şekilde içeri aktardık
nltk.download('stopwords')

# Dosya yolunu belirle
corpusFile = "dosya_yolu_buraya"  # Gerçek dosya yolunu belirtmelisiniz

STOPWORDS = set(stopwords.words('turkish'))

df_NRC_VAD_Lexicon = None
corpusSeries = None
corpusFreq = None

if not os.path.exists("my_Lexicon.csv"):
    NRC_VAD_Lexicon_path = "NRC-VAD-Lexicon/NRC-VAD-Lexicon/OneFilePerLanguage/Turkish-NRC-VAD-Lexicon.csv"
    MTL_path = "MTL_grouped/en.tsv"
    df_NRC_VAD_Lexicon = pd.read_csv(NRC_VAD_Lexicon_path, sep=";")
    df_MTL = pd.read_csv(MTL_path, sep="\t")  # word,valence,arousal,dominance,joy,anger,sadness,fear,disgust
    df_NRC_VAD_Lexicon = df_NRC_VAD_Lexicon[["English Word", "Valence", "Arousal", "Dominance", "Turkish Word"]].dropna()

    for index, row in tqdm(df_NRC_VAD_Lexicon.iterrows(), total=df_NRC_VAD_Lexicon.shape[0]):
        mtl_index = df_MTL[df_MTL["word"] == row["English Word"]].index
        if not df_MTL[df_MTL["word"] == row["English Word"]].empty:
            df_NRC_VAD_Lexicon.loc[index, "Joy"] = df_MTL.loc[mtl_index, "joy"].values[0]
            df_NRC_VAD_Lexicon.loc[index, "Anger"] = df_MTL.loc[mtl_index, "anger"].values[0]
            df_NRC_VAD_Lexicon.loc[index, "Sadness"] = df_MTL.loc[mtl_index, "sadness"].values[0]
            df_NRC_VAD_Lexicon.loc[index, "Fear"] = df_MTL.loc[mtl_index, "fear"].values[0]
            df_NRC_VAD_Lexicon.loc[index, "Disgust"] = df_MTL.loc[mtl_index, "disgust"].values[0]

    df_NRC_VAD_Lexicon.to_csv("my_Lexicon.csv")
else:
    df_NRC_VAD_Lexicon = pd.read_csv("my_Lexicon.csv")

# Dosyayı oku
with open(corpusFile, 'r', encoding='utf-8') as f:
    corpusSeries = pd.Series(f.readlines())
    corpusFreq = get_word_freq(corpusSeries)
    print(corpusFreq.most_common(10))
\`\`\`
 Oluşan csc dosyasını : [Google](https://github.com/cihatcoban/NLP/blob/main/my_Lexicon.csv) adresinden inceleyebilirsiniz.

 Son olarak verimizi kullanalım ve karşılaştırma işlemlerini tamamlayalım.
 \`\`\`python
lex = pd.read_csv(
    "my_Lexicon.csv", 
    delimiter = ",", 
    index_col = 0, 
    header = 0,
  )
  #index,English Word,Valence,Arousal,Dominance,Turkish Word,Joy,Anger,Sadness,Fear,Disgust

lex = lex[["English Word","Turkish Word","Valence","Arousal","Dominance", "Joy","Anger","Sadness","Fear","Disgust"]]

import ast
comperatation = pd.DataFrame(columns=["Model","Word","English Word","Turkish Word","Valence","Arousal","Dominance", "Joy","Anger","Sadness","Fear","Disgust"])
counter=0
for index, row in df_results.iterrows():
    list_of_k=ast.literal_eval(row["List of k"])
    print(row["Model"])
    for i in list_of_k:
      print(i)
      if row["Model"] == "turkish_model_1" or row["Model"] == "turkish_model_2":
        index_list=lex.index[lex["Turkish Word"]==i[0]].tolist()
        for index in index_list:
          print(k_list[len(i)])
          comperatation.loc[counter, "Model"] = row["Model"]
          comperatation.loc[counter, "Word"] = row["Word"]
          comperatation.loc[counter, "English Word"] = lex.loc[index, "English Word"]
          comperatation.loc[counter, "Turkish Word"] = lex.loc[index, "Turkish Word"]
          comperatation.loc[counter, "Valence"] = lex.loc[index, "Valence"]
          comperatation.loc[counter, "Arousal"] = lex.loc[index, "Arousal"]
          comperatation.loc[counter, "Dominance"] = lex.loc[index, "Dominance"]
          comperatation.loc[counter, "Joy"] = lex.loc[index, "Joy"]
          comperatation.loc[counter, "Anger"] = lex.loc[index, "Anger"]
          comperatation.loc[counter, "Sadness"] = lex.loc[index, "Sadness"]
          comperatation.loc[counter, "Fear"] = lex.loc[index, "Fear"]
          comperatation.loc[counter, "Disgust"] = lex.loc[index, "Disgust"]
          #comperatation.loc[counter, "K Value"] = k_list[len(i)] TODO: add K value
          counter+=1
      else:
        index_list=lex.index[lex["English Word"]==i[0]].tolist()
        for index in index_list:
          print(k_list[len(i)])
          comperatation.loc[counter, "Model"] = row["Model"]
          comperatation.loc[counter, "Word"] = row["Word"]
          comperatation.loc[counter, "English Word"] = lex.loc[index, "English Word"]
          comperatation.loc[counter, "Turkish Word"] = lex.loc[index, "Turkish Word"]
          comperatation.loc[counter, "Valence"] = lex.loc[index, "Valence"]
          comperatation.loc[counter, "Arousal"] = lex.loc[index, "Arousal"]
          comperatation.loc[counter, "Dominance"] = lex.loc[index, "Dominance"]
          comperatation.loc[counter, "Joy"] = lex.loc[index, "Joy"]
          comperatation.loc[counter, "Anger"] = lex.loc[index, "Anger"]
          comperatation.loc[counter, "Sadness"] = lex.loc[index, "Sadness"]
          comperatation.loc[counter, "Fear"] = lex.loc[index, "Fear"]
          comperatation.loc[counter, "Disgust"] = lex.loc[index, "Disgust"]
          #comperatation.loc[counter, "K Value"] = k_list[len(i)] # TODO: add K value
          counter+=1
    counter+=1
comperatation.to_csv("comperatation.csv", index=False)
\`\`\`
 Elde etiğimiz son değerlendirmeye göz atmak için : [Google](https://github.com/cihatcoban/NLP/blob/main/comperatation.csv) adresini inceleye bilirsiniz .
