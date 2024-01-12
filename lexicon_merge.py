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
