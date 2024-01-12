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