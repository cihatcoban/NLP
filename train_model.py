from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# wiki.tr.txt çıktı dosyasından satırları okuma
sentences = list(LineSentence('C:\\Users\\cihat\\Downloads\\NLP\\wiki.tr.txt'))

# Word2Vec modelini eğitelim
model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=5, workers=4)

# Eğitilen modeli kaydedelim
model.wv.save_word2vec_format("C:\\Users\\cihat\\Downloads\\NLP\\turkish_word2vec_model.model", binary=True)
