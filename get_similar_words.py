from gensim.models import Word2Vec
import csv

model_path = "C:\\Users\\cihat\\Downloads\\NLP\\turkish_word2vec_model.model"
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
output_file = "C:\\Users\\cihat\\Downloads\\NLP\\similar_words_results.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Keyword', 'TopN', 'Similar Words']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for result in results:
        writer.writerow({'Keyword': result[0], 'TopN': result[1], 'Similar Words': result[2]})

print(f"Sonuçlar CSV dosyasına {output_file} olarak kaydedildi.")
