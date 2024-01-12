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
    inputFile = "C:\\Users\\cihat\\Downloads\\NLP\\trwiki-20240101-pages-articles .xml.bz2"
    outputFile = "C:\\Users\\cihat\\Downloads\\NLP\\wiki.tr.txt"

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

    
