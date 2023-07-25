import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import download
download('stopwords')
download('wordnet')
from nltk.corpus import stopwords
from wordcloud import WordCloud
import json
from nltk.stem import WordNetLemmatizer
# task 1 - loading
data = pd.read_json('recipes.json', lines=True)
data.head(5)
data.info()
data['Ingredients'].head(5)
# task 2 - cleaning
data['Ingredients']=data['Ingredients'].map(lambda x: re.sub(r'[^a-zA-Z]',' ', str(x)))
data = data.dropna(subset=['Ingredients'])
stop = stopwords.words('english') + ['tsp','tbsp','finely','extra','chopped']
stop
# task 3 - remove encoding
def remove_encoding_word(word):
    word=str(word)
    word=word.encode('ASCII','ignore').decode('ASCII')
    return word

def remove_encoding_text(text):
    text = str(text)
    text = ' '.join(remove_encoding_word(word) for word in text.split() if word not in stop)
    return text
# task 4 - define lemmatizing
data['Ingredients']=data['Ingredients'].apply(remove_encoding_text)
text=" ".join(words for words in data['Ingredients'])
len(text)
lemma= WordNetLemmatizer().lemmatize
lemma('leaves')
# task 5 - fit and transform text, with or without lemmatizing
def tokenize(document):
    tokens = [lemma(w) for w in document.split() if len(w)>3 and w.isalpha()]
    return tokens
vectorizer = TfidfVectorizer(tokenizer = tokenize, ngram_range = (2,2), stop_words = stop, strip_accents='unicode')
tdm=vectorizer.fit_transform(data['Ingredients'])
vectorizer.vocabulary_.items()
# task 6 -  get word frequencies and create wordcloud
tfidf_weights=[(word,tdm.getcol(idx).sum()) for word, idx in vectorizer.vocabulary_.items()]
tfidf_weights=dict(tfidf_weights)
w=WordCloud(width=1500,height=1200, mode='RGBA', background_color='white', max_words=2000).fit_words(dict(tfidf_weights))
plt.figure(figsize=(20,15))
plt.imshow(w)
plt.axis('off')
plt.savefig('recipes_wordcloud.png')
txtFile = open("output.txt","w")
sortedWeights = sorted(tfidf_weights.items(), key=lambda x:x[1],reverse=True)
txtFile.write("Ingredient      :        Weight \n")
for key,values in sortedWeights:
    txtFile.write(str(key)+" : " +str(values) + "\n")
txtFile.close()

