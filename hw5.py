import spacy
import spacy.displacy as displacy
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation
from newsapi import NewsApiClient
from collections import Counter
from wordcloud import WordCloud

nlp = spacy.load('en_core_web_sm')
newsapi = NewsApiClient(api_key = 'd81405ff96dd4b31b2685a31d5c0e376')
dados = {}
titles = []
dates = []
descriptions = []
contents = []
results = []

temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2022-03-05', to='2022-03-25', sort_by='relevancy', page_size=100)

articles = temp['articles']
# print(articles)

for i, article in enumerate(articles):
    titles.append(article['title'])
    dates.append(article['publishedAt'].split("T")[0])
    descriptions.append(article['description'])
    contents.append(article['content'])
    dados.update({'title': titles, 'date': dates, 'desc': descriptions, 'content': contents})

df = pd.DataFrame(dados)
df = df.dropna()
df.head()

def get_keywords(text):
    result = []
    pos_tag = ['VERB', 'NOUN', 'PROPN']
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue

        if(token.pos_ in pos_tag):
            result.append(token.text)

    return result

for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords(content)).most_common(5)])
df['keywords'] = results

#display(df)
df.to_csv('data.csv', index=False)

text = ''
for x in results:
  for y in x:
    text += " " + y
wordcloud = WordCloud(max_font_size=50, max_words=200, background_color="black", colormap='tab20c').generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
