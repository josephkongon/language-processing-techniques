import spacy
from gensim import corpora, models
from gensim.models import CoherenceModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd


nlp = spacy.load('en_core_web_sm')


corpus = pd.read_csv('data.csv')['text'].tolist()

processed_corpus = []
for text in corpus:
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    processed_corpus.append(tokens)

# Create dictionary and corpus
dictionary = corpora.Dictionary(processed_corpus)
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]


num_topics_range = range(2, 11)
passes_range = [10, 15, 20]

best_score = 0.0
best_num_topics = 0
best_passes = 0

for num_topics in num_topics_range:
    for passes in passes_range:
        lda_model = models.LdaModel(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=passes)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_corpus, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        
        if coherence_score > best_score:
            best_score = coherence_score
            best_num_topics = num_topics
            best_passes = passes

print(f"Best Coherence Score: {best_score}")
print(f"Best Number of Topics: {best_num_topics}")
print(f"Best Number of Passes: {best_passes}")

# Final LDA model with best parameters
lda_model = models.LdaModel(bow_corpus, num_topics=best_num_topics, id2word=dictionary, passes=best_passes)
topics = lda_model.print_topics(num_words=5)

print("\nIdentified Topics:")
topic_keywords = []
for topic_idx, topic in topics:
    print(f"Topic {topic_idx}: {topic}")
    topic_keywords.extend(topic.split())

# Named Entity Recognition
print("\nNamed Entities:")
entities = []
for text in corpus:
    doc = nlp(text)
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
        print(ent.text, ent.label_)

# Word Cloud Visualization for Entities and Topics
entity_text = ' '.join([ent[0] for ent in entities])
topic_text = ' '.join(topic_keywords)
combined_text = entity_text + ' ' + topic_text
wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(combined_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
