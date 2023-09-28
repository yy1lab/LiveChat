"""Some more tests on the dataset, maybe this code could be added to the data_collection somewhere
"""
import json
with open("/media/livechat/dataset/train_emotes.json") as train:
    train_json = json.load(train)

with open("/media/livechat/dataset/test_emotes.json") as test:
    test_json = json.load(test)


nb_comments = 0
nb_words = 0
nb_char = 0



print(len(train_json))
for item in train_json:
    comments = item["chat_context"]
    nb_comments+=len(comments)
    words = [word for comment in comments for word in comment.split()]
    nb_words+=len(words)
    nb_char+=sum([len(word) for word in words])
    
for item in test_json:
    comments = item["chat_context"]
    nb_comments+=len(comments)
    words = [word for comment in comments for word in comment.split()]
    nb_words+=len(words)
    nb_char+=sum([len(word) for word in words])

print(nb_comments, nb_words, nb_char)
print(nb_char/nb_words)