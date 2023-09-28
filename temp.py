"""Some tests on the dataset, maybe this code could be added to the data_collection somewhere
"""
import json
from tqdm import tqdm



# with open("/media/livechat/dataset/train.json") as train:
#     train_json = json.load(train)

# with open("/media/livechat/dataset/test.json") as test:
#     test_json = json.load(test)

# with open("dataset/candidate_comments.json") as candidates:
#     candidates_json = json.load(candidates)

# print(len(candidates_json), len(train_json)+len(test_json))

# comments_json = test_json


# # 1817767920 1773040627 1817565106 1787952512
# for comment in tqdm(test_json):
#     for candidate in candidates_json:
#         if comment["id_video"]==candidate["id_video"] and \
#         comment["start"]==candidate["start"] and \
#         comment["offset_start"]==candidate["offset_start"]:
#             if len(candidate["candidates"])==91:
#                 comment["candidates"] = candidate["candidates"]
#             else:
#                 print("not ok")

# with open("dataset/test_with_candidates.json", "w") as output:
#     json.dump(test_json, output)

with open("dataset/test_emotes_candidates.json") as train:
    train_json = json.load(train)

reduced_train = []

for element in tqdm(train_json):

    index_to_remove = []
    for index, comment in enumerate(element["chat_context"]):
        comment = comment.lower()
        for word in comment.split(" "):
            if word.startswith("www") or ("http" in word):
                index_to_remove.append(index)
                break
        if "@user" in comment:
            element["chat_context"][index] = comment.replace("@user", "")
    for index in reversed(index_to_remove):
        element["chat_context"].pop(index)
    
    index_to_remove = []
    for index, comment in enumerate(element["responses"]):
        comment = comment.lower()
        for word in comment.split(" "):
            if word.startswith("www") or ("http" in word):
                index_to_remove.append(index)
                break
        if "@user" in comment:
            element["responses"][index] = comment.replace("@user", "")
    for index in reversed(index_to_remove):
        element["responses"].pop(index)        
    
    if element["chat_context"] != [] and  element["responses"] != []:
        reduced_train.append(element)

with open("dataset/test_reduced_candidates.json", "w") as output:
    json.dump(reduced_train, output, indent=4)
