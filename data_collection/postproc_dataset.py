from collections import Counter
from dataclasses import asdict, dataclass
from itertools import chain
import json
import os
import numpy as np
import moviepy.editor as mp
from tqdm import tqdm
import random
from torchvision import transforms, models
import torch
import torch.nn as nn
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class DatasetElement:
    id_video: str
    start: int
    offset_start: int
    chat_context: list
    responses: list
    response_index: int
    transcript_audio: str
    context_emotes: list
    responses_emotes: list
    category: str
    path_to_video: str

def transcripts_audio(video_path: str, device:str="cpu"):
    """
    Generate transcripts for audio segments extracted from a video.

    This function takes the path of a video file and extracts 20 audio segments, each spanning 30 seconds.
    It then uses OpenAI's Whisper ASR (Automatic Speech Recognition) model to generate transcripts for each audio segment.

    The function returns a list of generated transcripts.

    @param video_path: str
        The path to the video file from which audio segments will be extracted.

    @param device: str, optional (default="cpu")
        The device on which to run the ASR model. Possible values: "cpu", "cuda".

    @return: List[str]
        A list of generated transcripts for each 30-second audio segment.
    """
    audios = []
    full_video = mp.VideoFileClip(video_path)
    video_length = full_video.duration
    for i in range(20):
        video = full_video.subclip(i*30, min(video_length, (i+1)*30))
        audio = video.audio
        temp_audio_file = f'temp/temp_audio_{str(i)}.wav'
        audio.write_audiofile(temp_audio_file, codec='pcm_s16le')
        audios.append(temp_audio_file)
    whisper = pipeline('automatic-speech-recognition', model = 'openai/whisper-medium', device=device)
    transcripts = whisper(audios)
    return transcripts

def video_transcripted(video, transcript_list):
    """
    Check if a video has been transcripted.

    This function takes a video dictionary and a list of transcript dictionaries as input. It checks if the
    video's ID matches any of the video IDs in the transcript list.

    If a transcript with the same video ID is found in the list, the function returns True, indicating that
    the video has been transcripted. Otherwise, it returns False.

    @param video: dict
        A dictionary representing the video. It should contain an 'id_video' field representing the video ID.

    @param transcript_list: List[dict]
        A list of dictionaries representing transcripts. Each dictionary should have an 'id_video' field representing
        the video ID to which the transcript belongs.

    @return: bool
        True if the video has been transcripted, False otherwise.
    """
    for transcript in transcript_list:
        if transcript["id_video"]==video["id_video"]:
            return True
    return False

def transcript_all(root: str):
    """
    Transcribe all the videos in the dataset and store the transcripts.

    This function iterates through all the videos in the dataset and transcribes each one using the 'transcripts_audio'
    function. The transcripts are stored in a list, and any encountered errors during the transcription process are recorded.

    The transcripts are written to the file 'dataset/transcripts.json'.

    @param root: str
        The root directory for the dataset.
    """
    transcripts_list = []
    errors = []
    with open(root+"dataset/headers/dataset.json", "r") as header:
        header_json = json.load(header)
    with open("dataset/temp/transcripts.json") as old_transcript:
        transcripts_list = json.load(old_transcript)
    for video_index, video in enumerate(header_json):
        print(video_index, "/", len(header_json))
        if not video_transcripted(video, transcripts_list):
            for timestep in sorted([int(ts) for ts in video["timesteps"]]):
                print(video["id_video"])
                try:
                    transcripts = transcripts_audio(root+"dataset/videos/"+video["category"]+"/"+video["id_video"]+"_"+str(timestep)+".mp4", device="cuda:0")
                    for i in range(20):
                        elem = {"id_video": video["id_video"], "start": timestep, "offset_start":i*30, "transcript_audio":transcripts[i]["text"], "category":video["category"]}
                        transcripts_list.append(elem)
                except OSError:
                    print("dataset/videos/"+video["category"]+"/"+video["id_video"]+"_"+str(timestep)+".mp4")
                    errors.append("dataset/videos/"+video["category"]+"/"+video["id_video"]+"_"+str(timestep)+".mp4")
            with open("dataset/temp/transcripts_temp.json", "w") as temp_output:
                json.dump(transcripts_list, temp_output, indent=4)
            with open("dataset/temp/files_errors.txt", "a") as err_file:
                print(errors, file=err_file)
    with open("dataset/transcripts.json", "w") as output:
        json.dump(transcripts_list, output, indent=4)

def compute_correlation(input_sentence, target_list):
    """
    Calculate the cosine similarity between an input sentence and a list of target sentences.

    This function uses the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer to transform the input
    sentence and target sentences into numerical vectors. It then calculates the cosine similarity between the
    input sentence and each target sentence in the list.

    The function returns the index of the target sentence in the list that has the highest cosine similarity
    to the input sentence.

    @param input_sentence: List[str]
        The input sentence to be compared with the target sentences.

    @param target_list: List[str]
        A list of target sentences to be compared with the input sentence.

    @return: int
        The index of the target sentence with the highest cosine similarity to the input sentence.
    """
    vectorizer = TfidfVectorizer()
    input_sentence = "".join(sent+" " for sent in input_sentence)
    try:
        vectors = vectorizer.fit_transform([input_sentence] + target_list)
        similarities = cosine_similarity(vectors[0], vectors[1:])
    except:
        print(input_sentence, target_list)
        return(0)
    return np.argmax(similarities[0])

def create_dataset_header(root: str):
    """
    Generate the dataset header from the video files in the specified root directory.

    This function creates a dataset header that contains information about each video in the dataset. It iterates through
    the video files in the specified root directory and extracts the video ID, category, and timesteps from the filenames.
    The information is then organized into a list of dictionaries, with each dictionary representing a video and
    containing the video's ID, category, and a list of timesteps.

    The dataset header is written to the file 'dataset/headers/dataset.json'.

    @param root: str
        The root directory for the dataset.
    """
    video_folder = root+"dataset/videos"
    categories = ["18122", "21779", "26936", "27471", "32399", "33214", "509658", "509660", "509663", "515025", "516575"]
    header_temp = {}
    for category in categories:
        folder_path = os.path.join(video_folder, category)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                if file.endswith(".mp4") and not file.startswith("."):
                    id_video = file.split("_")[0]
                    timestep = file.split("_")[1].split(".")[0]
                    if id_video in header_temp:
                        header_temp[id_video]["timesteps"].append(timestep)
                    else:
                        header_temp[id_video] = {"id_video": id_video, "category": category, "timesteps": [timestep]}
    header = [header_temp[video] for video in header_temp]
    with (open("dataset/headers/dataset.json", "w")) as dataset:
        json.dump(header, dataset, indent=4)

def find_transcript(transcripts, id_video, start, offset_start):
    """
    Search for a specific transcript in a list of transcripts.

    This function searches for a transcript in a list of transcripts based on the provided 'id_video', 'start',
    and 'offset_start' values. If a matching transcript is found, its transcript_audio text is returned. If no
    matching transcript is found, an error message is printed, and an empty string is returned.

    @param transcripts: List[Dict[str, Any]]
        A list of transcripts containing dictionaries with transcript information.

    @param id_video: str
        The ID of the video associated with the transcript.

    @param start: int
        The starting timestamp of the video segment associated with the transcript.

    @param offset_start: int
        The offset starting timestamp of the transcript.

    @return: str
        The transcript_audio text of the matching transcript if found, otherwise an empty string.
    """
    for transcript in transcripts:
        if transcript["id_video"]==id_video and\
            transcript["start"]==start and\
            transcript["offset_start"]==offset_start:
            return transcript["transcript_audio"]
    print("Unable to find transcript: ", id_video, start, offset_start)
    return ""

def process_comment_fragment(comment_element):
    """
    Process a comment element and extract the processed comment text and comment emotes.

    This function takes a comment element, represented as a list containing fragments of a comment, and processes
    the fragments to extract the comment text and comment emotes. Each fragment in the comment element can either
    contain plain text or an emoticon. The emoticons are stored in a list of dictionaries, where each dictionary
    represents an emoticon and its corresponding text.

    The processed comment text is created by replacing user mentions (words starting with '@') with the placeholder
    '@user' to maintain user privacy.

    @param comment_element: List[Dict[str, Any]]
        A list containing fragments of a comment.

    @return: Tuple[str, List[Dict[str, str]]]
        A tuple containing the processed comment text and a list of comment emotes.
        The processed comment text is a string, and the comment emotes are stored as a list of dictionaries,
        where each dictionary contains an emoticon ID as the key and its corresponding text as the value.
    """
    comment_text = ""
    comment_emotes = []
    for frag in comment_element:
        if frag["emoticon"]==None:
            comment_text+=frag["text"]
        else:
            comment_emotes.append({frag["emoticon"]["emoticon_id"]: frag["text"]})
    
    processed_comment_text = ""
    for word in comment_text.split():
        if word[0]=='@':
            processed_comment_text+="@user"+" "
        else: 
            processed_comment_text+=word+" "
    processed_comment_text=processed_comment_text.strip()
    return processed_comment_text, comment_emotes

def create_processed_dataset(root, target_file, comments_offset):
    """
    Process the dataset and create a processed dataset with specific information for each element.

    This function reads the dataset header from the 'dataset/headers/dataset.json' file and the transcript information
    from the 'dataset/transcripts.json' file. It then processes each video in the dataset, extracting chat context
    and responses from the comments and matching the corresponding transcript_audio text for each element. The chat
    context includes comments that occurred before a specific video timestamp, and the responses include comments
    that occurred after the same timestamp.

    The function creates a processed dataset by assembling DatasetElement instances, which are stored as dictionaries.
    The DatasetElement contains information about the video ID, start timestamp, offset start timestamp, chat context,
    responses, response index, transcript_audio text, context emotes, responses emotes, category, and path to the video.

    The processed dataset is saved to the 'target_file' in JSON format.

    @param root: str
        The root directory path.

    @param target_file: str
        The path to the target file where the processed dataset will be saved.

    @param comments_offset: int
        The offset value (in seconds) used to include comments that occurred before a specific video timestamp.

    """
    processed_dataset = []
    with open(root+"dataset/headers/dataset.json", "r") as header:
        header_json = json.load(header)
    with open("dataset/transcripts.json") as transcripts:
        transcripts_json = json.load(transcripts)
    for video in tqdm(header_json):
        with open(root+"dataset/comments/"+video["category"]+"/"+video["id_video"]+".json", "r") as comments:
            comments_json = json.load(comments)
        comments_list = [
            {
                "offset": comment["content_offset_seconds"], 
                "fragments": comment["message"]["fragments"]
            } for comment in comments_json["comments"]
        ]
        current_index=0
        max_index=len(comments_list)
        for timestep in sorted([int(ts) for ts in video["timesteps"]]):
            for i in range(20):
                element = DatasetElement(
                    id_video=video["id_video"], 
                    start=timestep,
                    offset_start=i*30, 
                    chat_context=[], 
                    responses=[], 
                    response_index=0,
                    transcript_audio=find_transcript(transcripts_json, video["id_video"], timestep, i*30), 
                    context_emotes=[],
                    responses_emotes=[],
                    category=video["category"], 
                    path_to_video=video["category"]+"/"+video["id_video"]+"_"+str(timestep)+".mp4"
                )                
                while current_index<max_index and comments_list[current_index]["offset"]<timestep+(i*30)+comments_offset:
                    current_index+=1
                while current_index<max_index and comments_list[current_index]["offset"]<timestep+(i*30)+20+comments_offset:
                    comment_text, comment_emotes=process_comment_fragment(comments_list[current_index]["fragments"])
                    if comment_text!="":
                        element.chat_context.append(comment_text)
                    if len(comment_emotes)>0:
                        element.context_emotes+=comment_emotes
                    current_index+=1
                while current_index<max_index and comments_list[current_index]["offset"]<timestep+(i*30)+30+comments_offset:
                    comment_text, comment_emotes=process_comment_fragment(comments_list[current_index]["fragments"])
                    if comment_text!="":
                        element.responses.append(comment_text)
                    if len(comment_emotes)>0:
                        element.responses_emotes+=comment_emotes
                    current_index+=1
                if len(element.chat_context) == 0 or len(element.responses) == 0:
                    continue
                element.chat_context = list(set(element.chat_context))
                element.responses = list(set(element.responses))
                element.response_index=int(compute_correlation(element.chat_context, element.responses))
                
                processed_dataset.append(asdict(element))

    print(len(processed_dataset))
    with open(target_file, "w") as output:
        json.dump(processed_dataset, output, indent=4)

def train_test_split(file: str, test_ratio):
    """
    Perform a train-test split on the given dataset JSON file.

    This function reads the dataset from the specified JSON file, shuffles the data randomly, and then splits it into
    two parts: a training set and a test set. The `test_ratio` parameter determines the proportion of data in the test set.
    The rest of the data is allocated to the training set.

    The shuffled dataset is split at the `split_index`, which is calculated as `int((1 - test_ratio) * dataset_length)`.
    The training set consists of the data from the beginning up to the `split_index`, and the test set contains the data
    from the `split_index` to the end.

    The function saves the resulting train and test datasets into separate JSON files named "dataset/train.json" and
    "dataset/test.json" respectively.

    @param file: str
        The path to the JSON file containing the dataset.

    @param test_ratio: float
        The proportion of data to be allocated to the test set. Should be a value between 0 and 1.

    """
    with open(file, "r") as f:
        json_dataset = json.load(f)
    dataset_length = len(json_dataset)
    random.shuffle(json_dataset)
    split_index = int((1-test_ratio)*dataset_length)
    train = json_dataset[:split_index]
    test = json_dataset[split_index:]
    with open("dataset/train.json", "w") as train_file:
        json.dump(train, train_file, indent=4)
    with open("dataset/test.json", "w") as test_file:
        json.dump(test, test_file, indent = 4)

def process_resnet_video(video_path, model):
    """
    Process a video using a pre-trained ResNet model and extract features from each clip of the video.

    The video is divided into 20 clips, each lasting for 4 seconds. For each clip, frames are extracted from the video
    between the start and end time of the clip. These frames are then transformed into tensors and passed through the
    pre-trained ResNet model to obtain feature representations for each frame. The feature representations of all frames
    in a clip are concatenated to represent the clip.

    The resulting feature representations for each clip are returned as a numpy array.

    Note: The model should be pre-trained on ImageNet and expect input frames of size 224x224.

    @param video_path: str
        The file path to the video to be processed.

    @param model: torch.nn.Module
        The pre-trained ResNet model to be used for feature extraction.

    @return: np.ndarray
        An array of shape (20, 120, num_features) containing feature representations for each clip of the video.
    """
    video = mp.VideoFileClip(video_path)
    features = []
    clip_features = []
    for clip_index in range(0, 20, 4):
        start_time = clip_index * 30
        end_time = (clip_index + 4) * 30


        frame_tensors = []
        for frame_time in range(start_time, end_time):
            frame = video.get_frame(frame_time)

            frame_tensor = transforms.ToTensor()(frame)
            frame_tensor = transforms.Resize(224)(frame_tensor)
            frame_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(frame_tensor.unsqueeze(0))

            frame_tensors.append(frame_tensor)
        batch_tensor = torch.stack(frame_tensors).squeeze().to("cuda")

        with torch.no_grad():
            features = model(batch_tensor)

        for i in range(4):
            clip_features.append(features.squeeze()[i*30:(i+1)*30].cpu().numpy())

    features = np.array(clip_features)

    return features

def resnet_category(root: str, category, model):
    """
    Process all videos of a specific category using a pre-trained ResNet model and extract features from each clip of the videos.

    For each video in the specified category, the function checks if the features for each clip have already been extracted
    and saved as numpy files. If not, the function calls the 'process_resnet_video' function to extract features for each clip
    of the video and saves them as numpy files in the 'dataset/features' directory.

    The function takes the root directory, the category name, and the pre-trained ResNet model as inputs.

    Note: The model should be pre-trained on ImageNet and expect input frames of size 224x224.

    @param root: str
        The root directory where the dataset is stored.

    @param category: str
        The name of the category for which videos' features are to be extracted.

    @param model: torch.nn.Module
        The pre-trained ResNet model to be used for feature extraction.

    @return: None
    """
    with open("dataset/headers/dataset.json", "r") as header:
        header_json = json.load(header)
    processed_json = [elem for elem in header_json if elem["category"]==category]

    for video in tqdm(processed_json):
        for timestep in sorted([int(ts) for ts in video["timesteps"]]):
            if not os.path.exists(root+"dataset/features/"+category+"/"+video["id_video"]+"_"+str(timestep)+"_0"+".npy"):
                video_features = process_resnet_video(root+"dataset/videos/"+category+"/"+video["id_video"]+"_"+str(timestep)+".mp4", model)
                for i in range(20):
                    filename = root+"dataset/features/"+category+"/"+video["id_video"]+"_"+str(timestep)+"_"+str(i*30)+".npy"
                    np.save(filename, video_features[i])

def resnet_everything():
    """
    Perform feature extraction using a pre-trained ResNet-50 model for every video in all specified categories.

    The function loads the pre-trained ResNet model, removes its final fully connected layer, moves the model to the GPU (if available),
    sets the model to evaluation mode, and then calls the 'resnet_category' function for each category.

    Note: The pre-trained ResNet model should be pre-trained on ImageNet and expect input frames of size 224x224.

    @return: None
    """
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to("cuda")
    model.eval()

    categories = ["18122", "21779", "26936", "27471", "32399", "33214", "509658", "509660", "509663", "515025", "516575"]

    for category in categories:
        print(category)
        resnet_category("/media/livechat/", category, model)

def candidate_comments(json_files, n_plausible=20, n_popular=20, n_random=50):
    """
    Generate candidate comments for each video in the given list of JSON files.

    The function creates a new JSON file containing the candidate comments for each video, including plausible, popular, and random comments.
    It uses TF-IDF similarity to find plausible comments based on the chat context of each video.

    @param json_files: List of JSON files containing video data and transcripts.
    @param n_plausible: Number of plausible comments to select as candidates. Default is 20.
    @param n_popular: Number of popular comments to select as candidates. Default is 20.
    @param n_random: Number of random comments to select as candidates. Default is 50.

    @return: Lists of plausible, popular, and random comments (for the last video in the input list).
    """
    #list of comments
    contexts = {}
    json_output = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            jason = json.load(f)
        for data in jason:
            new_entry = {}
            if data["id_video"] not in contexts.keys():
                contexts[data["id_video"]] = []
            new_entry["id_video"] = data["id_video"]
            new_entry["start"] = data["start"]
            new_entry["offset_start"] = data["offset_start"]
            new_entry["chat_context"] = data["chat_context"]
            new_entry["candidates"] = [[data["responses"][data["response_index"]]]]
            new_entry["correct_index"] = 0
            new_entry["responses"] = data["responses"]
            json_output.append(new_entry)
            contexts[data["id_video"]].append(data["chat_context"])
            contexts[data["id_video"]].append(data["responses"])
    for key in contexts.keys():
        contexts[key] = list(chain(*contexts[key]))
    for entry in tqdm(json_output, desc="searching candidates"):
        all_comments = contexts[entry["id_video"]]
        # plausible comments
        tvec = TfidfVectorizer()
        chat_context = " ".join(entry["chat_context"])
        tall_comments = tvec.fit_transform(all_comments)
        tcontext = tvec.transform([chat_context])
        cosine_similarity = np.ravel((tcontext * tall_comments.T).todense())
        sorted_similarity = np.argpartition(cosine_similarity, -1)[::-1]
        plausible_comments = np.array(all_comments)[sorted_similarity].tolist()
        if entry["id_video"]=="1773040627":# and entry["start"]==143400 and entry["offset_start"]==570:
            print(all_comments)
        plausible_comments = list(filter(lambda x:x not in entry["responses"], plausible_comments))[:n_plausible]
        entry["candidates"].append(plausible_comments)
        if len(plausible_comments)!=n_plausible:    
            print("Plausible ", len(plausible_comments), len(all_comments), entry["id_video"], entry["start"], entry["offset_start"])
        # popular comments
        counter = Counter(all_comments)
        popular_comments = [pop[0] for pop in counter.most_common(len(counter))]
        popular_comments = list(filter(lambda x:x not in entry["responses"], popular_comments))
        popular_comments = popular_comments[:n_popular]
        entry["candidates"].append(popular_comments)
        if len(popular_comments)!=n_popular:
            print("Popular ", len(popular_comments), len(all_comments), entry["id_video"], entry["start"], entry["offset_start"])
        # random comments
        random_comments = random.choices(all_comments, k=n_random*2)
        random_comments = list(filter(lambda x:x not in entry["responses"], random_comments))
        random_comments = random_comments[:n_random]
        entry["candidates"].append(random_comments)
        if len(random_comments)!=n_random:
            print("Random ", len(random_comments), len(all_comments), entry["id_video"], entry["start"], entry["offset_start"])
        # flatten the responses entry
        entry["candidates"] = list(chain(*entry["candidates"]))
        if len(entry["candidates"])!=n_plausible+n_popular+n_random+1:
            print(len(entry["candidates"]))
    # write the json
    print("---writing---")
    with open("dataset/candidate_comments.json", "w") as json_output_file:
        json.dump(json_output, json_output_file, indent=4)
    print("---done---")
    return plausible_comments, popular_comments, random_comments

def sanity_check(root: str):
    """
    Perform a sanity check to verify if the data in the train, test, and transcripts JSON files is consistent with the dataset JSON file.

    The function compares the number of data points in the train and test sets with the total number of timesteps in the dataset and the number of transcripts available.

    @param root: The root directory path where the train, test, dataset, and transcripts JSON files are located.

    @return: None
    """
    with open(root+"train.json") as train:
        train_json = json.load(train)
    with open(root+"test.json") as test:
        test_json = json.load(test)
    with open(root+"headers/dataset.json") as dataset:
        dataset_json = json.load(dataset)
        dataset_length = np.sum([len(item["timesteps"]) for item in dataset_json])
    with open("dataset/transcripts.json") as transcripts:
        transcripts_json = json.load(transcripts)
    
    print(len(train_json)+len(test_json), dataset_length*20, len(transcripts_json))

if __name__=='__main__':
    # create_dataset_header("/media/livechat/")
    # create_processed_dataset("/media/livechat/", "dataset/processed_split_emotes.json", 0)

    # candidate_comments(["dataset/temp/procesed_0.json"])
    # resnet_everything()
    # train_test_split("dataset/processed_split_emotes.json", 0.1)
    # candidate_comments(["dataset/processed_split_emotes.json"], n_random=39)

    with open("/media/livechat/dataset/test_emotes.json") as test:
        test_json = json.load(test)
    with open("dataset/candidate_comments.json") as candidates:
        candidates_json = json.load(candidates)
    
    idx_to_remove = []
    for idx, item in tqdm(enumerate(test_json)):
        for candidate in candidates_json:
            if item["id_video"]==candidate["id_video"] and item["start"]==candidate["start"] and item["offset_start"]==candidate["offset_start"]:
                if len(candidate["candidates"])!=80:
                    idx_to_remove.append(idx)
                    print(len(candidate["candidates"]))
                else:
                    item["candidates"] = candidate["candidates"]
    for idx in sorted(idx_to_remove, reverse=True):
        test_json.pop(idx)
    
    with open("dataset/test_emotes_candidates.json", "w") as out:
        json.dump(test_json, out)
    
    
