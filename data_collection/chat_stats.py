import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def hist_comments(file: str):
    """
    Generate a histogram of comment timestamps in a JSON comment file.

    This function reads a JSON file containing comments and extracts the comment timestamps.
    It generates a histogram of the comment timestamps in intervals of 10 minutes (600 seconds) up to the maximum video length.

    @param file: str
        The path to the JSON file containing the comments data.

    @return: tuple
        A tuple containing two arrays:
        - The first array represents the bin edges of the histogram.
        - The second array represents the counts of comments within each bin.
    """
    with open(file, 'r') as f:
        js = json.load(f)
        try:
            comments_timestamp = [comment['content_offset_seconds'] for comment in js['comments']]
        except KeyError:
            comments_timestamp = [comment['commented_at'] for comment in js['comments']]
        max_timestamp = int(js['video']['length'])
        return np.histogram(comments_timestamp, range(0, max_timestamp, 60*10))

def hist_nb_words(file: str):
    """
    Generate a histogram of the number of words in comments from a JSON comment file.

    This function reads a JSON file containing comments and calculates the number of words in each comment.
    It generates a histogram of the number of words per comment, counting the occurrences of each word count in intervals up to 50 words.

    @param file: str
        The path to the JSON file containing the comments data.

    @return: list
        A list representing the histogram of word counts in the comments.
        The index 'i' of the list represents the number of comments with 'i' words.
        The last element of the list (index -1) represents the number of comments with more than 50 words.
    """
    hist = [0 for _ in range(50)]
    counter = 0
    with open(file, 'r') as f:
        js = json.load(f)
        for comment in js['comments']:
            message = comment['message']["body"]
            nb_words = len(message.split())
            counter+=nb_words
            if nb_words<50:
                hist[nb_words]+=1
            else:
                hist[-1]+=1
    return hist

def get_nb_videos_from_headers(downloaded:bool=False, root:str="/media/livechat"):
    """
    Get the number of videos from the headers of the downloaded or all categories.

    This function counts the number of videos present in the headers of either all categories or only the downloaded ones.
    It reads the JSON files containing category information and checks the 'downloaded' field of each video.

    @param downloaded: bool, optional (default=False)
        A flag indicating whether to count only the downloaded videos (True) or all videos (False).

    @param root: str, optional (default="/media/livechat")
        The root directory where the 'dataset' folder is located.

    @return: int
        The number of videos found in the headers of the selected categories.
    """
    nb_videos = 0
    for file in os.listdir(os.fsencode(os.path.join(root, "dataset/headers/categories"))):
        filename = os.fsdecode(file)
        category_name = os.path.join(root, "dataset/headers/categories/"+filename)
        if '._' in category_name:
            continue
        with open(category_name, "r") as category:
            category_json = json.load(category)
            for video in category_json:
                if not "downloaded" in video:
                    video["downloaded"] = "false"
                if not downloaded or video["downloaded"]=="true":
                    nb_videos+=1
        with open(os.path.join(root, "dataset/headers/categories/"+filename), "w") as category:
            json.dump(category_json, category, indent=4)
    return nb_videos

def get_nb_videos_from_files(root="/media/livechat"):
    """
    Get the number of video files in the 'dataset/videos' directory.

    This function counts the number of video files present in the 'dataset/videos' directory and its subdirectories.
    It iterates through each category folder and counts the video files with the '.mp4' extension.

    @param root: str, optional (default="/media/livechat")
        The root directory where the 'dataset' folder is located.

    @return: int
        The total number of video files found in the 'dataset/videos' directory.
    """
    video_folder = os.path.join(root, "dataset/videos")
    categories = os.listdir(video_folder)
    video_file_count = 0
    for category in categories:
        folder_path = os.path.join(video_folder, category)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                if file.startswith("._"):
                    continue
                if file.endswith(".mp4"):
                    video_file_count += 1
    return video_file_count

def get_differences_header_videos():
    ...

def get_total_duration(root:str="/media/livechat"):
    """
    Get the total duration of all video files in the 'dataset/videos' directory.

    This function calculates the total duration of all video files present in the 'dataset/videos' directory and its subdirectories.
    It uses the `get_nb_videos_from_files` function to count the number of video files and assumes each video has a duration of 10 seconds.

    @param root: str, optional (default="/media/livechat")
        The root directory where the 'dataset' folder is located.

    @return: int
        The total duration (in seconds) of all video files found in the 'dataset/videos' directory.
    """
    nb_videos = get_nb_videos_from_files(root)
    return nb_videos * 10

def get_nb_categories(root:str="/media/livechat"):
    """
    Get the number of categories from the 'dataset/headers/categories.json' file.

    This function reads the 'categories.json' file, which contains information about different video categories.
    It then returns the total number of categories listed in the file.

    @param root: str, optional (default="/media/livechat")
        The root directory where the 'dataset' folder is located.

    @return: int
        The total number of video categories listed in the 'dataset/headers/categories.json' file.
    """
    with open(os.path.join(root, "dataset/headers/categories.json"), "r") as file:
        categories = json.load(file)
    return len(categories)

def get_nb_streamer(root:str="/media/livechat"):
    """
    Get the number of unique streamers from the video category headers.

    This function scans the headers of all video categories in the 'dataset/headers/categories' folder.
    It counts the number of unique streamers across all categories by examining the 'user_id' field in each video header.

    @param root: str, optional (default="/media/livechat")
        The root directory where the 'dataset' folder is located.

    @return: int
        The total number of unique streamers found in the video category headers.
    """
    unique_streamers = set()
    categories_folder = os.path.join(root, "dataset/headers/categories")
    categories = os.listdir(categories_folder)
    for category in categories:
        if category.startswith("._"):
            continue
        if category.endswith(".json"):
            filename = os.path.join(categories_folder, category)
            file = open(filename, "r")
            category_json = json.load(file)
            for video in category_json:
                unique_streamers.add(video["user_id"])
            file.close()
    return len(unique_streamers)

def get_nb_comments(total_number:bool=False, root:str="/media/livechat"):
    """
    Get the number of comments from the video comment files.

    This function counts the number of comments present in the video comment files stored in the 'dataset/comments' folder.
    It iterates through each category folder, reads the comment files, and calculates the total number of comments.

    By default, the function calculates the total number of comments from all comment files.
    If 'total_number' is set to False, it calculates the total number of comments from the top 3 video moments with the highest comment count.

    @param total_number: bool, optional (default=False)
        A flag indicating whether to calculate the total number of comments (True) or the number of comments from the top 3 video moments (False).

    @param root: str, optional (default="/media/livechat")
        The root directory where the 'dataset' folder is located.

    @return: int
        The total number of comments found in the video comment files.
        If 'total_number' is False, it returns the total number of comments from the top 3 video moments with the highest comment count.
    """
    comments_folder = os.path.join(root, 'dataset/comments')
    categories = os.listdir(comments_folder)
    nb_comments = 0
    with tqdm(total=get_nb_videos_from_headers(True)) as pbar:
        for category in categories:
            folder_path = os.path.join(comments_folder, category)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                for file in files:
                    if file.startswith("._"):
                        continue
                    if file.endswith(".json"):
                        filename = os.path.join(folder_path, file)
                        if total_number:
                            comments_file = open(filename, "r")
                            comments_json = json.load(comments_file)
                            nb_comments+=len(comments_json["comments"])
                            comments_file.close()
                        else:
                            hist, _ = hist_comments(filename)
                            top_3_moments = np.sort(hist)[-3:]
                            for video in top_3_moments:
                                nb_comments+=video
                        pbar.update(1)
    return nb_comments

def get_video_repartition():
    """
    Get the repartition of videos across categories.

    This function counts the number of videos present in each category within the 'dataset/videos' folder.
    It returns a list of categories and a corresponding list of video counts, representing the repartition of videos across different categories.

    @return: tuple
        A tuple containing two lists:
        - The first list represents the categories found in the 'dataset/videos' folder.
        - The second list represents the number of videos in each category, corresponding to the categories in the first list.
    """
    video_folder = "../dataset/videos"
    categories = os.listdir(video_folder)
    try:
        categories.remove(".DS_Store")
    except:
        pass
    nb_videos = [0 for _ in range(get_nb_categories())]
    for index, category in enumerate(categories):
        folder_path = os.path.join(video_folder, category)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            for file in files:
                if file.endswith(".mp4"):
                    nb_videos[index]+=1
    return categories, nb_videos

def get_comment_length(root:str="/media/livechat"):
    """
    Get the histogram of comment lengths (in words) from the video comment files.

    This function calculates the histogram of comment lengths (in words) from the video comment files stored in the 'dataset/comments' folder.
    It iterates through each category folder, reads the comment files, and calculates the histogram of comment lengths.

    The histogram represents the distribution of comment lengths in intervals from 0 to 49 words.

    @param root: str, optional (default="/media/livechat")
        The root directory where the 'dataset' folder is located.

    @return: list
        A list representing the histogram of comment lengths (in words).
        The index 'i' of the list represents the number of comments with 'i' words.
        The last element of the list (index 49) represents the number of comments with 49 or more words.
    """
    comments_folder = os.path.join(root, "dataset/comments")
    categories = os.listdir(comments_folder)
    hist = [0 for _ in range(50)]
    with tqdm(total=get_nb_videos_from_headers(True, root)) as pbar:
        for category in categories:
            if category.startswith("._"):
                continue
            folder_path = os.path.join(comments_folder, category)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                for file in files:
                    if file.startswith("._"):
                        continue
                    if file.endswith(".json"):
                        filename = os.path.join(folder_path, file)
                        new_hist = hist_nb_words(filename)
                        hist = [hist[i] + new_hist[i] for i in range(50)]
                        pbar.update(1)
    return hist

def compute_correlation(transcript, context):
    """
    Compute the cosine similarity between a transcript and a context.

    This function calculates the cosine similarity between a given transcript and a context.
    It first vectorizes the transcript and the context using TF-IDF (Term Frequency-Inverse Document Frequency).
    Then, it calculates the cosine similarity between the transcript and each sentence in the context.

    @param transcript: str
        The transcript text for which the similarity is to be computed.

    @param context: List[str]
        A list of sentences representing the context.

    @return: np.ndarray
        A NumPy array containing the cosine similarity scores between the transcript and each sentence in the context.
    """
    vectorizer = TfidfVectorizer()
    context = "".join(sent+" " for sent in context)
    try:
        vectors = vectorizer.fit_transform([transcript] + [context])
        similarities = cosine_similarity(vectors[0], vectors[1:])
    except:
        print(transcript, context)
        return(0)
    return similarities

def correlation_over_offset():
    """
    Calculate the average cosine similarity between transcripts and contexts over different offsets.

    This function calculates the average cosine similarity between transcripts and their corresponding contexts
    for different offsets. It iterates through a range of offsets (from 0 to 59) and processes JSON files containing
    information about transcripts and contexts. The processed JSON files are assumed to be located in the 'dataset/temp' folder.

    @return: List[float]
        A list of average cosine similarity values, where each value represents the average correlation for a specific offset.
    """
    correlations = []
    for offset in range(60):
        avg_correlation = 0
        with open(f"dataset/temp/procesed_{offset}.json") as processed:
            processed_json = json.load(processed)
        for clip in tqdm(processed_json):
            avg_correlation+=compute_correlation(clip["transcript_audio"], clip["chat_context"]+clip["responses"])[0][0]
        avg_correlation/=len(processed_json)
        correlations.append(avg_correlation)
        print(avg_correlation)
    return correlations

if __name__=="__main__":
    root="/media/livechat"
    print("Number of categories: ", get_nb_categories(root))
    print("Number of unique streamers", get_nb_streamer(root))
    print("Number of extracts downloaded: ", get_nb_videos_from_headers(True, root))
    print("Number of extracts downloaded: ", get_nb_videos_from_files(root))
    print("Total duration in hours: ", get_total_duration()/60)

    # print("Number of comments corresponding to the video timesteps: ", get_nb_comments(False, root))
    # print("Total number of comments: ", get_nb_comments(True, root))
    print("Comments length histogram", get_comment_length(root))

    # print(get_video_repartition())
    # correlation_over_offset()
