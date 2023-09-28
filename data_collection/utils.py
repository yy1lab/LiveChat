import os, json, shutil, time, requests, enchant, cv2
import moviepy.editor as mp

from better_profanity import profanity
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

from data_collection.CREDENTIALS import CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN

DCT = enchant.Dict('en-US')

HEADERS = {
        'Client-ID': CLIENT_ID,
        'Authorization': f'Bearer {ACCESS_TOKEN}',
    }

def get_new_token():
    """
    Generate an API token

    @return: json object containing the token
    """
    url='https://id.twitch.tv/oauth2/token'
    data=f'client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}&grant_type=client_credentials'
    return requests.post(url, headers='Content-Type: application/x-www-form-urlencoded', data=data).json()

def get_top_games():
    """
    Retreive the top 50 games from Twitch

    @return: json object containing the games
    """
    url='https://api.twitch.tv/helix/games/top?first=50'
    return requests.get(url, headers=HEADERS).json()

def get_top_videos(game_id: str, nb_videos: int, period: str="all"):
    """
    Retreive the most viewed videos in a particular category

    @param game_id: str -> Unique ID of the game
    @param nb_videos: int -> the number of videos to retreive
    @param period: str -> "all, "week" or "month" depending on the period to look for

    @return: json object containing the video urls, names and info
    """ 
    url=f'https://api.twitch.tv/helix/videos?game_id={game_id}&type=archive&language=en&sort=views&period={period}&first={str(nb_videos)}'
    return requests.get(url, headers=HEADERS).json()

def get_top_videos_pagination(game_id: str, nb_videos: int, pagination: str, period: str="all"):
    """
    Retreive the most viewed videos in a particular category after a pagination

    @param game_id: str -> Unique ID of the game
    @param nb_videos: int -> the number of videos to retreive
    @param pagination: str -> the pagination identifier after which to look for
    @param period: str -> "all, "week" or "month" depending on the period to look for

    @return: json object containing the video urls, names and info
    """
    url=f'https://api.twitch.tv/helix/videos?game_id={game_id}&type=archive&language=en&sort=views&period={period}&first={str(nb_videos)}&after={pagination}'
    return requests.get(url, headers=HEADERS).json()

def get_top_streamers(game_id: str, nb_streamers: int):
    """
    Retreive the most popular streamers currently live in a particular category

    @param game_id: str -> Unique ID of the game
    @param nb_streamers: int -> the number of streamers to retreive

    @return: json object containing the streamer information
    """
    url=f'https://api.twitch.tv/helix/streams?game_id={game_id}&language=en&sort=followers&first={str(nb_streamers)}'
    return requests.get(url, headers=HEADERS).json()

def dl_comments(video_url: str, output: str):
    """
    Download the comments of a Twitch video

    @param video_url: str -> the url of the desired video comments
    @param output: str -> the output file to write the comments into
    """
    os.system('./twitch_downloader/TwitchDownloaderCLI chatdownload -u {url} -o dataset/comments/{output}'.format(url=video_url, output=output))

def dl_video(video_url: str, output: str, quality: str, start: int, end: int):
    """
    Download a Twitch video

    @param video_url: str -> the url of the desired video
    @param output: str -> the output file to write the video into
    @param quality: str -> the desired quality of the video
    @param start: int -> the starting timestamp of the video
    @param end: int -> the ending timestamp of the video
    """
    os.system('./twitch_downloader/TwitchDownloaderCLI videodownload -u {url} -o dataset/videos/{output} -q {quality} -b {start} -e {end} --ffmpeg-path twitch_downloader/ffmpeg'.format(url=video_url, output=output, quality=quality, start=start, end=end))

def dl_clip(video_url: str, output: str, quality: str):
    """
    Download a Twitch clip

    @param video_url: str -> the url of the desired clip
    @param output: str -> the output file to write the video into
    @param quality: str -> the desired quality of the video
    """
    os.system('./twitch_downloader/TwitchDownloaderCLI clipdownload -u {url} -o dataset/videos/{output} -q {quality}'.format(url=video_url, output=output, quality=quality))

def get_video_info_from_comment_file(comments_file: str):
    """
    Extract video information from a JSON comment file.

    This function reads a JSON file containing comments and retrieves information related to a specific video.
    It returns a JSON-formatted string representing the video information, indented for better readability.

    @param comments_file: str
        The path to the JSON file containing the comments data.

    @return: str
        A JSON-formatted string representing the video information from the comments, indented for better readability.
    """
    with open(comments_file, 'r') as comments:
        comments_json = json.load(comments)
        return json.dumps(comments_json['video'], indent=4)

def get_streamer_info_from_comment_file(comments_file: str):
    """
    Extract streamer information from a JSON comment file.

    This function reads a JSON file containing comments and retrieves information related to the streamer.
    It returns a JSON-formatted string representing the streamer information, indented for better readability.

    @param comments_file: str
        The path to the JSON file containing the comments data.

    @return: str
        A JSON-formatted string representing the streamer information from the comments, indented for better readability.
    """
    with open(comments_file, 'r') as comments:
        comments_json = json.load(comments)
        return json.dumps(comments_json['streamer'], indent=4)

def get_comment_number(comments_file: str):
    """
    Get the number of comments from a JSON comment file.

    This function reads a JSON file containing comments and determines the total number of comments present.

    @param comments_file: str
        The path to the JSON file containing the comments data.

    @return: int
        The total number of comments present in the JSON file.
    """
    with open(comments_file, 'r') as comments:
        comments_json = json.load(comments)
        return len(comments_json['comments'])

def get_comment(comments_file: str, comment_id: int):
    """
    Get a specific comment from a JSON comment file.

    This function reads a JSON file containing comments and retrieves a specific comment identified by the given comment_id.
    It returns the requested comment as a JSON-formatted string, indented for better readability.

    @param comments_file: str
        The path to the JSON file containing the comments data.

    @param comment_id: int
        The index of the desired comment in the 'comments' list.

    @return: str
        A JSON-formatted string representing the requested comment, indented for better readability.
    """
    with open(comments_file, 'r') as comments:
        comments_json = json.load(comments)
        return json.dumps(comments_json['comments'][comment_id], indent=4)

def prettify_json(comments_file: str, output_file: str, keep_emotes:bool=True):
    """
    Prettify a JSON comment file while optionally keeping or removing emotes from comments.

    This function reads a JSON file containing comments and creates a new JSON file with prettified data.
    It provides an option to keep or remove emotes from the comments based on the 'keep_emotes' parameter.

    @param comments_file: str
        The path to the JSON file containing the comments data.

    @param output_file: str
        The path to the new JSON file where the prettified data will be written.

    @param keep_emotes: bool, optional (default=True)
        A flag to determine whether to keep emotes in comments or remove them.
        If True, emotes will be preserved in the comment text. If False, emotes will be removed.

    @return: None
        The function does not return anything. It writes the prettified JSON data to the output_file.
    """
    with open(comments_file, 'r') as comments:
        comments_json = json.load(comments)
        output_json = {}
        output_json['streamer'] = comments_json['streamer']
        output_json['video'] = comments_json['video']
        
        output_json['comments'] = []
        for comment in comments_json['comments']:
            new_message=''
            for fragment in comment['message']['fragments']:
                if keep_emotes or not fragment['emoticon']:
                    new_message+=fragment['text']
            new_message=new_message.rstrip().lstrip()
            if new_message!='':
                comment_info = {
                    'commented_at': comment['content_offset_seconds'],
                    'message': new_message
                }
                output_json['comments'].append(comment_info)
    with open(output_file, "w") as output:
        json.dump(output_json, output, indent=4)

def filter_profanities(comments_file: str, output_file: str):
    """
    Filter out comments containing profanities from a JSON comment file.

    This function reads a JSON file containing comments and filters out any comments that contain profane language.
    The filtered comments are written to a new JSON file specified by the 'output_file' parameter.

    @param comments_file: str
        The path to the JSON file containing the comments data.

    @param output_file: str
        The path to the new JSON file where the filtered comments will be written.

    @return: None
        The function does not return anything. It writes the filtered JSON data to the output_file.
    """
    with open(comments_file, 'r') as comments:
        comments_json = json.load(comments)
        output_json = {}
        output_json['streamer'] = comments_json['streamer']
        output_json['video'] = comments_json['video']
        
        output_json['comments'] = []
        for comment in tqdm(comments_json['comments']):
            if not profanity.contains_profanity(comment['message']):
                output_json['comments'].append(comment)
    with open(output_file, "w") as output:
        json.dump(output_json, output, indent=4)

def is_english(sentence: str):
    """
    Check if a sentence is predominantly in English.

    This function takes a sentence as input and checks if the majority of the words in the sentence are in the English language.
    It uses a simple heuristic by comparing the number of English words recognized by the DCT (Dictionary Check Tool) against the total number of words in the sentence.
    If more than 30% of the words are recognized as English, the function considers the sentence to be predominantly in English.

    @param sentence: str
        The sentence to be evaluated for language detection.

    @return: bool
        True if the majority of the words in the sentence are recognized as English, False otherwise.
    """
    sentence_list = sentence.split()
    if not len(sentence_list): return False
    counter=0
    for word in sentence_list:
        counter+=DCT.check(word)
    return counter>len(sentence_list)*0.3

def filter_english_words(comments_file: str, output_file: str):
    """
    Filter out non-English comments from a JSON comment file.

    This function reads a JSON file containing comments and filters out any comments that are predominantly in a language other than English.
    The filtering is done using the `is_english` function, which checks if the majority of the words in a sentence are recognized as English.

    @param comments_file: str
        The path to the JSON file containing the comments data.

    @param output_file: str
        The path to the new JSON file where the filtered English comments will be written.

    @return: None
        The function does not return anything. It writes the filtered JSON data to the output_file.
    """
    with open(comments_file, 'r') as comments:
        comments_json = json.load(comments)
        output_json = {}
        output_json['streamer'] = comments_json['streamer']
        output_json['video'] = comments_json['video']
        
        output_json['comments'] = []
        for comment in tqdm(comments_json['comments']):
            if is_english(comment['message']):
                output_json['comments'].append(comment)
    with open(output_file, "w") as output:
        json.dump(output_json, output, indent=4)

def cut_video(input: str, output: str, start: int, end: int):
    """
    Cut a video clip from the specified start time to the end time.

    This function uses FFmpeg to extract a portion of a video file, starting from the 'start' time (in seconds) to the 'end' time (in seconds).
    The extracted video clip is saved to the 'output' file.

    Note: To use this function, FFmpeg must be installed on your system and accessible from the command line.

    @param input: str
        The path to the input video file.

    @param output: str
        The path to save the extracted video clip.

    @param start: int
        The start time (in seconds) from where the video clip should begin.

    @param end: int
        The end time (in seconds) at which the video clip should end.

    @return: None
        The function does not return anything. It extracts and saves the video clip to the 'output' file.
    """
    ffmpeg_extract_subclip(input, start, end, output)

def cut_comments(input: str, output: str, start: int, end: int):
    """
    Cut comments from a JSON comment file within the specified time range.

    This function reads a JSON file containing comments and filters out comments that fall within the specified time range.
    It creates a new JSON file containing the filtered comments, along with the adjusted 'commented_at' timestamps relative to the start time.

    @param input: str
        The path to the input JSON file containing the comments data.

    @param output: str
        The path to the new JSON file where the filtered comments will be written.

    @param start: int
        The start time (in seconds) of the time range for extracting comments.

    @param end: int
        The end time (in seconds) of the time range for extracting comments.

    @return: None
        The function does not return anything. It writes the filtered JSON data to the 'output' file.
    """
    with open(input, 'r') as comments:
        comments_json = json.load(comments)
        output_json = {}
        output_json['streamer'] = comments_json['streamer']
        output_json['video'] = comments_json['video']
        
        output_json['comments'] = []
        for comment in comments_json['comments']:
            if start < comment['commented_at'] < end:
                output_json['comments'].append({"commented_at": comment['commented_at'], 'message': comment['message']})
        
        start_time = output_json['comments'][0]['commented_at']
        for comment in output_json['comments']:
            comment['commented_at']-=start_time
    with open(output, "w") as output:
        json.dump(output_json, output, indent=4)

def is_video_present(video, video_list):
    """
    Check if a video is present in the list of videos.

    This function takes a video and a list of videos as input and checks if the video is present in the list.
    It compares the 'id' attribute of the input video with the 'id' attribute of each video in the list.
    If a video with the same 'id' is found in the list, the function returns True; otherwise, it returns False.

    @param video: dict
        A dictionary representing the video to check. It should have an 'id' attribute.

    @param video_list: list of dict
        A list of dictionaries representing videos, each with an 'id' attribute.

    @return: bool
        True if the video is present in the video_list, False otherwise.
    """
    for vid in video_list:
        if vid["id"]==video["id"]:
            return True
    return False

def reduce_quality(input_source: str):
    """
    Reduce the quality of a video file to a lower resolution and frame rate.

    This function takes an input video file and reduces its quality to a specified resolution and frame rate.
    The reduced video is saved to the same location as the original input video file.

    @param input_source: str
        The path to the input video file.

    @return: None
        The function does not return anything. It reduces the quality of the video and saves it to the same location.
    """
    input_video = mp.VideoFileClip(input_source)
    fps = 5
    target_resolution = (1280, 720)
    output_video_filename = "temp/reduced_2.mp4"
    input_video = input_video.set_fps(fps)
    input_video = input_video.resize(target_resolution)
    input_video.write_videofile(output_video_filename, codec='libx264')
    input_video.close()
    time.sleep(2)
    shutil.move(output_video_filename, input_source)

def reduce_category_quality(category: str):
    """
    Reduce the quality of videos in a specific category to a lower resolution and frame rate.

    This function takes a category name as input and processes all video files within that category.
    Videos with a frame count exceeding 20 minutes (based on a frame rate of 30 fps) or a width other than 1280 pixels
    will have their quality reduced using the 'reduce_quality' function.

    @param category: str
        The name of the category containing the videos to be processed.

    @return: None
        The function does not return anything. It reduces the quality of specific videos within the category.
    """
    for file in tqdm(os.listdir(os.fsencode("dataset/videos/"+category))):
        filename = os.fsdecode(file)
        if cv2.VideoCapture("dataset/videos/"+category+"/"+filename).get(cv2.CAP_PROP_FRAME_COUNT)/600 > 20 or cv2.VideoCapture("dataset/videos/"+category+"/"+filename).get(cv2.CAP_PROP_FRAME_WIDTH) != 1280.0:
            reduce_quality("dataset/videos/"+category+"/"+filename)

def reduce_all_quality():
    """
    Reduce the quality of videos in all categories to lower resolutions and frame rates.

    This function reads the category information from the 'dataset/headers/categories.json' file.
    It iterates through each category and reduces the quality of videos within each category using the 'reduce_category_quality' function.

    @return: None
        The function does not return anything. It reduces the quality of videos in all categories.
    """
    with open('dataset/headers/categories.json', 'r') as file:
        categories = json.load(file)
    for category in categories:
        print("Downgrading category", category["id"])
        reduce_category_quality(category["id"])

if __name__=='__main__':
    while True:
        reduce_all_quality()  