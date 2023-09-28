import json
import os
import logging
import numpy as np

import data_collection.utils as utils
import data_collection.chat_stats as chat_stats

def get_top_videos_category(
        category: str, 
        nb_videos: int=30,
        period: str="all",
        root="/media/livehat/"
    ):
    """
    Get the top videos from a specific category.

    This function retrieves the top videos from a given category. It makes use of pagination to ensure the specified number
    of videos (nb_videos) is obtained. The videos are fetched based on the provided category and period.

    @param category: str
        The category from which to fetch the top videos.

    @param nb_videos: int, optional (default=30)
        The number of top videos to retrieve.

    @param period: str, optional (default="all")
        The period for which to fetch the top videos. Possible values: "all", "day", "week", "month".

    @param root: str, optional (default="/media/livehat/")
        The root directory for the dataset.

    @return: List[Dict[str, Any]]
        A list of dictionaries, where each dictionary contains information about a top video from the specified category.
        The dictionaries include fields such as "user_login", "downloaded", and other relevant video information.
    """
    top_videos_1=utils.get_top_videos(category, 100, period=period)

    nb_videos_total = 0
    pagination_index = 1
    top_videos=[]
    user_logins=[]
    for video in top_videos_1["data"]:
        if video["user_login"] not in user_logins:
            video["downloaded"]="false"
            top_videos.append(video)
            user_logins.append(video["user_login"])
            nb_videos_total+=1
        if nb_videos_total==nb_videos:
            return top_videos
    
    while pagination_index<=5 and nb_videos_total<nb_videos:
        pagination=top_videos_1["pagination"]["cursor"]
        pagination_index+=1
        top_videos_1=utils.get_top_videos_pagination(category, 100, pagination, period=period)

        for video in top_videos_1["data"]:
            if video["user_login"] not in user_logins:
                video["downloaded"]="false"
                top_videos.append(video)
                user_logins.append(video["user_login"])
                nb_videos_total+=1        
            if nb_videos_total==nb_videos:
                return top_videos
    return top_videos

def get_videos_from_all_categories(period: str="all", root="/media/livehat/"):
    """
    Get and update the top videos for all categories.

    This function fetches the top videos for all categories and updates the corresponding JSON files
    with the new video information. It makes use of the 'get_top_videos_category' function to fetch the
    top videos for each category.

    The JSON files for each category are assumed to be located in the 'dataset/headers/categories' folder.

    @param period: str, optional (default="all")
        The period for which to fetch the top videos. Possible values: "all", "day", "week", "month".

    @param root: str, optional (default="/media/livehat/")
        The root directory for the dataset.

    """
    with open(root+"dataset/headers/categories.json", "r") as file:
        categories = json.load(file)
        for category in categories:
            category_id = category["id"]
            print("Category: ", category_id)
            top_videos = get_top_videos_category(category_id, period=period, root=root)
            old_file=open(root+"dataset/headers/categories/"+category_id+".json", "r")
            old_json = json.load(old_file)
            for video in top_videos:
                if not utils.is_video_present(video, old_json):
                    old_json.append(video)
            old_file.close()
            json.dump(old_json, fp=open(root+"dataset/headers/categories/"+category_id+".json", "w"), indent=4)

def dl_category(category: str, root="/media/livehat/"):
    """
    Download comments and videos for a given category.

    This function downloads comments and videos for a specific category. It reads the category JSON file,
    which contains information about videos, their URLs, and their download status. It then iterates through
    the videos in the category, checking if they have already been downloaded. If not, it downloads the comments
    for the video and extracts certain moments from the video based on histogram analysis of the comments.

    The category JSON file is assumed to be located in the 'dataset/headers/categories' folder, and the downloaded
    comments and videos will be stored in the 'dataset/comments' and 'dataset/videos' folders, respectively.

    @param category: str
        The filename of the category JSON file (excluding the folder path).

    @param root: str, optional (default="/media/livehat/")
        The root directory for the dataset.

    """
    logging.info("Reading category "+category)
    with open(root+"dataset/headers/categories/"+category, "r") as cat:
        category_json = json.load(cat)
        useless_indexes = []
        for video_index, video in enumerate(category_json):
            if not "downloaded" in video:
                video["downloaded"]="false"
            if video["downloaded"]=="true":
                logging.info("Video "+str(video["id"])+" already downloaded")
                continue
            logging.info("Download of the comment for video "+str(video["id"]))
            utils.dl_comments(video["url"], category.rstrip(".json")+"/"+video["id"]+".json")
            try:
                with open(root+"dataset/comments/"+category.rstrip(".json")+"/"+video["id"]+".json") as comments:
                    comments_json = json.load(comments)
                with open(root+"dataset/comments/"+category.rstrip(".json")+"/"+video["id"]+".json", "w") as file:
                    json.dump(comments_json, file, indent=4)
            except FileNotFoundError:
                logging.info(str(video["id"])+": Download failed...")
                useless_indexes.append(video_index)
            else:
                hist, bin_edges = chat_stats.hist_comments(root+"dataset/comments/"+category.rstrip(".json")+"/"+video["id"]+".json")
                if len(hist)>12:
                    indexes = np.argpartition(hist, -3)[-3:]
                    for index in indexes:
                        logging.info("Download of the video "+str(video["id"])+", starting at timestep "+str(bin_edges[index]))
                        utils.dl_video(video["url"],category.rstrip(".json")+"/"+video["id"]+"_"+str(bin_edges[index])+".mp4", "720p60", bin_edges[index], bin_edges[index+1])
                    video["downloaded"] = "true"
                else:
                    os.remove(root+"dataset/comments/"+category.rstrip(".json")+"/"+video["id"]+".json")
                    useless_indexes.append(video_index)
        for index in reversed(useless_indexes):
            category_json.pop(index)
    with open(root+"dataset/headers/categories/"+category, "w") as cat:
        json.dump(category_json, fp=cat, indent=4)

def dl_all_categories(root="/media/livehat/"):
    """
    Download comments and videos for all categories in the dataset.

    This function reads the 'categories.json' file, which contains information about all categories and their IDs.
    It then iterates through each category and calls the 'dl_category' function to download comments and videos
    for that category.

    The 'categories.json' file is assumed to be located in the 'dataset/headers' folder.

    @param root: str, optional (default="/media/livehat/")
        The root directory for the dataset.

    """
    with open(root+"dataset/headers/categories.json") as categories:
        categories_json = json.load(categories)
    for cat in categories_json:
        logging.info("START downloading category "+str(cat["id"]))
        dl_category(cat["id"]+".json", root="/media/livehat/")

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    get_videos_from_all_categories(period="week", root="/media/livehat/")
    dl_all_categories(root="/media/livehat/")
