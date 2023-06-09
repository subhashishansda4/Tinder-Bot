## Description
Built a **Tinder Bot** to automatically *like/dislike* potential partners according to a preference set by the user\
The bot compares the facial features of someone the user is attracted to and every other profile on Tinder to find the best match possible

https://github.com/subhashishansda4/Tinder-Bot/assets/84849423/db00a887-6dcd-4a14-85e6-4b3e3f4c6849

## Methodology
### Raw Data
Collected photos of actresses I found cute used **OpenCV** to generate embeddings for their facial features

### Tinder API
Took help from this github repo on how to reverse engineer the api\
[auto-tinder](https://github.com/joelbarmettlerUZH/auto-tinder)

Collected profile data and photos and logged it in a text file and folder

![logs](https://github.com/subhashishansda4/Tinder-Bot/blob/main/misc/logs.jpg)

### Face Detection & Rating
Made functions to detect faces in the photos and rate them based on the **cosine similarity** between faces of that of actresses

Rated the photos and generated a score to use as a threshold for like/dislike by using facial embeddings

![likes](https://github.com/subhashishansda4/Tinder-Bot/blob/main/misc/likes.jpg)

## Conclusion
I AM STILL SINGLE :)
