import os
import cv2
from pytubefix import YouTube

with open('list_youtube_video_release.txt', 'r') as file:
    VIDEO_URL_txt = file.read().splitlines()

VIDEO_URL_list = []
for video_url_folder in VIDEO_URL_txt:
    print(video_url_folder)
    vvideo_url, folder, num_skip = video_url_folder.split(" ")
    VIDEO_URL_list.append([vvideo_url, folder, float(num_skip)])
    
# Configuration
OUTPUT_DIR = './videos'

# Create directories if they don't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Download the video
def download_video(url, output_path, filename):
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    print(f"Downloading video: {yt.title}")
    stream.download(output_path=output_path, filename=filename)
    print("Download completed!")

# Extract images from the video
def extract_frames(video_path, save_folder, num_skip):
    images_dir = os.path.join("../dataset/dataset_LeLaN_youtube", folder, 'image')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)    

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    pickles_dir = os.path.join("../dataset/dataset_LeLaN_youtube", folder, 'pickle') 
    n_pic = len(os.listdir(pickles_dir))
    #n_pic = 10
        
    print("fps", fps, "N_pic", n_pic, "folder", save_folder)
    if fps == 0.0:
        fps = 15    

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seconds_per_frame = 1 / fps
    
    frame_count = 0
    save_count = 0 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # save image at 2 fps (removing first num_skip images. The total frames is less than n_pic.)
        if num_skip == 60.0 or num_skip == 5.0:
            if frame_count % int(fps*0.5) == 0 and frame_count > num_skip*fps and save_count < n_pic:      
                frame_process = frame
                print(frame_count)
                cv2.imwrite(images_dir + "/" + str(save_count).zfill(8) + ".jpg", cv2.resize(frame_process, (224, 224), interpolation = cv2.INTER_LINEAR))
                save_count += 1
        else:
            if (frame_count - int(num_skip*fps) - 1) % int(fps*0.5) == 0 and frame_count > num_skip*fps and save_count < n_pic:        
                frame_process = frame
                print(frame_count)
                cv2.imwrite(images_dir + "/" + str(save_count).zfill(8) + ".jpg", cv2.resize(frame_process, (224, 224), interpolation = cv2.INTER_LINEAR))
                save_count += 1
                        
        if save_count == n_pic:
            break
        frame_count += 1
        
    cap.release()
    print(f"Extracted {frame_count} frames!")

# Main function
def main(VIDEO_URL, folder, num_skip):
    # Define paths
    video_file = os.path.join(OUTPUT_DIR, folder + '.mp4')
    # Download video
    print(VIDEO_URL, OUTPUT_DIR)
    download_video(VIDEO_URL, OUTPUT_DIR, folder + '.mp4')

    # Extract frames
    extract_frames(video_file, folder, num_skip)

if __name__ == "__main__":
    for VIDEO_URL, folder, num_skip in VIDEO_URL_list:    
        main(VIDEO_URL, folder, num_skip)
