import os
import cv2
import time

# Set to fullscreen
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def delete_oldest_files(directory, max_files):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if len(files) > max_files:
        files.sort(key=os.path.getctime)  # Sort files by creation time
        for f in files[:-max_files]:  # Keep the newest 'max_files' files
            print(f"Deleting old file: {f}")
            os.remove(f)

def play():
    delete_oldest_files("test_morphs", 1000)
    paths = sorted([os.path.join("test_morphs", im) for im in os.listdir("test_morphs")])
    while True:
        for im in paths:
            frame = cv2.imread(im)
            if frame is None:
                print(f"Error loading image: {im}")
                continue
            print(im)
            cv2.imshow("window", frame)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                return  # Exit the function if 'q' is pressed

def play_with_restart():
    while True:
        try:
            play()
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Restarting the play function...")
            time.sleep(1)  # Wait for a short time before restarting

# Call the play_with_restart function
play_with_restart()
cv2.destroyAllWindows()  # Clean up and close the window when the loop exits
