"""
 *  @file  crop_image.py
 *  @brief Prints out the cropped co-ordinates based out of the drawn rectangle.
 *
 *  @author Kalp Garg.
"""
import tkinter as tk
from PIL import ImageTk, Image

class ImageCropper:
    def __init__(self, image_path):
        self.image_path = image_path
        self.start_x, self.start_y = None, None
        self.end_x, self.end_y = None, None

        # Create the Tkinter window
        self.root = tk.Tk()
        self.root.title("Image Cropper")

        # Load the image and create an ImageTk object
        self.image = Image.open(self.image_path)
        self.image_tk = ImageTk.PhotoImage(self.image)

        # Create the canvas and bind mouse events
        self.canvas = tk.Canvas(self.root, width=self.image.width, height=self.image.height)
        self.canvas.create_image(0, 0, image=self.image_tk, anchor=tk.NW)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.pack()

    def run(self):
        self.root.mainloop()

    def on_mouse_press(self, event):
        self.start_x, self.start_y = event.x, event.y

    def on_mouse_drag(self, event):
        self.end_x, self.end_y = event.x, event.y
        self.canvas.delete("rectangle")
        self.canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline="red", tags="rectangle")

    def on_mouse_release(self, event):
        # Crop the image using the selected coordinates
        x = min(self.start_x, self.end_x)
        y = min(self.start_y, self.end_y)
        width = abs(self.start_x - self.end_x)
        height = abs(self.start_y - self.end_y)
        print(x, y, x + width, y + height)
        cropped_image = self.image.crop((x, y, x + width, y + height))
        cropped_image.show()

        # Reset the coordinates
        self.start_x, self.start_y = None, None
        self.end_x, self.end_y = None, None

if __name__ == "__main__":
    image_path = "/Users/kgarg/extras/home_cam_security/capture_face_log/input_db/raw_untouched_data/del_me1_2a9cdee8-09aa-11ee-bf25-d68f61189c9d.jpg"
    cropper = ImageCropper(image_path)
    cropper.run()