"""
 *  @file  crop_image.py
 *  @brief Prints out the cropped co-ordinates based out of the drawn rectangle or polygon
 *
 *  @author Kalp Garg.
"""
import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

class ImageCropper:
    def __init__(self, image_path, extraction_type="rectangle"):
        self.image_path = image_path
        self.start_x, self.start_y = None, None
        self.end_x, self.end_y = None, None
        self.vertices = []
        self.drawing = False

        # Create the Tkinter window
        self.root = tk.Tk()
        self.root.title("Image Cropper")

        # Load the image and create an ImageTk object
        self.image = Image.open(self.image_path)
        self.image_tk = ImageTk.PhotoImage(self.image)

        if extraction_type == "rectangle":
            # Create the canvas and bind mouse events
            self.canvas = tk.Canvas(self.root, width=self.image.width, height=self.image.height)
            self.canvas.create_image(0, 0, image=self.image_tk, anchor=tk.NW)
            self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
            self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
            self.canvas.pack()
        elif extraction_type == "polygon":
            # Create a button to start the extraction process
            button = tk.Button(self.root, text="Extract Vertices", command=self.extract_vertices)
            button.pack()
        else:
            print("Extraction type doesn't exist. Choose either from polygon or rectangle")
            quit()

    def run(self):
        self.root.mainloop()

    def mouse_click(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.vertices.append((x, y))
            self.drawing = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def extract_vertices(self):
        global image_path, vertices

        # Load the image
        image = cv2.imread(image_path)

        # Create a window and set the callback function
        cv2.namedWindow('Select Vertices')
        cv2.setMouseCallback('Select Vertices', self.mouse_click)

        while True:
            # Display the image
            cv2.imshow('Select Vertices', image)

            # Break the loop if 'Esc' is pressed
            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

        # Print the extracted vertices
        print("Extracted Vertices:")
        for vertex in self.vertices:
            print(vertex)

        # Perform polygon cropping using the extracted vertices
        self.perform_polygon_cropping()

    def perform_polygon_cropping(self):

        # Load the image
        image = cv2.imread(self.image_path)

        # Create an empty mask with the same dimensions as the image
        mask = np.zeros_like(image)

        # Fill the polygon region in the mask with white (255) pixels
        cv2.fillPoly(mask, [np.array(self.vertices)], (255, 255, 255))

        # Apply the mask to the image to extract the region of interest
        roi = cv2.bitwise_and(image, mask)

        # Display the cropped ROI
        cv2.imshow('ROI', roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    image_path = "/Users/kgarg/extras/home_cam_security/cam_stream_log/input_db/cam2/2.jpg"
    # image_path = "/Users/kgarg/extras/home_cam_security/input/input_db/kalp/kalp0_b8a02c4a-c224-11ed-80d9-acde48001122.jpg"
    cropper = ImageCropper(image_path, extraction_type="polygon")  #extraction_type="rectangle"
    cropper.run()