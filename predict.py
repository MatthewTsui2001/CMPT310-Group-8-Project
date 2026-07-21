# Importing libraries
import tkinter as tk
import tensorflow as tf
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
from model import model


#model = tf.keras.models.load_model('my_model.keras')

# image uploader function
def imageUploader():
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    # if file is selected
    if len(path):
        img = Image.open(path).convert("RGB")

        # --- Display version ---
        img_display = img.resize((200, 200))
        pic = ImageTk.PhotoImage(img_display)

        # re-sizing the app window in order to fit picture
        # and buttom
        app.geometry("560x300")
        imgLabel.config(image=pic)
        imgLabel.image = pic

        img_for_model = img.resize((32, 32))
        img_array = tf.keras.utils.img_to_array(img_for_model)
        img_array = tf.expand_dims(img_array, 0)

    # if no file is selected, then we are displaying below message
    else:
        print("No file is Choosen !! Please choose a file.")




# defining tkinter object
app = tk.Tk()

# setting title and basic size to our App
app.title("AI Image Detector")
app.geometry("560x270")

# adding background image
imgLabel = Label(app)
imgLabel.pack(side = tk.TOP)



labels = tk.Label(app, text="Original Text", fg="Green", bg="white", font=("Arial", 14))
labels.pack(side=tk.BOTTOM, pady=5)

# adding background color to our upload button
app.option_add("*Label*Background", "white")
app.option_add("*Button*Background", "lightgreen")


# defining our upload buttom
uploadButton = tk.Button(app, text="Upload Image", command=imageUploader)
uploadButton.pack(side=tk.BOTTOM, pady=20)


app.mainloop()