from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import glob
import random

from Backend.test import ImageProcessingAlgorithms




root = Tk()
root.title('Aplikacja wykrywająca defekty w deskach')
root.geometry("1400x800")
root.resizable(True,True)

#Labels


Label(root, text="Aplikacja wykrywająca defekty w deskach",font=("Helvetica", 18)).place(x=450, y=25)           #Title label
Label(root, text="Wybierz zdjęcie do analizy:",font=("Helvetica", 10)).place(x=20, y=125)                       #Załaduj zdjęcie
Label(root, text="Załaduj losowe zdjęcie:",font=("Helvetica", 10)).place(x=20, y=175)                           #Załaduj losowe zdjęcie

Label(root, text="Wybrane zdjęcie:",font=("Helvetica", 14)).place(x=275, y=325)                                 #Wybrane zdjęcie
Label(root, text="Przeanalizowane zdjęcie:",font=("Helvetica", 14)).place(x=950, y=325)                         #Przeanalizowane zdjęcie
Label(root, text="Wykonaj detekcję:",font=("Helvetica", 10)).place(x=20, y=225)                                 #Załaduj losowe zdjęcie

Label(root, text="Parametry analizy:",font=("Helvetica", 10)).place(x=400, y=125)                               #Wybór parametrów
Label(root, text="Rezultaty:",font=("Helvetica", 10)).place(x=900, y=125)                               #Wyniki


def open_filename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(initialdir="/Users/Eryk/Desktop/deski",title="Wybór zdjęcia")
    return filename

def open_random_photo():

    file_path_type = ["/Users/Eryk/Desktop/deski/2_proba/*.jpg"]
    random_filename = glob.glob(random.choice(file_path_type))
    random_image = random.choice(random_filename)
    print(random_image)

    return random_image

def load_image():

    # przypisanie ścieżki
    path = open_filename()
    # otwarcie zdjęcia
    img = Image.open(path)
    # zmiana rozmiaru
    img = img.resize((500, 375))
    img = ImageTk.PhotoImage(img)
    # stworzenie labelki ze zdjęcia
    panel = Label(root, image=img)
    # dodanie do aplikacji
    panel.image = img
    panel.place(x=100, y=375)

def load_random_image():

    # przypisanie ścieżki
    random_path = open_random_photo()
    # otwarcie zdjęcia
    random_img = Image.open(random_path)
    # zmiana rozmiaru
    random_img = random_img.resize((500, 375))
    random_img = ImageTk.PhotoImage(random_img)
    # stworzenie labelki ze zdjęcia
    panel = Label(root, image=random_img)
    # dodanie do aplikacji
    panel.image = random_img
    panel.place(x=100, y=375)

def Results():
    algorithm = ImageProcessingAlgorithms(open_random_photo())
    Result = algorithm.ImageProcess()

    return Result


# Buttony
Otworz_zdjecie_btn = Button(root, text='Otwórz zdjęcie', command=load_image).place(x=200,y=123)               #Button do wyboru zdjęcia
Losowe_zdjecie_btn = Button(root, text='Wylosuj zdjęcie', command=load_random_image).place(x=200,y=173)       #Button do losowego zdjęcia
Przetworz_btn = Button(root, text='Detekcja', command=Results).place(x=200,y=223)                    #Button do losowego zdjęcia


root.mainloop()

#open_random_photo()