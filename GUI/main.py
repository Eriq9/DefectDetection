import tkinter
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
Label(root, text="Rezultaty:",font=("Helvetica", 10)).place(x=900, y=125)                                       #Wyniki

Label(root, text="Ilość znalezionych defektów:",font=("Helvetica", 10)).place(x=900, y=175)
Label(root, text="Powierzchnia zniszczonych elementów:",font=("Helvetica", 10)).place(x=900, y=225)
Label(root, text="Procent zniszczenia deski:",font=("Helvetica", 10)).place(x=900, y=275)

def open_filename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(initialdir="/Users/Eryk/Desktop/deski",title="Wybór zdjęcia")
    return filename

def open_random_photo():
    global random_image
    #file_path_type = ["/Users/Eryk/Desktop/deski/2_proba/*.jpg"]
    file_path_type = ["/Users/Eryk/Desktop/deskinowe/*.jpg"]
    random_filename = glob.glob(random.choice(file_path_type))
    random_image = random.choice(random_filename)
    print("random image ",random_image)
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


    DefectCountLabel = Label(root, text="                                                        ")
    DefectCountLabel.place(x=1150, y=175)

    DefectAreaLabel = Label(root, text="                                                         ")
    DefectAreaLabel.place(x=1150, y=225)

    DefectPercentLabel = Label(root, text="                                                      ")
    DefectPercentLabel.place(x=1150, y=275)


    algorithm = ImageProcessingAlgorithms(random_image)
    #print("dsdsd ",algorithm.random_image_path)
    Result = algorithm.ImageProcess()

    # zmiana rozmiaru
    random_img2 = ImageTk.PhotoImage(image=Image.fromarray(Result[0]))
    print("Main:",Result[1])
    print("Main:",Result[2])

    DefectCountLabel = Label(root, text=Result[2])
    DefectCountLabel.place(x=1150, y=175)

    DefectAreaLabel = Label(root, text=Result[1])
    DefectAreaLabel.place(x=1150, y=225)

    DefectPercentLabel = Label(root, text=Result[3])
    DefectPercentLabel.place(x=1150, y=275)




    # stworzenie labelki ze zdjęcia
    panel2 = Label(root, image=random_img2, width=500, height=375)
    # dodanie do random_img2
    panel2.image = random_img2
    panel2.place(x=800, y=375)

    #print("result",Result)

    #return Result


# Buttony
Otworz_zdjecie_btn = Button(root, text='Otwórz zdjęcie', command=load_image).place(x=200,y=123)               #Button do wyboru zdjęcia
Losowe_zdjecie_btn = Button(root, text='Wylosuj zdjęcie', command=load_random_image).place(x=200,y=173)       #Button do losowego zdjęcia
Przetworz_btn = Button(root, text='Detekcja', command=Results).place(x=200,y=223)                    #Button do losowego zdjęcia


root.mainloop()

#open_random_photo()