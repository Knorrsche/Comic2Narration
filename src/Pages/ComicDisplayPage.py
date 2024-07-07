import tkinter as tk
from tkinter import IntVar
from Classes.Comic import Comic
from PIL import Image,ImageTk,ImageDraw

class ComicDisplayPage:
    def __init__(self,parent,comic:Comic):
        self.parent = parent
        self.comic = comic
        self.current_page_pair_index = 0
        
        self.frame = tk.Frame(parent.root)
        self.frame.pack(padx=20,pady=20)

        self.label = tk.Label(self.frame,text="hallo",font=("Arial",14,"bold"))
        self.label.grid(row=0,column=0,pady=10,columnspan=2)