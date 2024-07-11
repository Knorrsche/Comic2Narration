import tkinter as tk
from tkinter import filedialog, messagebox
from src.Classes.Series import Series
from src.Components.ComicPreprocessor import ComicPreprocessor
from src.Utils.IOUtils import convert_pdf_to_image
from src.Components.SpeechBubbleExtractor import SpeechBubbleExtractor
import pyttsx3;
import pickle



class FileInputPage:
    def __init__(self, parent):
        self.parent = parent
        self.frame = tk.Frame(parent.root)
        self.frame.pack(padx=20, pady=20)
        
        self.label = tk.Label(self.frame, text="Comic Selection", font=("Arial", 14, "bold"))
        self.label.pack(pady=10)
        
        self.file_path = None
        self.file_label = tk.Label(self.frame, text="Select Comic File:")
        self.file_label.pack(pady=5)
        
        self.file_entry = tk.Entry(self.frame, width=50)
        self.file_entry.pack(pady=5)
        
        self.select_file_button = tk.Button(self.frame, text="Browse", command=self.browse_file)
        self.select_file_button.pack(pady=5)

        self.name_label = tk.Label(self.frame, text="Name:")
        self.name_label.pack(pady=5)
        self.name_entry = tk.Entry(self.frame, width=50)
        self.name_entry.pack(pady=5)

        self.volume_label = tk.Label(self.frame, text="Volume:")
        self.volume_label.pack(pady=5)
        self.volume_entry = tk.Entry(self.frame, width=50)
        self.volume_entry.pack(pady=5)

        self.main_series_label = tk.Label(self.frame, text="Main Series:")
        self.main_series_label.pack(pady=5)
        self.main_series_entry = tk.Entry(self.frame, width=50)
        self.main_series_entry.pack(pady=5)

        self.secondary_series_label = tk.Label(self.frame, text="Secondary Series:")
        self.secondary_series_label.pack(pady=5)
        self.secondary_series_entry = tk.Entry(self.frame, width=50)
        self.secondary_series_entry.pack(pady=5)

        self.next_button = tk.Button(self.frame, text="Next", command=self.handle_next)
        self.next_button.pack(pady=10)
        
        self.import_button = tk.Button(self.frame,text="Import",command=self.import_comic)
        self.import_button.pack(pady=10)

        self.talk_button = tk.Button(self.frame,text="Talk",command=self.talk)
        self.talk_button.pack(pady=10)

    def talk(self):
        engine = pyttsx3.init();
        engine.say("Hello guys today i am talking and how are you all doing this is a story about spiderman");
        engine.runAndWait() ;

        
    def browse_file(self):
        filename = filedialog.askopenfilename(parent=self.parent.root, title='Browse File')
        if filename:
            self.file_path = filename
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            print(f"Selected file: {filename}")
    
    def handle_next(self):
        file_path = self.file_entry.get()
        name = self.name_entry.get()
        volume = self.volume_entry.get()
        main_series = Series(name=self.main_series_entry.get())
        secondary_series = []
        secondary_series.append(self.secondary_series_entry.get())

        if not (file_path and name and volume and main_series.name):
            messagebox.showerror("Error", "Please fill in all fields (Secondary Series is Optional).")
            return

        messagebox.showinfo("Info", "The Comic PDF is now being read. This may take a few minutes. Please wait.")

        rgb_arrays = convert_pdf_to_image(file_path)
        self.comic_preprocessor = ComicPreprocessor(name, volume, main_series, secondary_series,rgb_arrays)
        self.speech_bubble_extractor = SpeechBubbleExtractor(self.comic_preprocessor.current_comic)
        #TODO: Remove this if pdf export works
        self.save_comic_data('comic.pkl', self.comic_preprocessor.current_comic)
        self.parent.show_comic_display_screen(self.speech_bubble_extractor.current_comic)

    def save_comic_data(self,filename, data):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    def load_comic_data(self,filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    #TODO: Implement import if pdf export works
    def import_comic(self):
        self.parent.show_comic_display_screen(self.load_comic_data('comic.pkl'))