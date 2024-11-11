import tkinter as tk
from tkinter import filedialog, messagebox
from Classes.Series import Series
from Components.ComicPreprocessor import ComicPreprocessor
from Utils.IOUtils import convert_pdf_to_image,parse_comic,read_xml_from_pdf,add_image_data
from Components.SpeechBubbleExtractor import SpeechBubbleExtractor
import pyttsx3


class FileInputPage:
    def __init__(self, parent):
        self.speech_bubble_extractor = None
        self.comic_preprocessor = None
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

        self.import_button = tk.Button(self.frame, text="Import", command=self.import_comic)
        self.import_button.pack(pady=10)

        self.talk_button = tk.Button(self.frame, text="Talk", command=self.talk)
        self.talk_button.pack(pady=10)

    @staticmethod
    def talk():
        engine = pyttsx3.init()
        engine.say("Hello guys today i am talking and how are you all doing this is a story about spiderman")
        engine.runAndWait()

    def browse_file(self):
        filename = filedialog.askopenfilename(parent=self.parent.root, title='Browse File',
                                              filetypes=(("PDF files", "*.pdf"), ("All files", "*.*")))
        if filename:
            self.file_path = filename
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)

    def handle_next(self):
        file_path = self.file_entry.get()
        name = self.name_entry.get()
        volume = self.volume_entry.get()
        main_series = Series(name=self.main_series_entry.get())
        secondary_series = [self.secondary_series_entry.get()]

        if not (file_path and name and volume and main_series.name):
            messagebox.showerror("Error", "Please fill in all fields (Secondary Series is Optional).")
            return

        messagebox.showinfo("Info", "The Comic PDF is now being read. This may take a few minutes. Please wait.")

        rgb_arrays = convert_pdf_to_image(file_path)
        self.comic_preprocessor = ComicPreprocessor(name, volume, main_series, secondary_series, rgb_arrays)
        self.speech_bubble_extractor = SpeechBubbleExtractor(self.comic_preprocessor.current_comic,self.comic_preprocessor)
        self.parent.show_comic_display_screen(self.speech_bubble_extractor.current_comic, self.comic_preprocessor,file_path,)

    def import_comic(self):
        self.browse_file()
        if self.file_path:
            try:
                embedded_xml = read_xml_from_pdf(self.file_path)
                imported_comic = parse_comic(embedded_xml)
                add_image_data(imported_comic,self.file_path)
                imported_comic.update_scenes()
                self.parent.show_comic_display_screen(imported_comic,self.file_path)
            except Exception as e:
                e.with_traceback()
                print(e)
                messagebox.showerror("Error", f"Failed to import XML file: {e}")
