import tkinter as tk
from Pages.FileInputPage import FileInputPage
from Pages.ComicDisplayPage import ComicDisplayPage
from Classes.Comic import Comic
from Components.ComicPreprocessor import ComicPreprocessor
from Components.SpeechBubbleExtractor import SpeechBubbleExtractor

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title('Comic Reader')
        self.root.geometry("800x600")
        self.comic_preprocessor: ComicPreprocessor
        self.speech_bubble_extractor: SpeechBubbleExtractor

        self.file_input_screen = FileInputPage(self)
        self.comic_display_screen = None
        
        self.show_file_input_screen()
        
    def show_file_input_screen(self):
        if self.comic_display_screen:
            self.comic_display_screen.frame.pack_forget()
        
        self.file_input_screen.frame.pack(padx=20, pady=20)
    
    def show_comic_display_screen(self,comic_preprocessor: ComicPreprocessor):
        self.file_input_screen.frame.pack_forget()
        
        self.comic_display_screen = ComicDisplayPage(self, comic_preprocessor)
        self.comic_display_screen.frame.pack(padx=20, pady=20)
    
def main():
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
    
if __name__ == "__main__":
    main()
