import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from Classes import Page
from Utils import IOUtils as io


class ComicDisplayPage:
    def __init__(self, parent, comic, filepath):
        self.parent = parent
        self.comic = comic
        self.current_page_pair_index = 0
        self.last_window_height = self.parent.root.winfo_height()
        self.last_window_width = self.parent.root.winfo_width()
        self.show_speech_bubbles = True
        self.show_panels = True
        self.import_path = filepath

        self.menu = tk.Menu(self.parent.root)
        self.parent.root.config(menu=self.menu)
        self.create_menu()

        self.frame_left = tk.Frame(self.parent.root)
        self.frame_left.pack(side='left', expand=True, fill='both')

        self.frame_right = tk.Frame(self.parent.root)
        self.frame_right.pack(side='right', expand=True, fill='both')

        self.label_left = tk.Label(self.frame_left)
        self.label_left.pack(expand=True, fill='both')

        self.label_right = tk.Label(self.frame_right)
        self.label_right.pack(expand=True, fill='both')

        self.left_image = None
        self.right_image = None
        self.left_photo_image = None
        self.right_photo_image = None

        self.resize_pending = None
        self.resize_delay = 10  # milliseconds

        self.parent.root.bind('<Configure>', self.on_window_resize)
        self.parent.root.bind('<Left>', self.show_previous_page_pair)
        self.parent.root.bind('<Right>', self.show_next_page_pair)

        self.display_images()

    def create_menu(self):
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_comic)
        file_menu.add_command(label="Save", command=self.save_comic)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.parent.root.quit)

        export_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export XML", command=self.export_as_xml)
        export_menu.add_command(label="Export Annotated PDF", command=self.export_as_annotated_pdf)

        display_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Display", menu=display_menu)
        display_menu.add_command(label="Toggle Panels", command=self.toggle_panels)
        display_menu.add_command(label="Toggle Speech Bubbles", command=self.toggle_speech_bubbles)

    def display_images(self):
        if self.current_page_pair_index < 0 or self.current_page_pair_index >= len(self.comic.page_pairs):
            return

        left_page: Page = self.comic.page_pairs[self.current_page_pair_index][0]
        right_page: Page = self.comic.page_pairs[self.current_page_pair_index][1]

        left_image_array = left_page.annotated_image(self.show_panels,
                                                     self.show_speech_bubbles) if left_page is not None else self.create_blank_image()
        right_image_array = right_page.annotated_image(self.show_panels,
                                                       self.show_speech_bubbles) if right_page is not None else self.create_blank_image()

        self.left_image = Image.fromarray(left_image_array)
        self.right_image = Image.fromarray(right_image_array)

        self.resize_and_update_images()

    def resize_and_update_images(self):
        if self.left_image is None or self.right_image is None:
            return

        max_width = self.parent.root.winfo_width() // 2
        max_height = self.parent.root.winfo_height()

        resized_left_image = self.left_image.resize((max_width, max_height), Image.LANCZOS)
        resized_right_image = self.right_image.resize((max_width, max_height), Image.LANCZOS)

        self.left_photo_image = ImageTk.PhotoImage(resized_left_image)
        self.right_photo_image = ImageTk.PhotoImage(resized_right_image)

        self.label_left.config(image=self.left_photo_image)
        self.label_right.config(image=self.right_photo_image)

        self.frame_left.config(width=max_width)
        self.frame_right.config(width=max_width)

        self.label_left.bind('<Button-1>', self.show_previous_page_pair)
        self.label_right.bind('<Button-1>', self.show_next_page_pair)

    def create_blank_image(self):
        width, height = 800, 600
        blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        return blank_image

    def on_window_resize(self, event):
        if event.width == self.last_window_width and event.height == self.last_window_height:
            return

        if event.widget == self.parent.root:
            if self.resize_pending:
                self.parent.root.after_cancel(self.resize_pending)
            self.last_window_width = event.width
            self.last_window_height = event.height
            self.resize_pending = self.parent.root.after(self.resize_delay, self.handle_delayed_resize)

    def handle_delayed_resize(self):
        self.resize_and_update_images()
        self.resize_pending = None

    def show_previous_page_pair(self, event):
        if self.current_page_pair_index > 0:
            self.current_page_pair_index -= 1
            self.display_images()

    def show_next_page_pair(self, event):
        if self.current_page_pair_index < len(self.comic.page_pairs) - 1:
            self.current_page_pair_index += 1
            self.display_images()

    def toggle_panels(self):
        self.show_panels = not self.show_panels
        self.display_images()

    def toggle_speech_bubbles(self):
        self.show_speech_bubbles = not self.show_speech_bubbles
        self.display_images()

    def export_as_xml(self):
        export_path = filedialog.asksaveasfilename(defaultextension=".xml",
                                                   filetypes=[("XML Files", "*.xml")],
                                                   title="Select where to save the XML")
        if export_path:
            xml = self.comic.to_xml()
            xml_str = io.prettify_xml(xml)
            io.save_xml_to_file(export_path, xml_str)

            messagebox.showinfo("Export Successfull", "The XML was exported successfully")
        else:
            messagebox.showerror("Error", "Error while trying to export, please try again")

    def export_as_annotated_pdf(self):
        export_path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                   filetypes=[("PDF files", "*.pdf")],
                                                   title="Select where to save the annotated PDF")

        if export_path:
            xml = self.comic.to_xml()
            xml_str = io.prettify_xml(xml)
            io.add_annotation_to_pdf(self.import_path, xml_str,export_path)

            messagebox.showinfo("Export Successfully", "The annotated PDF was exported successfully")
        else:
            messagebox.showerror("Error", "Error while trying to export, please try again")

    #TODO: Add load comic
    def open_comic(self):
        pass

    # TODO: Add save comic
    def save_comic(self):
        pass
