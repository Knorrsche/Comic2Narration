import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from Classes import Page


class ComicDisplayPage:
    def __init__(self, parent, comic):
        self.parent = parent
        self.comic = comic
        self.current_page_pair_index = 0
        self.last_window_height = self.parent.root.winfo_height()
        self.last_window_width = self.parent.root.winfo_width()

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
        self.parent.root.bind('<Left>',self.show_previous_page_pair)
        self.parent.root.bind('<Right>',self.show_next_page_pair)

        self.display_images()

    def display_images(self):
        if self.current_page_pair_index < 0 or self.current_page_pair_index >= len(self.comic.page_pairs):
            return

        left_page:Page = self.comic.page_pairs[self.current_page_pair_index][0]
        right_page:Page = self.comic.page_pairs[self.current_page_pair_index][1]

        left_image_array = left_page.page_image if left_page is not None else self.create_blank_image()
        right_image_array = right_page.page_image if right_page is not None else self.create_blank_image()

        left_image_array = left_page.annotateted_image() if left_page is not None else left_image_array
        right_image_array = right_page.annotateted_image() if right_page is not None else right_image_array

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

    def show_previous_page_pair(self,event):
        if self.current_page_pair_index > 0:
            self.current_page_pair_index -= 1
            self.display_images()

    def show_next_page_pair(self,event):
        if self.current_page_pair_index < len(self.comic.page_pairs) - 1:
            self.current_page_pair_index += 1
            self.display_images()
