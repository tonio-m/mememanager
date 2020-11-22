import os
import json
import torch
import filetype
import tkinter as tk
from PIL import Image
from tkinter import filedialog
from torchvision import models
from torchvision import transforms
DISCORD_CACHE_DIR = os.path.expandvars('%APPDATA%/discord/Cache')


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()

        self.button1 = tk.Button(
            self,
            text='change directory',
            command=self.change_directory
        )

        self.button2 = tk.Button(
            self,
            text='import discord cache',
            command=self.import_discord_cache
        )

        self.button3 = tk.Button(
            self,
            text='rename with object detection',
            command=self.rename_with_object_detection
        )

        self.quit = tk.Button(
            self,
            text='quit',
            command=self.master.destroy
        )

        self.text = tk.Text(borderwidth=0)

        self.button1.pack()
        self.button2.pack()
        self.button3.pack()
        self.quit.pack()
        self.text.pack()

        directory_listing = '\n'.join(
                [os.getcwd()] +
                os.listdir()
        )
        self.display_lines(directory_listing)

    def change_directory(self):
        _dir = filedialog.askdirectory()
        if _dir == '':
            return
        os.chdir(_dir)
        directory_listing = '\n'.join([os.getcwd()] + os.listdir())
        self.display_lines(directory_listing)

    def import_discord_cache(self):
        os.chdir(DISCORD_CACHE_DIR)
        directory_listing = '\n'.join(
            [
                'run this app as admin and close discord for this to work.',
                os.getcwd()
            ] + 
            os.listdir()
        )
        self.display_lines(directory_listing)

        if not hasattr(self, 'imported_discord_cache_dir'):
            _dir = filedialog.askdirectory()
            if _dir == '':
                return
            os.mkdir(f'{_dir}/Cache')
            self.imported_discord_cache_dir = f'{_dir}/Cache'
            os.system(f'robocopy {DISCORD_CACHE_DIR} {_dir}/Cache')

            def infer_filetype(path):
                extension = filetype.guess_extension(path)
                if extension is None:
                    return
                return os.rename(path, f'{path}.{extension}')

            [infer_filetype(f'{self.imported_discord_cache_dir}/{f}')
             for f in os.listdir(self.imported_discord_cache_dir)]

        os.chdir(self.imported_discord_cache_dir)
        directory_listing = '\n'.join([os.getcwd()] + os.listdir())
        self.display_lines(directory_listing)

    def display_lines(self, _list):
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, _list)

    def rename_with_object_detection(self):
        index_path = f'{os.path.dirname(os.path.abspath(__file__))}/imagenet_class_index.json'
        imagenet_class_index = json.load(open(index_path))

        mod = models.resnet18(pretrained=True)
        mod = mod.eval()

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        reverse_find = lambda string, char: (len(string) - 1) - string[::-1].find(char)
        def detect_img(filename):
            img = Image.open(filename).convert('RGB')
            img = preprocess(img)
            img = torch.unsqueeze(img, 0)
            probs = mod(img)
            scores = [(i, float(score)) for i, score in enumerate(probs[0])]
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            return [tag for tag,score in scores[:3]]

        file_tags = [(filename, detect_img(filename)) for filename in os.listdir()]
        [os.rename(
            filename, 
            f"{'_'.join(tags)}{filename[reverse_find(filename,'.'):]}")
            for filename, tags in file_tags
        ]


root = tk.Tk()
app = Application(master=root)
root.wm_title('mememanager')
app.mainloop()
