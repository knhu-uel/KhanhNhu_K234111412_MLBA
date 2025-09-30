import tkinter as tk
from tkinter import ttk
import requests


class TextTranslatorApp:
    def __init__(self, root):
        self.root = root
        root.title("Text Translator")
        self.create_widgets()

    def create_widgets(self):
        label1 = tk.Label(self.root, text="Enter text to translate:")
        self.entry = tk.Entry(self.root, width=50)

        label2 = tk.Label(self.root, text="Choose source language:")
        self.source_lang = ttk.Combobox(self.root, values=["en", "es", "fr", "vi", "ja", "zh"])
        self.source_lang.set("en")

        label3 = tk.Label(self.root, text="Choose target language:")
        self.target_lang = ttk.Combobox(self.root, values=["en", "es", "fr", "vi", "ja", "zh"])
        self.target_lang.set("vi")

        translate_button = tk.Button(self.root, text="Translate", command=self.translate_text)

        self.result_label = tk.Label(self.root, text="Translated text will appear here.")

        # Grid layout
        label1.grid(row=0, column=0, padx=10, pady=10)
        self.entry.grid(row=0, column=1, padx=10, pady=10)
        label2.grid(row=1, column=0, padx=10, pady=10)
        self.source_lang.grid(row=1, column=1, padx=10, pady=10)
        label3.grid(row=2, column=0, padx=10, pady=10)
        self.target_lang.grid(row=2, column=1, padx=10, pady=10)
        translate_button.grid(row=3, column=0, columnspan=2, pady=10)
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

    def translate_text(self):
        api_key = "AIzaSyDNipyDAJDyranS9msg3KaqQy91FhmsrSg"
        text_to_translate = self.entry.get()

        url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
        params = {
            'q': text_to_translate,
            'source': self.source_lang.get(),
            'target': self.target_lang.get(),
        }

        response = requests.post(url, params=params)
        translated_text = response.json()['data']['translations'][0]['translatedText']
        self.result_label.config(text=translated_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = TextTranslatorApp(root)
    root.mainloop()
