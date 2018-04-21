import text_classifier
from Tkinter import *
from math import *

import Tkinter as tk

class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.entry = tk.Entry(self)
        self.button = tk.Button(self, text="Predict Genre", command=self.on_button)
        self.button.pack(side='bottom')
        self.entry.pack()
        self.labels = []
        
    def on_button(self):
        out = text_classifier.test([self.entry.get()])
        #self.button.destroy()
        for label in self.labels:
            label.destroy()
        label = Label(self, text= str(out))
        self.labels.append(label)
        label.pack()
        
app = SampleApp()
app.mainloop()
