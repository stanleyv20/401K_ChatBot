import tkinter

from execute_chat import get_bot_message, chatbot_name
from tkinter import *

# Defining some colors, manipulate as needed
BG_GRAY = 'white smoke'
BG_COLOR = '#17202A' # #116562
TEXT_COLOR = '#EAECEE'

FONT = 'Helvetica 14'
FONT_BOLD = 'Helvetica 13 bold'


class ChatBotApp:
    def __init__(self):
        self.window = Tk()
        self._generate_primary_window()

    def run(self):
        self.window.mainloop()

    def _generate_primary_window(self):
        self.window.title("401k Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        head_label = tkinter.Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR, text='401K ChatBot', font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)  # takes entire width of GUI window

        # Setting up divider
        line = Label(self.window, width=450)
        line.place(relwidth=1, rely=.07, relheight=.012)

        # Text widget
        self.text_view = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, padx=5, pady=5)
        self.text_view.place(relheight=.745, relwidth=1, rely=.08)
        self.text_view.configure(cursor="arrow", state=DISABLED)

        # Enable Scrolling
        my_scrollbar = Scrollbar(self.text_view)
        my_scrollbar.place(relheight=1, relx=0.97)
        my_scrollbar.configure(command=self.text_view.yview())

        # Last label
        below_label = Label(self.window, bg=BG_GRAY, height=80)
        below_label.place(relwidth=1, rely=.825)

        # Submit button
        send_button = Button(below_label, text="SEND", font=FONT_BOLD, width=20, bg=BG_GRAY,
                             command=lambda: self._submit_input(None))
        send_button.place(relx=.77, rely=.008, relheight=.06, relwidth=.22)

        # Enable entering input text
        self.text_input = Entry(below_label, bg='white smoke', fg='grey1', font=FONT)
        self.text_input.place(relwidth=.74, relheight=.06, rely=.008, relx=.011)
        self.text_input.focus()
        self.text_input.bind("<Return>", self._submit_input)

    def _submit_input(self, event):
        # Retrieves input text as a string
        text = self.text_input.get()
        self._insert_message(text, 'User Input')

    def _insert_message(self, text, sender):
        # In case of empty input
        if not text:
            return

        self.text_input.delete(0, END)
        message_1 = f'{sender}: {text}\n\n'
        self.text_view.configure(state=NORMAL)
        self.text_view.insert(END, message_1)
        self.text_view.configure(state=DISABLED)


if __name__ == '__main__':
    ui_app = ChatBotApp()
    ui_app.run()
