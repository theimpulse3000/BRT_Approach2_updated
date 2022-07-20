import os
from imbox import Imbox
import traceback

# enable less secure apps on your google account
# https://myaccount.google.com/lesssecureapps

gmail_host = "imap.gmail.com"
outlook_host = "imap-mail.outlook.com"
username = "sagarnmali3000@outlook.com"
password = 'Impulse@3000'
download_folder = "/Users/sagar_19/Downloads/"

if not os.path.isdir(download_folder):
    os.makedirs(download_folder, exist_ok=True)
    
mail = Imbox(outlook_host, username=username, password=password, ssl=True, ssl_context=None, starttls=False)
messages = mail.messages() # defaults to inbox

for (uid, message) in messages:
    mail.mark_seen(uid) # optional, mark message as read

    for idx, attachment in enumerate(message.attachments):
        try:
            att_fn = attachment.get('filename')
            download_path = f"{download_folder}/{att_fn}"
            print(download_path)
            with open(download_path, "wb") as fp:
                fp.write(attachment.get('content').read())
        except:
            print(traceback.print_exc())

mail.logout()


"""
Available Message filters: 

# Gets all messages from the inbox
messages = mail.messages()

# Unread messages
messages = mail.messages(unread=True)

# Flagged messages
messages = mail.messages(flagged=True)

# Un-flagged messages
messages = mail.messages(unflagged=True)

# Messages sent FROM
messages = mail.messages(sent_from='sender@example.org')

# Messages sent TO
messages = mail.messages(sent_to='receiver@example.org')

# Messages received before specific date
messages = mail.messages(date__lt=datetime.date(2018, 7, 31))

# Messages received after specific date
messages = mail.messages(date__gt=datetime.date(2018, 7, 30))

# Messages received on a specific date
messages = mail.messages(date__on=datetime.date(2018, 7, 30))

# Messages whose subjects contain a string
messages = mail.messages(subject='Christmas')

# Messages from a specific folder
messages = mail.messages(folder='Social')
"""
