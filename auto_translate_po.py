import polib
from googletrans import Translator

# Configure
po_files = [
    ("translations/hi/LC_MESSAGES/messages.po", "hi"),
    ("translations/te/LC_MESSAGES/messages.po", "te"),
]

translator = Translator()

for po_file, lang in po_files:
    po = polib.pofile(po_file)
    for entry in po:
        if not entry.msgstr.strip():  # only translate empty entries
            try:
                translated = translator.translate(entry.msgid, dest=lang).text
                entry.msgstr = translated
                print(f"{lang}: {entry.msgid} -> {translated}")
            except Exception as e:
                print(f"Error translating '{entry.msgid}': {e}")
    po.save()
    print(f"Saved translations to {po_file}")
