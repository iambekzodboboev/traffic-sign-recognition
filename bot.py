"""Stage 7.4 -- Telegram bot demo. User sends a photo of a sign; bot replies
with the predicted sign name, its category (mandatory/warning/etc.),
confidence, and a clean reference picture of that sign.

Reuses the same model-loading/prediction logic as scripts/predict_sign.py
and runs entirely on CPU -- no GPU, dataset, or Kaggle/Colab needed.

Run locally:
    python bot.py
Then message your bot on Telegram (search for the username you gave
BotFather) and send it a photo.
"""
import os
from pathlib import Path

import pandas as pd
import telebot
from dotenv import load_dotenv

from scripts.predict_sign import load_model, predict

load_dotenv()

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_NAMES_PATH = PROJECT_ROOT / "metadata" / "class_names.csv"
ICONS_DIR = PROJECT_ROOT / "assets" / "class_icons"

bot = telebot.TeleBot(BOT_TOKEN)
class_names_df = pd.read_csv(CLASS_NAMES_PATH)
model = load_model()
print("Model loaded. Bot starting...")


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(
        message,
        "Send me a photo of a traffic sign and I'll tell you what it is, "
        "its category, and how confident I am."
    )


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded = bot.download_file(file_info.file_path)

    tmp_path = PROJECT_ROOT / "bot_tmp_input.jpg"
    tmp_path.write_bytes(downloaded)

    try:
        results = predict(tmp_path, model, class_names_df, top_k=1)
        class_id, name, confidence = results[0]

        category_row = class_names_df.loc[class_names_df["class_id"] == class_id, "category"]
        category = category_row.values[0] if len(category_row) else "Unknown"

        caption = (
            f"Sign: {name}\n"
            f"Category: {category}\n"
            f"Confidence: {confidence * 100:.1f}%"
        )

        icon_candidates = list(ICONS_DIR.glob(f"{class_id}.*"))
        if icon_candidates:
            with open(icon_candidates[0], "rb") as icon_file:
                bot.send_photo(message.chat.id, icon_file, caption=caption)
        else:
            bot.reply_to(message, caption)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    bot.infinity_polling()
