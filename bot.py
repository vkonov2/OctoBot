import logging
import os
import PyPDF2
import pandas as pd
import docx
import mimetypes
import json
import re
from functools import wraps
from dotenv import load_dotenv
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    BotCommand, BotCommandScopeDefault
)
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, ContextTypes, CallbackQueryHandler
)
import aiohttp
import asyncio
import base64
from PIL import Image
from io import BytesIO

# Активные file-сессии: user_id -> dict с vector_store_id, assistant_id, thread_id, file_id, file_name
ACTIVE_FILE_SESSIONS = {}

for logger_name in [
    "httpx",
    "urllib3",
    "telegram",
    "apscheduler",
]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# --- Загрузка переменных окружения ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
PROXYAPI_KEY = os.getenv('PROXYAPI_KEY')
ALLOWED_USER_IDS = set(int(x) for x in os.getenv('ALLOWED_USER_IDS', '').split(',') if x)
ADMIN_USER_IDS = set(int(x) for x in os.getenv('ADMIN_USER_IDS', '').split(',') if x)

PROXYAPI_CHAT_ENDPOINT = "https://api.proxyapi.ru/openai/v1/chat/completions"
PROXYAPI_MODELS_ENDPOINT = "https://api.proxyapi.ru/openai/v1/models"
PROXYAPI_ASR_ENDPOINT = "https://api.proxyapi.ru/openai/v1/audio/transcriptions"

LOG_FORMAT = '[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(lineno)d: %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- История пользователя ---
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)
HISTORY_MAX_PAIRS = 40

def _history_path(user_id):
    return os.path.join(HISTORY_DIR, f"history_{user_id}.json")

def get_user_history(user_id):
    path = _history_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
            return history
        except Exception:
            return []
    return []

def add_to_history(user_id, role, content):
    path = _history_path(user_id)
    history = get_user_history(user_id)
    history.append({"role": role, "content": content})
    if len(history) > HISTORY_MAX_PAIRS * 2:
        history = history[-HISTORY_MAX_PAIRS*2:]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка при сохранении истории в файл: {e}")

def reset_history(user_id):
    path = _history_path(user_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.error(f"Ошибка при удалении истории: {e}")

def collect_history_stats(min_messages=1):
    stats = []
    if not os.path.exists(HISTORY_DIR):
        return stats
    for filename in os.listdir(HISTORY_DIR):
        if filename.startswith("history_") and filename.endswith(".json"):
            try:
                user_id = filename[len("history_"):-len(".json")]
                with open(os.path.join(HISTORY_DIR, filename), "r", encoding="utf-8") as f:
                    history = json.load(f)
                msg_count = len(history)
                if msg_count >= min_messages:
                    stats.append((user_id, msg_count))
            except Exception as e:
                logger.warning(f"Ошибка чтения истории {filename}: {e}")
    stats.sort(key=lambda x: x[1], reverse=True)
    return stats

# --- Модель по умолчанию ---
DEFAULT_MODEL_PATH = "default_model.txt"

def get_default_model():
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            with open(DEFAULT_MODEL_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            pass
    return "gpt-4.1-mini"

def set_default_model(model):
    with open(DEFAULT_MODEL_PATH, "w", encoding="utf-8") as f:
        f.write(model.strip())

# --- Модели ---
RELEVANT_MODELS = [
    "gpt-4o", "gpt-4-vision-preview", "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-2025-04-14",
    "gpt-4", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-0125-preview",
    "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview",
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
]
VISION_MODELS = ["gpt-4-vision-preview", "gpt-4o"]
TOP_MODELS_LIMIT = 10

MODEL_DESCRIPTIONS = {
    "gpt-4": "GPT-4 (флагман, самый точный, но дорогой; хорош для сложных задач)",
    "gpt-4-0125-preview": "GPT-4 January 2025 Preview (обновлённая версия GPT-4 с ускорениями и новыми возможностями)",
    "gpt-4-0613": "GPT-4 June 2023 (старый стабильный релиз; универсальный, быстрый)",
    "gpt-4-1106-preview": "GPT-4 Preview (быстрый, дешевле основного GPT-4, свежие фичи)",
    "gpt-4-turbo": "GPT-4 Turbo (ускоренная, более экономичная версия GPT-4, хорошо подходит для повседневных задач)",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo (апрельское обновление 2024, улучшена работа с контекстом)",
    "gpt-4-turbo-preview": "GPT-4 Turbo Preview (предварительная версия turbo, для тестирования новых функций)",
    "gpt-4.1": "GPT-4.1 (модернизированный GPT-4, ещё более быстрый и точный, оптимизирован для работы с длинным контекстом)",
    "gpt-4.1-2025-04-14": "GPT-4.1 (апрельское обновление 2025, ещё более умная версия GPT-4.1)",
    "gpt-4.1-mini": "GPT-4.1 Mini (упрощённая, быстрая и экономичная версия GPT-4.1, отлично для быстрых диалогов)",
    "gpt-4o": "GPT-4o (мультимодальная, понимает текст, фото, скриншоты)",
    "gpt-4-vision-preview": "GPT-4 Vision (мультимодальная; понимает текст + картинки)",
    "gpt-3.5-turbo": "GPT-3.5 turbo (быстрый и дешевый)",
    "gpt-3.5-turbo-16k": "GPT-3.5 turbo 16k (большой контекст)",
}

ADMIN_ONLY_MODELS = set()

available_models = []
user_models = {}

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "Ты Telegram-бот. Всегда форматируй свои ответы с помощью разметки Telegram HTML: "
        "<b>жирный</b>, <i>курсив</i>, <code>код</code>, <pre>блок кода</pre>. "
        "Не используй markdown! Для многострочного кода всегда используй <pre>...</pre>."
    )
}

async def fetch_available_models():
    headers = {"Authorization": f"Bearer {PROXYAPI_KEY}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(PROXYAPI_MODELS_ENDPOINT, headers=headers, timeout=20) as resp:
                resp.raise_for_status()
                j = await resp.json()
                models = j.get("data", [])
                models = sorted(models, key=lambda m: (not m["id"].startswith("gpt-4"), m["id"]))
                return models
    except Exception as e:
        logger.warning(f"Ошибка загрузки списка моделей: {e}")
        return []

async def reload_models():
    global available_models
    models = await fetch_available_models()
    if not models:
        logger.warning("Не удалось обновить список моделей.")
        return False
    available_models = models
    logger.info(f"Список моделей обновлен, найдено: {len(available_models)}")
    return True

def get_user_model(user_id):
    default_model = get_default_model()
    for m in available_models:
        if m["id"] not in ADMIN_ONLY_MODELS or is_admin(user_id):
            return user_models.get(user_id, default_model if any(am["id"] == default_model for am in available_models) else m["id"])
    return user_models.get(user_id, default_model if any(am["id"] == default_model for am in available_models) else (available_models[0]["id"] if available_models else "gpt-4.1-mini"))

def set_user_model(user_id, model):
    user_models[user_id] = model

def is_allowed(user_id):
    return user_id in ALLOWED_USER_IDS or user_id in ADMIN_USER_IDS

def is_admin(user_id):
    return user_id in ADMIN_USER_IDS

def allowed_only(func):
    @wraps(func)
    async def wrapper(update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if not is_allowed(user_id):
            if hasattr(update, "message") and update.message:
                await update.message.reply_text("У вас нет доступа к этому боту.")
            elif hasattr(update, "callback_query") and update.callback_query:
                await update.callback_query.answer("Нет доступа!")
            logger.warning(f"Попытка доступа от неразрешенного пользователя: {user_id}")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

def admin_only(func):
    @wraps(func)
    async def wrapper(update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if not is_admin(user_id):
            if hasattr(update, "message") and update.message:
                await update.message.reply_text("У вас нет админских прав.")
            elif hasattr(update, "callback_query") and update.callback_query:
                await update.callback_query.answer("Нет админских прав!")
            logger.warning(f"Попытка доступа к админ-команде от пользователя: {user_id}")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

# FSM для рассылки
BROADCAST_STATE_FILE = "broadcast_state.json"

def set_broadcast_wait(user_id):
    with open(BROADCAST_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"user_id": user_id}, f)

def clear_broadcast_wait():
    if os.path.exists(BROADCAST_STATE_FILE):
        os.remove(BROADCAST_STATE_FILE)

def get_broadcast_wait():
    if not os.path.exists(BROADCAST_STATE_FILE):
        return None
    try:
        with open(BROADCAST_STATE_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("user_id")
    except Exception:
        return None

@admin_only
async def broadcast_text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    # Проверяем, ждет ли бот сообщение для рассылки именно от этого админа
    if get_broadcast_wait() == user_id:
        message = update.message.text
        clear_broadcast_wait()
        # Собираем id всех пользователей с историей (можно по логике своей базы)
        all_user_ids = [int(filename[len("history_"):-len(".json")])
                        for filename in os.listdir(HISTORY_DIR)
                        if filename.startswith("history_") and filename.endswith(".json")]
        sent = 0
        for uid in all_user_ids:
            try:
                await context.bot.send_message(uid, message, parse_mode=ParseMode.HTML)
                sent += 1
            except Exception as e:
                logger.warning(f"Не удалось отправить сообщение {uid}: {e}")
        await update.message.reply_text(f"Сообщение отправлено {sent} пользователям.", parse_mode=ParseMode.HTML)
        return True  # обработано
    return False  # не было рассылки, обычная логика

async def notify_admins(application, text, only_first=True):
    for admin_id in ADMIN_USER_IDS:
        try:
            await application.bot.send_message(admin_id, text, parse_mode=ParseMode.HTML)
            if only_first:
                break
        except Exception as e:
            logger.warning(f"Не удалось отправить уведомление админу {admin_id}: {e}")

async def error_handler(update, context):
    import traceback
    tb = traceback.format_exception(None, context.error, context.error.__traceback__)
    error_text = f"❗️Ошибка:\n{''.join(tb)[-2000:]}"
    logger.error(error_text)
    app = context.application if hasattr(context, "application") else None
    if app:
        await notify_admins(app, f"*Ошибка в боте:*\n```{error_text}```", only_first=True)

def markdown_code_to_html(text):
    # Блоки кода
    text = re.sub(r'```(?:\w+)?\n(.*?)```', lambda m: f"<pre>{m.group(1)}</pre>", text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)
    text = text.replace("```", "")

    # Списки Markdown в •
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    # HTML списки в •
    text = re.sub(r"<ul>|</ul>|<ol>|</ol>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<li>(.*?)</li>", r"• \1\n", text, flags=re.IGNORECASE|re.DOTALL)
    # Убираем unsupported теги
    text = re.sub(r"</?(span|table|tr|td|th|hr|br)[^>]*>", "", text, flags=re.IGNORECASE)
    # Убираем любые другие неизвестные теги (опционально, только если очень нужно)
    # text = re.sub(r"</?[a-z][^>]*>", "", text)
    return text

async def download_and_encode_photo(bot, file_id, max_size=2 * 1024 * 1024):
    file = await bot.get_file(file_id)
    file_bytes = await file.download_as_bytearray()
    if len(file_bytes) > max_size:
        image = Image.open(BytesIO(file_bytes))
        scale = (max_size / len(file_bytes)) ** 0.5
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        image = image.resize((max(new_w, 1), max(new_h, 1)))
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
        file_bytes = buf.getvalue()
    base64_data = base64.b64encode(file_bytes).decode()
    mime_type = "image/jpeg"
    return base64_data, mime_type

async def transcribe_voice(bot, voice_file_id):
    file = await bot.get_file(voice_file_id)
    voice_bytes = await file.download_as_bytearray()
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('file', voice_bytes, filename="voice.ogg", content_type='audio/ogg')
        data.add_field('model', 'whisper-1')
        headers = {"Authorization": f"Bearer {PROXYAPI_KEY}"}
        async with session.post(PROXYAPI_ASR_ENDPOINT, headers=headers, data=data) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Ошибка распознавания: {resp.status} {text}")
                raise Exception("Ошибка распознавания речи")
            j = await resp.json()
            return j.get("text", "")

async def ask_llm_stream_history_vision(history, model):
    vision_history = [SYSTEM_PROMPT] + history
    headers = {
        'Authorization': f'Bearer {PROXYAPI_KEY}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": model,
        "messages": vision_history,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": True,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(PROXYAPI_CHAT_ENDPOINT, headers=headers, json=data, timeout=120) as response:
            if response.status != 200:
                logger.error(f"Ответ сервера: {await response.text()}")
                response.raise_for_status()
            async for line in response.content:
                if line:
                    try:
                        line = line.decode()
                        if line.startswith('data:'):
                            json_data = line[len('data:'):].strip()
                            if json_data == '[DONE]':
                                break
                            delta = json.loads(json_data)
                            yield delta
                    except Exception as e:
                        logger.warning(f"Ошибка парсинга vision-стрима: {e}")

@allowed_only
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    photo = update.message.photo[-1]
    caption = update.message.caption or ""
    user_model = get_user_model(user_id)

    reply_ref = ""
    if update.message.reply_to_message:
        orig = update.message.reply_to_message
        reply_content = orig.text or orig.caption or "[нет текста]"
        if len(reply_content) > 500:
            reply_content = reply_content[:500] + "…"
        reply_ref = f'💬 В ответ на:\n"{reply_content}"\n\n'

    prompt = (reply_ref + caption.strip()) if caption or reply_ref else "Опиши это изображение."

    if any(user_model.startswith(m) for m in VISION_MODELS):
        vision_model = user_model
    else:
        vision_model = next((m["id"] for m in available_models if m["id"] in VISION_MODELS), None)
        if not vision_model:
            await update.message.reply_text(
                "Нет доступных моделей, поддерживающих обработку изображений (Vision)."
            )
            return

    base64_img, mime_type = await download_and_encode_photo(context.bot, photo.file_id)

    history = get_user_history(user_id)
    multimodal_content = []
    if prompt:
        multimodal_content.append({"type": "text", "text": prompt})
    multimodal_content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}})
    history.append({"role": "user", "content": multimodal_content})
    add_to_history(user_id, "user", multimodal_content)

    msg = await update.message.reply_text("🖼️ Анализирую изображение...")

    reply_text = ""
    last_sent = ""
    token_count = 0
    has_updated_once = False

    try:
        async for chunk in ask_llm_stream_history_vision(history, vision_model):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0]['delta']
                content = delta.get('content', '')
                if content:
                    reply_text += content
                    token_count += 1
                    if (token_count % 8 == 0 or len(reply_text) - len(last_sent) > 30) or not has_updated_once:
                        try:
                            await msg.edit_text(reply_text + " ▍")
                            last_sent = reply_text
                            has_updated_once = True
                        except Exception as e:
                            logger.warning(f"Ошибка при edit_text: {e}\nОтвет: {reply_text}")
        safe_text = markdown_code_to_html(reply_text)
        await msg.edit_text(safe_text, parse_mode=ParseMode.HTML)
        add_to_history(user_id, "assistant", reply_text)
        logger.info(f'[Vision] Готовый ответ пользователю {user_id}: {reply_text[:100]}...')
    except Exception:
        await msg.edit_text("Ошибка при обработке изображения. Попробуйте позже.")


@allowed_only
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    voice = update.message.voice
    caption = update.message.caption or ""

    reply_ref = ""
    if update.message.reply_to_message:
        orig = update.message.reply_to_message
        reply_content = orig.text or orig.caption or "[нет текста]"
        if len(reply_content) > 500:
            reply_content = reply_content[:500] + "…"
        reply_ref = f'💬 В ответ на:\n"{reply_content}"\n\n'

    msg = await update.message.reply_text("🔊 Расшифровываю голосовое сообщение...")

    try:
        asr_text = await transcribe_voice(context.bot, voice.file_id)
        prompt = ""
        if caption and asr_text:
            prompt = f"{reply_ref}{caption.strip()}\n\n{asr_text}"
        elif asr_text:
            prompt = reply_ref + asr_text
        elif caption:
            prompt = reply_ref + caption.strip()
        else:
            prompt = reply_ref
            if not prompt.strip():
                await msg.edit_text("Ошибка: не удалось получить текст из голосового сообщения.")
                return

        history = get_user_history(user_id)
        history.append({"role": "user", "content": prompt})
        add_to_history(user_id, "user", prompt)

        model = get_user_model(user_id)
        reply_text = ""
        last_sent = ""
        token_count = 0
        has_updated_once = False

        await msg.edit_text("✍️ Генерирую ответ...")

        async for chunk in ask_llm_stream_history(history, model):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0]['delta']
                content = delta.get('content', '')
                if content:
                    reply_text += content
                    token_count += 1
                    if (token_count % 8 == 0 or len(reply_text) - len(last_sent) > 30) or not has_updated_once:
                        try:
                            await msg.edit_text(reply_text + " ▍")
                            last_sent = reply_text
                            has_updated_once = True
                        except Exception as e:
                            logger.warning(f"Ошибка при edit_text: {e}\nОтвет: {reply_text}")
        safe_text = markdown_code_to_html(reply_text)
        await msg.edit_text(safe_text, parse_mode=ParseMode.HTML)
        add_to_history(user_id, "assistant", reply_text)
        logger.info(f'[Voice] Готовый ответ пользователю {user_id}: {reply_text[:100]}...')
    except Exception as e:
        logger.error(f"Ошибка при обработке голосового: {e}")
        await msg.edit_text("Ошибка при расшифровке/ответе на голосовое сообщение.")


async def ask_llm_stream_history(history, model):
    full_history = [SYSTEM_PROMPT] + history
    headers = {
        'Authorization': f'Bearer {PROXYAPI_KEY}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": model,
        "messages": full_history,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": True,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(PROXYAPI_CHAT_ENDPOINT, headers=headers, json=data, timeout=90) as response:
            if response.status != 200:
                logger.error(f"Ответ сервера: {await response.text()}")
                response.raise_for_status()
            async for line in response.content:
                if line:
                    try:
                        line = line.decode()
                        if line.startswith('data:'):
                            json_data = line[len('data:'):].strip()
                            if json_data == '[DONE]':
                                break
                            delta = json.loads(json_data)
                            yield delta
                    except Exception as e:
                        logger.warning(f"Ошибка парсинга стрима: {e}")

# --- РАСШИРЕННОЕ ПРИВЕТСТВИЕ и СПРАВКА ---
WELCOME_DESCRIPTION = (
    "👋 <b>Добро пожаловать!</b>\n"
    "Этот Telegram-бот — ваш интеллектуальный ассистент на базе самых современных LLM моделей (GPT-4, GPT-4o и др.).\n\n"
    "Возможности:\n"
    "• <b>Диалоговый ассистент</b>: отвечаем на любые вопросы, разъяснения, перевод, креатив, программирование, учеба, советы и многое другое!\n"
    "• <b>Картинки</b>: отправьте фотографию или скриншот, бот поймёт изображение (опишет, ответит по содержимому, переведёт текст на картинке и т.д.).\n"
    "• <b>Голосовые</b>: отправляйте голосовые сообщения — бот расшифрует и ответит на них как на обычный текст.\n"
    "• <b>Быстрые действия</b>: под каждым ответом доступны кнопки для объяснения, перевода, сокращения текста и др.\n"
    "• <b>Реакции</b>: можно быстро поставить реакцию-эмодзи на ответ.\n"
    "• <b>История</b>: вся ваша переписка с ботом сохраняется, можно выгрузить диалог текстом через /export\n"
    "• <b>Смена модели</b>: поддерживается переключение между моделями (GPT-4, 4o, Vision и др.)\n"
    "• <b>Рассылка и статистика</b>: админы могут видеть активность пользователей, статистику, рассылать сообщения всем.\n"
    "• <b>Автоматическая справка</b>: все команды доступны в меню (☰) или через /menu и /help\n"
    "\n<b>Просто напишите любое сообщение, отправьте фото или голос — и получите ответ!</b>"
)

HELP_INTRO = (
    "<b>🛠 Справка по возможностям бота:</b>\n"
    "Этот бот — интеллектуальный ассистент на современных LLM. Доступны текст, голос, фото, быстрые действия, экспорт истории, меню и админка.\n\n"
    "🚀 <b>Фишки и основные функции:</b>\n"
    "— Вопрос-ответ, диалог\n"
    "— Поддержка изображений (Vision)\n"
    "— Голосовые сообщения\n"
    "— История (выгрузка /export)\n"
    "— Реакции и быстрые кнопки\n"
    "— Выбор моделей /model\n"
    "— Меню (☰) с командами\n"
    "— Админ-панель и рассылка\n"
    "\n"
    "<b>Как пользоваться?</b>\n"
    "1. Просто напишите текст, отправьте голосовое или фото.\n"
    "2. Используйте меню (☰) или /menu для всех команд.\n"
    "3. Для экспорта истории — команда /export\n"
    "4. Быстрые действия и реакции — сразу под ответом."
)

HELP_COMMANDS = [
    {"cmd": "/start", "desc": "Запустить бота. Приветственное сообщение и старт работы.", "admin": False},
    {"cmd": "/menu", "desc": "Показать меню всех доступных команд.", "admin": False},
    {"cmd": "/draw", "desc": "Сгенерировать картинку по текстовому описанию", "admin": False},
    {"cmd": "/help", "desc": "Показать справку и документацию по боту.", "admin": False},
    {"cmd": "/model", "desc": "Выбрать модель LLM для ответа. Доступны только поддерживаемые модели.", "admin": False},
    {"cmd": "/reset", "desc": "Начать новый диалог, очистить историю сообщений.", "admin": False},
    {"cmd": "/export", "desc": "Выгрузить свою историю чата с ботом (в виде txt-файла).", "admin": False},
    {"cmd": "/broadcast", "desc": "Админ-команда: массовая рассылка сообщения всем пользователям.", "admin": True},
    {"cmd": "/cancel", "desc": "Отменить текущий режим рассылки (для админов).", "admin": True},
    {"cmd": "/reloadmodels", "desc": "Обновить список доступных LLM моделей (админ-команда).", "admin": True},
    {"cmd": "/admin", "desc": "Открыть админ-панель управления и статистики.", "admin": True},
    {"cmd": "/closefile", "desc": "Завершить работу с загруженным документом", "admin": False}
]

HELP_USAGE = (
    "\n\n<b>📝 Подробнее:</b>\n"
    "— Просто пишите или отправляйте фото/голосовое — бот сам всё поймёт и ответит.\n"
    "— Меню со всеми возможностями — кнопка ☰ (рядом со скрепкой).\n"
    "— История вашего диалога с ботом сохраняется и доступна для выгрузки через /export.\n"
    "— Быстрые действия и реакции под каждым ответом позволяют упростить работу.\n"
    "— Для администраторов доступны команды статистики, рассылок, настройки.\n"
)

@allowed_only
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    is_admin_user = (user_id in ADMIN_USER_IDS)
    text = HELP_INTRO + "\n\n<b>Команды:</b>\n"
    for cmd in HELP_COMMANDS:
        if cmd["admin"] and not is_admin_user:
            continue
        admin_tag = " <i>[админ]</i>" if cmd["admin"] else ""
        text += f"{cmd['cmd']}: {cmd['desc']}{admin_tag}\n"
    text += HELP_USAGE
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

async def set_bot_commands(application):
    commands = [
        BotCommand("menu", "Показать меню команд"),
        BotCommand("draw", "Сгенерировать картинку по описанию"),
        BotCommand("help", "Справка и документация"),
        BotCommand("model", "Выбрать модель LLM"),
        BotCommand("reloadmodels", "Обновить список моделей (админы)"),
        BotCommand("admin", "Админ-панель"),
        BotCommand("reset", "Новый диалог (очистить историю)"),
        BotCommand("export", "Выгрузить историю чата (txt-файл)"),
        BotCommand("broadcast", "Рассылка всем пользователям (админы)"),
        BotCommand("cancel", "Отмена режима рассылки"),
        BotCommand("start", "Начать работу с ботом"),
    ]
    await application.bot.set_my_commands(commands, scope=BotCommandScopeDefault())

def get_last_logs(n=40):
    path = "bot.log"
    if not os.path.exists(path):
        return "Лог-файл не найден."
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[-n:])[-4000:]

@admin_only
async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📈 Статистика", callback_data="admin:stats")],
        [InlineKeyboardButton("📝 Последние логи", callback_data="admin:logs")],
        [InlineKeyboardButton("⚙️ Модель по умолчанию", callback_data="admin:defmodel")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    if hasattr(update, "message") and update.message:
        await update.message.reply_text("Админ-панель:", reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    elif hasattr(update, "callback_query") and update.callback_query:
        await update.callback_query.edit_message_text("Админ-панель:", reply_markup=reply_markup, parse_mode=ParseMode.HTML)

@admin_only
async def admin_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    if data == "admin:stats":
        stats = collect_history_stats(min_messages=1)
        if not stats:
            msg = "Нет данных."
        else:
            msg = "Топ-10 активных пользователей:\n"
            top = stats[:10]
            for i, (uid, msg_count) in enumerate(top, 1):
                msg += f"{i}) <code>{uid}</code>: сообщений {msg_count}\n"
            if len(stats) > 10:
                msg += f"\nВсего активных: {len(stats)}. Для просмотра всех — нажмите кнопку ниже."
        keyboard = []
        if len(stats) > 10:
            keyboard = [
                [InlineKeyboardButton("Показать всех", callback_data="admin:allusers")]
            ]
        await query.edit_message_text(msg, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None)

    elif data == "admin:allusers":
        stats = collect_history_stats(min_messages=1)
        if not stats:
            msg = "Нет данных."
        else:
            msg = "Все активные пользователи:\n"
            for i, (uid, msg_count) in enumerate(stats, 1):
                msg += f"{i}) <code>{uid}</code>: сообщений {msg_count}\n"
        await query.edit_message_text(msg, parse_mode=ParseMode.HTML)

    elif data == "admin:logs":
        logs = get_last_logs()
        await query.edit_message_text(
            f"<b>Последние логи:</b>\n<pre>{logs}</pre>", parse_mode=ParseMode.HTML
        )
    elif data == "admin:defmodel":
        current = get_default_model()
        models = [m["id"] for m in available_models if m["id"] in MODEL_DESCRIPTIONS]
        keyboard = [
            [InlineKeyboardButton(f"{'✅ ' if m == current else ''}{m}", callback_data=f"admin:setdefmodel:{m}")]
            for m in models
        ]
        await query.edit_message_text(
            f"Модель по умолчанию: <b>{current}</b>\n\nВыбери новую:", parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    elif data.startswith("admin:setdefmodel:"):
        new_model = data.split(":")[-1]
        set_default_model(new_model)
        await query.answer(f"Установлена модель по умолчанию: {new_model}")
        await admin_command(update, context)

@admin_only
async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    set_broadcast_wait(update.effective_user.id)
    await update.message.reply_text(
        "Введите сообщение для рассылки всем пользователям (или /cancel для отмены).", parse_mode=ParseMode.HTML
    )

@admin_only
async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    waiting_admin = get_broadcast_wait()
    if waiting_admin == update.effective_user.id:
        clear_broadcast_wait()
        await update.message.reply_text("Режим рассылки отменён.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("Нет активного режима рассылки.", parse_mode=ParseMode.HTML)

@allowed_only
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not os.path.exists(_history_path(user_id)):
        text = WELCOME_DESCRIPTION
    else:
        text = (
            "Рады снова видеть вас!\n"
            "Готовы продолжать диалог или начать новый вопрос.\n\n"
            "— Для справки и описания возможностей используйте команду /help."
        )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

@allowed_only
async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "<b>Список доступных команд:</b>\n"
        "/start — начать работу с ботом\n"
        "/help — справка и документация\n"
        "/menu — это меню\n"
        "/draw — сгенерировать картинку по описанию\n"
        "/model — выбрать модель LLM\n"
        "/reset — новый диалог (очистить историю)\n"
        "/export — выгрузить историю чата (txt-файл)\n"
        "/dochelp — инструкция по работе с файлами\n"
        "/closefile — закрыть активный документ\n"
        "/reloadmodels — обновить список моделей (админы)\n"
        "/admin — админ-панель (если доступна)\n"
        "/broadcast — рассылка всем пользователям (админы)\n"
        "/cancel — отменить рассылку\n"
        "— Можно отправлять фото и голосовые сообщения.\n"
        "— Просто напишите любой вопрос!\n"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)

@allowed_only
async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    reset_history(user_id)
    await update.message.reply_text("История диалога очищена. Можно начать новый разговор!", parse_mode=ParseMode.HTML)

@allowed_only
async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    history = get_user_history(user_id)
    if not history:
        await update.message.reply_text("У вас пока нет истории сообщений.", parse_mode=ParseMode.HTML)
        return

    lines = []
    for msg in history:
        role = "🧑‍💻 Вы" if msg["role"] == "user" else "🤖 Бот"
        content = msg["content"]
        if isinstance(content, list):
            line = role + ":\n"
            for part in content:
                if part["type"] == "text":
                    line += part["text"] + "\n"
                elif part["type"] == "image_url":
                    line += "[Изображение]\n"
            lines.append(line)
        else:
            lines.append(f"{role}:\n{content}\n")

    export_text = "\n".join(lines)
    export_filename = f"chat_history_{user_id}.txt"
    with open(export_filename, "w", encoding="utf-8") as f:
        f.write(export_text)

    with open(export_filename, "rb") as f:
        await update.message.reply_document(
            f,
            filename=export_filename,
            caption="Ваша история диалога"
        )

    try:
        os.remove(export_filename)
    except Exception:
        pass

@allowed_only
async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    current = get_user_model(user_id)

    def is_relevant(m):
        return (
            m["id"] in RELEVANT_MODELS
            or m["id"].startswith("gpt-4.1")
            or m["id"].startswith("gpt-4")
            or m["id"].startswith("gpt-3.5-turbo")
        ) and not any(x in m["id"] for x in [
            "embedding", "tts", "dall-e", "whisper", "image", "moderation", "transcribe", "audio", "search", "codex", "babbage", "davinci"
        ])

    models_to_show = [
        m for m in available_models
        if is_relevant(m) and (m["id"] not in ADMIN_ONLY_MODELS or is_admin(user_id))
    ][:TOP_MODELS_LIMIT]

    if not models_to_show:
        await update.message.reply_text("Нет доступных моделей для выбора.", parse_mode=ParseMode.HTML)
        return
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅ ' if m['id']==current else ''}{m['id']}",
            callback_data=f"set_model:{m['id']}")]
        for m in models_to_show
    ]
    text = "Доступные модели:\n"
    for m in models_to_show:
        desc = MODEL_DESCRIPTIONS.get(m["id"], "")
        text += f"— {m['id']}: {desc}\n"
    text += f"\nТекущая модель: {current}"
    reply_markup = InlineKeyboardMarkup(keyboard)
    if hasattr(update, "message") and update.message:
        await update.message.reply_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    elif hasattr(update, "callback_query") and update.callback_query:
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

@allowed_only
async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    if data.startswith("set_model:"):
        model = data.split(":", 1)[1]
        def is_relevant(m):
            return (
                m["id"] in RELEVANT_MODELS
                or m["id"].startswith("gpt-4.1")
                or m["id"].startswith("gpt-4")
                or m["id"].startswith("gpt-3.5-turbo")
            ) and not any(x in m["id"] for x in [
                "embedding", "tts", "dall-e", "whisper", "image", "moderation", "transcribe", "audio", "search", "codex", "babbage", "davinci"
            ])
        allowed_ids = [m["id"] for m in available_models if is_relevant(m) and (m["id"] not in ADMIN_ONLY_MODELS or is_admin(user_id))]
        if model in allowed_ids:
            set_user_model(user_id, model)
            await query.answer(f"Модель переключена на {model}")
            await model_command(update, context)
        else:
            await query.answer("Модель не найдена или недоступна")

@admin_only
async def reloadmodels_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    res = await reload_models()
    if res:
        await update.message.reply_text("Список моделей успешно обновлён.", parse_mode=ParseMode.HTML)
    else:
        await update.message.reply_text("Не удалось обновить список моделей.", parse_mode=ParseMode.HTML)

@admin_only
async def admin_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await admin_command(update, context)

MAX_TOKENS_PER_USER = 20000

def get_tokens_stat(user_id):
    history = get_user_history(user_id)
    return sum(
        len(part.get("text", "")) if isinstance(msg["content"], list)
        else len(msg["content"])
        for msg in history
        for part in (msg["content"] if isinstance(msg["content"], list) else [msg["content"]])
    )

@allowed_only
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    prompt = update.message.text

    # --- Если есть активный документ ---
    if user_id in ACTIVE_FILE_SESSIONS:
        session = ACTIVE_FILE_SESSIONS[user_id]
        thread_id = session["thread_id"]
        assistant_id = session["assistant_id"]
        async with aiohttp.ClientSession() as session_http:
            # Добавляем новый вопрос в thread
            url_msg = f"https://api.proxyapi.ru/openai/v1/threads/{thread_id}/messages"
            payload = {"role": "user", "content": prompt}
            headers = {
                "Authorization": f"Bearer {PROXYAPI_KEY}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "assistants=v2"
            }
            async with session_http.post(url_msg, headers=headers, json=payload) as resp:
                msg_data = await resp.json()
                if "id" not in msg_data:
                    await update.message.reply_text("Ошибка добавления вопроса к документу.")
                    return
            run_id = await run_thread(session_http, assistant_id, thread_id)
            msg = await update.message.reply_text("✍️ Вопрос по документу отправлен, жду ответа...")
            status = await wait_run_complete(session_http, thread_id, run_id)
            if status != "completed":
                await msg.edit_text("❗️AI не смог обработать документ вовремя. Попробуйте позже.")
                return
            result = await get_thread_response(session_http, thread_id)
            if result and result.strip():
                safe_text = markdown_code_to_html(result)
                footer = "\n\n<i>Чтобы завершить работу с этим файлом, используйте команду /closefile</i>"
                if len(safe_text) + len(footer) > 4000:
                    safe_text = safe_text[:4000 - len(footer)]
                safe_text += footer
                if len(safe_text) > 4000:
                    preview = safe_text[:4000]
                    await msg.edit_text(
                        preview + "\n\n<code>[Ответ слишком длинный, полный ответ во вложении]</code>",
                        parse_mode=ParseMode.HTML
                    )
                    filename = "full_result.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(result)
                    with open(filename, "rb") as f:
                        await update.message.reply_document(f, filename=filename, caption="Полный ответ")
                    os.remove(filename)
                else:
                    await msg.edit_text(safe_text, parse_mode=ParseMode.HTML)
                add_to_history(user_id, "assistant", result)
            else:
                await msg.edit_text("❗️AI не смог получить ответ по вашему документу.")
        return

    # --- Обычный диалог, если нет активной file-сессии ---
    reply_ref = ""
    if update.message.reply_to_message:
        orig = update.message.reply_to_message
        reply_content = orig.text or orig.caption or "[нет текста]"
        if len(reply_content) > 500:
            reply_content = reply_content[:500] + "…"
        reply_ref = f'💬 В ответ на:\n"{reply_content}"\n\n'

    full_prompt = reply_ref + prompt

    history = get_user_history(user_id)
    history.append({"role": "user", "content": full_prompt})
    add_to_history(user_id, "user", full_prompt)

    model = get_user_model(user_id)
    msg = await update.message.reply_text("✍️ Генерирую ответ...")

    reply_text = ""
    last_sent = ""
    token_count = 0
    has_updated_once = False

    try:
        async for chunk in ask_llm_stream_history(history, model):
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0]['delta']
                content = delta.get('content', '')
                if content:
                    reply_text += content
                    token_count += 1
                    if (token_count % 8 == 0 or len(reply_text) - len(last_sent) > 30) or not has_updated_once:
                        try:
                            await msg.edit_text(reply_text + " ▍")
                            last_sent = reply_text
                            has_updated_once = True
                        except Exception as e:
                            logger.warning(f"Ошибка при edit_text: {e}\nОтвет: {reply_text}")
        safe_text = markdown_code_to_html(reply_text)
        await msg.edit_text(safe_text, parse_mode=ParseMode.HTML)
        add_to_history(user_id, "assistant", reply_text)
        logger.info(f'[Text] Готовый ответ пользователю {user_id}: {reply_text[:100]}...')
    except Exception as e:
        logger.error(f"Ошибка при edit_text: {e}\nОтвет: {reply_text}")
        await msg.edit_text("Ошибка при генерации ответа. Попробуйте позже.")

@allowed_only
async def draw_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    prompt = " ".join(context.args) if context.args else None

    if not prompt:
        await update.message.reply_text("Пожалуйста, введите описание картинки после /draw\n\nНапример:\n/draw кот учёный на велосипеде в стиле комикса")
        return

    msg = await update.message.reply_text("🎨 Генерирую изображение...")

    try:
        url = "https://api.proxyapi.ru/openai/v1/images/generations"
        headers = {"Authorization": f"Bearer {PROXYAPI_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=120) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"DALL-E ошибка: {resp.status} {text}")
                    await msg.edit_text(f"Ошибка генерации изображения: {text}")
                    return
                result = await resp.json()

        image_url = result["data"][0]["url"]
        # Отправить картинку
        await msg.delete()
        await update.message.reply_photo(photo=image_url, caption=f'Вот сгенерированное изображение по описанию:\n<code>{prompt}</code>', parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Ошибка генерации изображения: {e}")
        await msg.edit_text("Ошибка при генерации изображения. Попробуйте позже.")

async def create_vector_store_and_upload_file(session, file_bytes, file_name):
    # 1. Создать vector store
    url_vs = "https://api.proxyapi.ru/openai/v1/vector_stores"
    headers = {
        "Authorization": f"Bearer {PROXYAPI_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    async with session.post(url_vs, headers=headers, json={}, timeout=30) as resp:
        vs_data = await resp.json()
    if "id" not in vs_data:
        logger.error(f"Ошибка создания vector store: {json.dumps(vs_data, ensure_ascii=False)}")
        raise Exception(f"Ошибка создания vector store: {vs_data.get('error', vs_data)}")
    vector_store_id = vs_data["id"]

    # 2. Загружаем файл в /files корректно!
    url_files = "https://api.proxyapi.ru/openai/v1/files"
    headers_files = {
        "Authorization": f"Bearer {PROXYAPI_KEY}",
        "OpenAI-Beta": "assistants=v2"
    }
    # Получаем mimetype (по желанию)
    mime = mimetypes.guess_type(file_name)[0] or 'application/octet-stream'
    data_files = aiohttp.FormData()
    data_files.add_field(
        'file',
        BytesIO(file_bytes),            # ВАЖНО!
        filename=file_name,
        content_type=mime
    )
    data_files.add_field('purpose', 'assistants')
    async with session.post(url_files, headers=headers_files, data=data_files, timeout=120) as resp:
        file_obj = await resp.json()
    if "id" not in file_obj:
        logger.error(f"Ошибка загрузки файла: {json.dumps(file_obj, ensure_ascii=False)}")
        raise Exception(f"Ошибка загрузки файла: {file_obj.get('error', file_obj)}")
    file_id = file_obj["id"]

    # 3. Привязываем file_id к vector_store
    url_vs_files = f"https://api.proxyapi.ru/openai/v1/vector_stores/{vector_store_id}/files"
    headers_vs_files = {
        "Authorization": f"Bearer {PROXYAPI_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    payload_vs_files = {"file_id": file_id}
    async with session.post(url_vs_files, headers=headers_vs_files, json=payload_vs_files, timeout=30) as resp:
        vs_file_obj = await resp.json()
    if "id" not in vs_file_obj:
        logger.error(f"Ошибка привязки файла к vector store: {json.dumps(vs_file_obj, ensure_ascii=False)}")
        raise Exception(f"Ошибка привязки файла к vector store: {vs_file_obj.get('error', vs_file_obj)}")

    return vector_store_id, file_id

async def create_assistant(session, vector_store_id, model="gpt-4o", instructions=None):
    url = "https://api.proxyapi.ru/openai/v1/assistants"
    payload = {
        "model": model,
        "instructions": instructions or "Ты ассистент, помоги ответить на вопросы по прикреплённому файлу.",
        "tools": [{"type": "file_search"}],
        "tool_resources": {
            "file_search": {
                "vector_store_ids": [vector_store_id]
            }
        }
    }
    headers = {
        "Authorization": f"Bearer {PROXYAPI_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
        data = await resp.json()
    if "id" not in data:
        logger.error(f"[ProxyAPI] Ошибка создания ассистента: {json.dumps(data, ensure_ascii=False)}")
        raise Exception(f"Ошибка ProxyAPI (assistant): {data.get('error', data)}")
    return data["id"]

async def create_thread(session, user_message):
    url = "https://api.proxyapi.ru/openai/v1/threads"
    message = {
        "role": "user",
        "content": user_message,
    }
    payload = {
        "messages": [message]
    }
    headers = {
        "Authorization": f"Bearer {PROXYAPI_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
        data = await resp.json()
    if "id" not in data:
        logger.error(f"Ошибка при создании thread: {json.dumps(data, ensure_ascii=False)}")
        raise Exception(f"Ошибка ProxyAPI (thread): {data.get('error', data)}")
    return data["id"]

async def run_thread(session, assistant_id, thread_id):
    url = f"https://api.proxyapi.ru/openai/v1/threads/{thread_id}/runs"
    payload = {"assistant_id": assistant_id}
    headers = {
        "Authorization": f"Bearer {PROXYAPI_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
        data = await resp.json()
    if "id" not in data:
        logger.error(f"Ошибка при запуске run: {json.dumps(data, ensure_ascii=False)}")
        raise Exception(f"Ошибка ProxyAPI (run): {data.get('error', data)}")
    return data["id"]

async def wait_run_complete(session, thread_id, run_id, timeout=90):
    url = f"https://api.proxyapi.ru/openai/v1/threads/{thread_id}/runs/{run_id}"
    headers = {
        "Authorization": f"Bearer {PROXYAPI_KEY}",
        "OpenAI-Beta": "assistants=v2"
    }
    import asyncio
    for _ in range(timeout // 2):
        async with session.get(url, headers=headers, timeout=10) as resp:
            data = await resp.json()
            if "status" not in data:
                logger.error(f"Ошибка ожидания завершения run: {json.dumps(data, ensure_ascii=False)}")
                raise Exception(f"Ошибка ProxyAPI (run status): {data.get('error', data)}")
            if data.get("status") in ("completed", "failed", "cancelled", "expired"):
                return data.get("status")
        await asyncio.sleep(2)
    logger.error("Превышено время ожидания завершения run")
    return "timeout"

async def get_thread_response(session, thread_id):
    url = f"https://api.proxyapi.ru/openai/v1/threads/{thread_id}/messages"
    headers = {
        "Authorization": f"Bearer {PROXYAPI_KEY}",
        "OpenAI-Beta": "assistants=v2"
    }
    async with session.get(url, headers=headers, timeout=30) as resp:
        data = await resp.json()
    if "data" not in data:
        logger.error(f"Ошибка получения сообщений thread: {json.dumps(data, ensure_ascii=False)}")
        raise Exception(f"Ошибка ProxyAPI (get messages): {data.get('error', data)}")
    msgs = data.get("data", [])
    for msg in reversed(msgs):
        if msg.get("role") == "assistant":
            parts = msg.get("content", [])
            for part in parts:
                if part["type"] == "text":
                    return part["text"]["value"]
    logger.warning("Нет ответа ассистента в сообщениях thread")
    return None

@allowed_only
async def closefile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in ACTIVE_FILE_SESSIONS:
        ACTIVE_FILE_SESSIONS.pop(user_id, None)
        await update.message.reply_text(
            "Документ закрыт. Теперь ваши вопросы больше не будут связаны с файлом."
        )
    else:
        await update.message.reply_text(
            "Нет активного документа для закрытия."
        )

def xlsx_to_txt_bytes(file_bytes, file_name="file.xlsx", max_rows=50):
    excel_io = BytesIO(file_bytes)
    df = pd.read_excel(excel_io, engine='openpyxl')
    # Обрезаем по максимуму строк (чтобы не переполнить контекст)
    df = df.head(max_rows)
    # Форматируем как таблицу (табами или пробелами)
    txt = df.to_string(index=False)
    return txt.encode('utf-8')

SUPPORTED_EXTS = {
    ".pdf", ".txt", ".csv", ".doc", ".docx", ".ppt", ".pptx",
    ".epub", ".html", ".htm", ".md", ".json", ".rtf"
}

import os
import pandas as pd
from io import BytesIO

SUPPORTED_EXTS = {
    ".pdf", ".txt", ".doc", ".docx", ".ppt", ".pptx",
    ".epub", ".html", ".htm", ".md", ".json", ".rtf"
}

MAX_TELEGRAM_MSG_LEN = 4000  # чуть меньше лимита

def xlsx_to_txt_bytes(file_bytes, file_name="file.xlsx", max_rows=2000):
    excel_io = BytesIO(file_bytes)
    df = pd.read_excel(excel_io, engine='openpyxl')
    df = df.head(max_rows)
    txt = df.to_string(index=False)
    return txt.encode('utf-8')

@allowed_only
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import traceback

    user = update.effective_user
    user_id = user.id
    document = update.message.document
    file_name = document.file_name
    file_id = document.file_id
    ext = os.path.splitext(file_name)[-1].lower()

    tg_file = await context.bot.get_file(file_id)
    file_bytes = await tg_file.download_as_bytearray()

    if ext == ".xlsx":
        try:
            file_bytes = xlsx_to_txt_bytes(file_bytes, file_name, max_rows=3000)
            file_name = file_name.rsplit(".", 1)[0] + ".txt"
            ext = ".txt"
        except Exception as e:
            await update.message.reply_text(f"Не удалось преобразовать Excel в TXT: {e}")
            return

    if ext not in SUPPORTED_EXTS:
        await update.message.reply_text(
            f"Формат файла <b>{ext}</b> не поддерживается для поиска по содержимому. "
            "Поддерживаются форматы: PDF, DOCX, TXT, PPTX, EPUB, HTML, MD, JSON, RTF.",
            parse_mode=ParseMode.HTML
        )
        return

    if document.file_size > 40 * 1024 * 1024:  # увеличил лимит до 40 МБ
        await update.message.reply_text("Файл слишком большой (максимум 40 МБ для документов).")
        return

    msg = await update.message.reply_text(f"📄 Загружаю и анализирую документ {file_name}...")

    try:
        async with aiohttp.ClientSession() as session:
            vector_store_id, _ = await create_vector_store_and_upload_file(session, file_bytes, file_name)
            assistant_id = await create_assistant(session, vector_store_id)
            question = update.message.caption.strip() if update.message.caption else "Документ загружен."
            thread_id = await create_thread(session, question)
            ACTIVE_FILE_SESSIONS[user.id] = {
                "vector_store_id": vector_store_id,
                "assistant_id": assistant_id,
                "thread_id": thread_id,
                "file_id": file_id,
                "file_name": file_name,
            }
            if update.message.caption and update.message.caption.strip():
                # Был вопрос — сразу отвечаем
                run_id = await run_thread(session, assistant_id, thread_id)
                await msg.edit_text("✍️ Вопрос по документу отправлен, жду ответа...")
                status = await wait_run_complete(session, thread_id, run_id)
                if status != "completed":
                    await msg.edit_text("❗️AI не смог обработать документ вовремя. Попробуйте позже.")
                    return
                result = await get_thread_response(session, thread_id)
                if result and result.strip():
                    safe_text = markdown_code_to_html(result)
                    footer = "\n\n<i>Чтобы завершить работу с этим файлом, используйте команду /closefile</i>"
                    if len(safe_text) + len(footer) > 4000:
                        safe_text = safe_text[:4000 - len(footer)]
                    safe_text += footer
                    await msg.edit_text(safe_text, parse_mode=ParseMode.HTML)
                    add_to_history(user_id, "assistant", result)
                else:
                    await msg.edit_text("❗️AI не смог получить ответ по вашему документу.")
            else:
                await msg.edit_text(
                    "Документ успешно загружен. Теперь вы можете задавать к нему вопросы обычными сообщениями.\n"
                    "Чтобы закрыть файл — используйте /closefile"
                )
            logger.info(f'[Document/AssistantsAPI] Документ {file_name} для {user.id} успешно загружен.')
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Ошибка при работе с документом (AssistantsAPI): {e!r}\n{tb}")
        await msg.edit_text(
            f"Ошибка при анализе документа. Подробнее:\n<pre>{e!r}</pre>\nПопробуйте позже.",
            parse_mode=ParseMode.HTML
        )

async def dochelp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📄 Просто отправьте PDF, DOCX, TXT или другой документ в этот чат!\n"
        "Можете добавить подпись — это будет ваш вопрос к файлу.\n"
        "Например:\n"
        "— \"Найди ключевые тезисы\"\n"
        "— \"Сделай краткое резюме\"\n"
        "— \"Есть ли упоминание города Москва?\"\n"
        "Бот загрузит документ, проанализирует и ответит."
    )
    await update.message.reply_text(text)

def main():
    logger.info("Запуск Telegram бота")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(reload_models())
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("admin", admin_command_handler))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("reloadmodels", reloadmodels_command))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("export", export_command))
    app.add_handler(CommandHandler("broadcast", broadcast_command))
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CommandHandler("draw", draw_command))
    app.add_handler(CommandHandler("dochelp", dochelp_command))
    app.add_handler(CommandHandler("closefile", closefile_command))
    app.add_handler(CallbackQueryHandler(model_callback, pattern="set_model:"))
    app.add_handler(CallbackQueryHandler(admin_callback, pattern="admin:"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, broadcast_text_handler), group=0)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message), group=1)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_error_handler(error_handler)
    app.post_init = set_bot_commands
    app.run_polling()

if __name__ == '__main__':
    main()
