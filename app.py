import os
import json
import logging
import time
from functools import wraps
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
)
from flask_login import (
    LoginManager, UserMixin, login_user, login_required, logout_user, current_user
)
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, FileField, SelectField
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename
import aiohttp
import asyncio
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
import docx
import PyPDF2

# --- Настройки и переменные ---
load_dotenv()
PROXYAPI_KEY = os.getenv('PROXYAPI_KEY')
ALLOWED_USER_IDS = set(int(x) for x in os.getenv('ALLOWED_USER_IDS', '').split(',') if x)
ADMIN_USER_IDS = set(int(x) for x in os.getenv('ADMIN_USER_IDS', '').split(',') if x)
SECRET_KEY = os.getenv('SECRET_KEY', 'defaultkey')
HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOGS_DIR, "site.log")
DEFAULT_MODEL_PATH = "default_model.txt"

# --- Логирование ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s:%(filename)s:%(lineno)d: %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Flask приложение ---
app = Flask(__name__)
app.secret_key = SECRET_KEY

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Классы пользователей ---
class User(UserMixin):
    def __init__(self, user_id, admin=False):
        self.id = user_id
        self.is_admin = admin

    def get_id(self):
        return str(self.id)

# --- Flask-Login loader ---
@login_manager.user_loader
def load_user(user_id):
    try:
        user_id = int(user_id)
        return User(user_id, admin=user_id in ADMIN_USER_IDS)
    except:
        return None

# --- Авторизация (для админки) ---
class LoginForm(FlaskForm):
    user_id = StringField('User ID', validators=[DataRequired()])
    submit = SubmitField('Войти')

# --- История пользователя ---
def history_path(user_id):
    return os.path.join(HISTORY_DIR, f"history_{user_id}.json")

def get_user_history(user_id):
    path = history_path(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def add_to_history(user_id, role, content):
    path = history_path(user_id)
    history = get_user_history(user_id)
    history.append({"role": role, "content": content})
    # Ограничиваем историю (40 пар)
    if len(history) > 80:
        history = history[-80:]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка при сохранении истории: {e}")

def reset_history(user_id):
    path = history_path(user_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.error(f"Ошибка при удалении истории: {e}")

# --- Дефолтная модель ---
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

# --- Актуальные модели (будет загружаться с ProxyAPI) ---
RELEVANT_MODELS = [
    "gpt-4o", "gpt-4-vision-preview", "gpt-4.1-mini", "gpt-4.1", "gpt-4.1-2025-04-14",
    "gpt-4", "gpt-4-0613", "gpt-4-1106-preview", "gpt-4-0125-preview",
    "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview",
    "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
]
VISION_MODELS = ["gpt-4-vision-preview", "gpt-4o"]

available_models = []
user_models = {}

async def fetch_available_models():
    headers = {"Authorization": f"Bearer {PROXYAPI_KEY}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.proxyapi.ru/openai/v1/models", headers=headers, timeout=20) as resp:
                resp.raise_for_status()
                j = await resp.json()
                models = j.get("data", [])
                models = sorted(models, key=lambda m: (not m["id"].startswith("gpt-4"), m["id"]))
                return models
    except Exception as e:
        logger.warning(f"Ошибка загрузки списка моделей: {e}")
        return []

def get_user_model(user_id):
    default_model = get_default_model()
    for m in available_models:
        if m["id"] in RELEVANT_MODELS:
            return user_models.get(user_id, default_model if any(am["id"] == default_model for am in available_models) else m["id"])
    return user_models.get(user_id, default_model if any(am["id"] == default_model for am in available_models) else (available_models[0]["id"] if available_models else "gpt-4.1-mini"))

def set_user_model(user_id, model):
    user_models[user_id] = model

# --- Асинхронный LLM-запрос ---
async def ask_llm(history, model):
    SYSTEM_PROMPT = {
        "role": "system",
        "content": (
            "Ты веб-бот. Форматируй свои ответы используя <b>жирный</b>, <i>курсив</i>, <code>код</code>, <pre>код-блок</pre>."
            " Не используй markdown!"
        )
    }
    full_history = [SYSTEM_PROMPT] + history
    headers = {'Authorization': f'Bearer {PROXYAPI_KEY}', 'Content-Type': 'application/json'}
    data = {
        "model": model,
        "messages": full_history,
        "temperature": 0.7,
        "max_tokens": 1024,
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.proxyapi.ru/openai/v1/chat/completions", headers=headers, json=data, timeout=90) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"Ошибка chat-completion: {resp.status} {text}")
                raise Exception("Ошибка запроса к LLM")
            result = await resp.json()
            content = result["choices"][0]["message"]["content"]
            return content

# --- Синхронный враппер для async вызова из Flask ---
def llm_sync(history, model):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    res = loop.run_until_complete(ask_llm(history, model))
    loop.close()
    return res

# --- Flask routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    # Проверка авторизации (можно выключить, если не нужно)
    if "user_id" not in session:
        return redirect(url_for("login"))
    user_id = int(session["user_id"])
    # Загрузка истории и текущей модели
    history = get_user_history(user_id)
    model = get_user_model(user_id)
    # Отправка сообщения
    if request.method == "POST":
        user_msg = request.form.get("message", "").strip()
        if not user_msg:
            flash("Введите сообщение.")
            return redirect(url_for("index"))
        history.append({"role": "user", "content": user_msg})
        add_to_history(user_id, "user", user_msg)
        try:
            reply = llm_sync(history, model)
            add_to_history(user_id, "assistant", reply)
        except Exception as e:
            reply = "Ошибка генерации ответа."
            logger.error(f"Ошибка LLM: {e}")
        return redirect(url_for("index"))
    # Отображение страницы
    return render_template("chat.html", history=history, model=model, models=[m["id"] for m in available_models if m["id"] in RELEVANT_MODELS], is_admin=(user_id in ADMIN_USER_IDS))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        try:
            user_id = int(form.user_id.data.strip())
        except:
            flash("ID должен быть числом.")
            return render_template("login.html", form=form)
        if user_id in ALLOWED_USER_IDS or user_id in ADMIN_USER_IDS:
            login_user(User(user_id, admin=(user_id in ADMIN_USER_IDS)))
            session["user_id"] = user_id
            return redirect(url_for("index"))
        else:
            flash("Нет доступа.")
    return render_template("login.html", form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop("user_id", None)
    return redirect(url_for("login"))

@app.route('/reset')
@login_required
def reset():
    user_id = int(session["user_id"])
    reset_history(user_id)
    flash("История очищена.")
    return redirect(url_for("index"))

@app.route('/export')
@login_required
def export():
    user_id = int(session["user_id"])
    history = get_user_history(user_id)
    lines = []
    for msg in history:
        role = "🧑‍💻 Вы" if msg["role"] == "user" else "🤖 Бот"
        content = msg["content"]
        lines.append(f"{role}:\n{content}\n")
    export_text = "\n".join(lines)
    export_filename = f"chat_history_{user_id}.txt"
    with open(export_filename, "w", encoding="utf-8") as f:
        f.write(export_text)
    return send_file(export_filename, as_attachment=True)

@app.route('/set_model', methods=['POST'])
@login_required
def set_model():
    user_id = int(session["user_id"])
    model = request.form.get("model")
    set_user_model(user_id, model)
    return redirect(url_for("index"))

@app.route('/admin')
@login_required
def admin():
    if int(session["user_id"]) not in ADMIN_USER_IDS:
        flash("Нет доступа.")
        return redirect(url_for("index"))
    # Сбор статистики и логов
    users = []
    for fn in os.listdir(HISTORY_DIR):
        if fn.startswith("history_"):
            uid = fn.replace("history_", "").replace(".json", "")
            hist = get_user_history(uid)
            users.append((uid, len(hist)))
    users = sorted(users, key=lambda x: x[1], reverse=True)
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            logs = f.readlines()[-40:]
    current_model = get_default_model()
    models = [m["id"] for m in available_models if m["id"] in RELEVANT_MODELS]
    return render_template("admin.html", users=users, logs=logs, current_model=current_model, models=models)

@app.route('/admin/set_default_model', methods=['POST'])
@login_required
def admin_set_default_model():
    if int(session["user_id"]) not in ADMIN_USER_IDS:
        return "Forbidden", 403
    model = request.form.get("model")
    set_default_model(model)
    flash(f"Модель по умолчанию теперь: {model}")
    return redirect(url_for("admin"))

# Загрузка файлов (документы и картинки)
@app.route('/upload', methods=['POST'])
@login_required
def upload():
    user_id = int(session["user_id"])
    if 'file' not in request.files:
        flash("Нет файла.")
        return redirect(url_for("index"))
    file = request.files['file']
    filename = secure_filename(file.filename)
    # Сохраняем файл во временное место
    temp_path = os.path.join("history", f"tmp_{int(time.time())}_{filename}")
    file.save(temp_path)
    ext = filename.lower().split('.')[-1]
    # Документы (pdf, docx, txt)
    if ext in ['pdf', 'docx', 'txt', 'csv']:
        text = extract_text_from_file(temp_path, ext)
        user_msg = request.form.get("message", f"Покажи полный текст файла {filename}")
        history = get_user_history(user_id)
        history.append({"role": "user", "content": user_msg + f"\n\n[Документ: {filename}]\n{text[:2000]}"})
        add_to_history(user_id, "user", user_msg + f"\n\n[Документ: {filename}]\n{text[:2000]}")
        reply = llm_sync(history, get_user_model(user_id))
        add_to_history(user_id, "assistant", reply)
        os.remove(temp_path)
        return redirect(url_for("index"))
    # Картинки (jpg, png, jpeg)
    elif ext in ['jpg', 'jpeg', 'png']:
        img_base64 = encode_image_to_base64(temp_path)
        prompt = request.form.get("message", "Опиши это изображение.")
        history = get_user_history(user_id)
        multimodal_content = []
        if prompt:
            multimodal_content.append({"type": "text", "text": prompt})
        multimodal_content.append({"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_base64}"}})
        history.append({"role": "user", "content": multimodal_content})
        add_to_history(user_id, "user", multimodal_content)
        reply = llm_sync(history, get_user_model(user_id))
        add_to_history(user_id, "assistant", reply)
        os.remove(temp_path)
        return redirect(url_for("index"))
    else:
        flash("Неподдерживаемый тип файла.")
        os.remove(temp_path)
        return redirect(url_for("index"))

def extract_text_from_file(path, ext):
    try:
        if ext == "pdf":
            reader = PyPDF2.PdfReader(path)
            text = ""
            for page in reader.pages[:10]:
                text += page.extract_text() or ""
            return text
        elif ext == "docx":
            doc = docx.Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        elif ext == "txt":
            with open(path, encoding="utf-8") as f:
                return f.read()
        elif ext == "csv":
            df = pd.read_csv(path)
            return df.to_string(index=False)
    except Exception as e:
        logger.error(f"Ошибка разбора файла: {e}")
        return "[Ошибка разбора файла]"
    return ""

def encode_image_to_base64(path):
    with open(path, "rb") as f:
        img_bytes = f.read()
    b64 = base64.b64encode(img_bytes).decode()
    return b64

# --- Запуск и загрузка моделей при старте ---
def load_models():
    global available_models
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    models = loop.run_until_complete(fetch_available_models())
    available_models = models
    loop.close()

@app.before_request
def before_first_request():
    global available_models
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    models = loop.run_until_complete(fetch_available_models())
    available_models = models
    loop.close()

if __name__ == "__main__":
    # load_models()
    app.run(debug=True, host="127.0.0.1", port=5001)
