import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from stop_words import get_stop_words
import os
import random
from wordcloud import WordCloud
import datetime

st.set_page_config(layout="wide")
st.title("🇷🇺 Анализ чата Telegram группы с дополнительными функциями")

# Загружаем JSON файл result.json из папки со скриптом
json_path = os.path.join(os.path.dirname(__file__), "result.json")

if not os.path.exists(json_path):
    st.error("Файл 'result.json' не найден в папке со скриптом.")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

messages = data.get("messages", [])

rows = []
for msg in messages:
    if msg.get("type") != "message":
        continue
    sender = msg.get("from", "Неизвестный")
    date = msg.get("date")
    text = msg.get("text")
    media = None
    if "photo" in msg:
        media = "фото"
    elif "file" in msg:
        media = "файл"
    elif msg.get("media_type") == "animation":
        media = "анимация"

    if isinstance(text, list):
        new_text = []
        for t in text:
            if isinstance(t, str):
                new_text.append(t)
            elif isinstance(t, dict):
                new_text.append(t.get("text", ""))
        text = "".join(new_text)

    rows.append({
        "sender": sender,
        "date": date,
        "text": text,
        "media": media,
    })

raw_df = pd.DataFrame(rows)
raw_df["date"] = pd.to_datetime(raw_df["date"])
raw_df["day"] = raw_df["date"].dt.date
raw_df["hour"] = raw_df["date"].dt.hour
raw_df["weekday"] = raw_df["date"].dt.day_name()


excluded_patterns = [
    r"ID-C",
    r"ࣧ+"
]

mask = pd.Series([False]*len(raw_df))
for pattern in excluded_patterns:
    mask |= raw_df["sender"].astype(str).str.contains(pattern, regex=True, na=False)

df = raw_df[~mask].reset_index(drop=True)

# Фильтры в сайдбаре
users = sorted(df["sender"].dropna().unique())
filtered_df = df.copy()

# Самые активные пользователи
st.subheader("👤 Самые активные пользователи")
st.bar_chart(df["sender"].value_counts().head(10))

# Сообщения по дням
st.subheader("🕒 Сообщения по дням")
daily_counts = df.groupby("day").size()
st.line_chart(daily_counts)

# Сообщения по дням недели
st.subheader("📅 Сообщения по дням недели")
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_counts = df["weekday"].value_counts().reindex(weekday_order)
plt.figure(figsize=(8, 4))
sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette="pastel")
plt.xticks(rotation=45)
plt.ylabel("Количество сообщений")
plt.title("Сообщения по дням недели")
st.pyplot(plt.gcf())

# Тепловая карта активности по 
st.subheader("⌚ Тепловая карта активности по часам")
heatmap_data = df.groupby(["hour", "sender"]).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu")
st.pyplot(plt.gcf())

# Медиа в сообщениях
st.subheader("📎 Медиа в сообщениях")
media_counts = df["media"].value_counts()
st.bar_chart(media_counts)

# Самые употребляемые слова (русский)
st.subheader("💬 Самые употребляемые слова (русский)")
all_text = " ".join(filtered_df["text"].dropna().astype(str))
words = re.findall(r"\b\w+\b", all_text.lower())
stopwords_ru = set(get_stop_words("russian"))
words = [w for w in words if w not in stopwords_ru and len(w) > 2]
common_words = Counter(words).most_common(20)
word_df = pd.DataFrame(common_words, columns=["Слово", "Количество"])
plt.figure(figsize=(8, 5))
sns.barplot(data=word_df, y="Слово", x="Количество", palette="viridis")
st.pyplot(plt.gcf())

# Средняя длина сообщения по пользователям
st.subheader("✍️ Средняя длина сообщения по пользователям")
df["msg_length"] = df["text"].fillna("").apply(len)
avg_len = df.groupby("sender")["msg_length"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 4))
sns.barplot(x=avg_len.values, y=avg_len.index, palette="mako")
plt.xlabel("Средняя длина сообщения (символы)")
st.pyplot(plt.gcf())

# Топ слов по пользователю или по всем
st.subheader("🔤 Топ слов по пользователю или по всему чату")
target_user = st.selectbox("Выберите пользователя", ["Весь чат"] + users)
if target_user == "Весь чат":
    target_df = df
else:
    target_df = df[df["sender"] == target_user]

all_text_user = " ".join(target_df["text"].dropna().astype(str))
words_user = re.findall(r"\b\w+\b", all_text_user.lower())
words_user = [w for w in words_user if w not in stopwords_ru and len(w) > 2]
common_words_user = Counter(words_user).most_common(20)
word_user_df = pd.DataFrame(common_words_user, columns=["Слово", "Количество"])
plt.figure(figsize=(8, 5))
sns.barplot(data=word_user_df, y="Слово", x="Количество", palette="coolwarm")
plt.title(f"Топ слов для {target_user}")
st.pyplot(plt.gcf())

# Распределение длины сообщений
st.subheader("📏 Распределение длины сообщений")
plt.figure(figsize=(8, 4))
sns.histplot(df["msg_length"], bins=30, kde=True)
plt.xlabel("Длина сообщения (символы)")
plt.title("Распределение длины сообщений")
st.pyplot(plt.gcf())

# Процент сообщений с медиа
st.subheader("📊 Процент сообщений с медиа")
total_msgs = len(df)
media_msgs = df["media"].notna().sum()
media_pct = media_msgs / total_msgs * 100
st.write(f"Сообщений с медиа: {media_msgs} / {total_msgs} ({media_pct:.2f}%)")

# Количество сообщений по часам
st.subheader("Количество сообщений по часам")
hourly_counts = df.groupby("hour").size()
plt.figure(figsize=(10, 4))
sns.lineplot(x=hourly_counts.index, y=hourly_counts.values)
plt.xticks(range(0, 24))
plt.xlabel("Час дня")
plt.ylabel("Количество сообщений")
plt.title("Сообщения по часам")
st.pyplot(plt.gcf())

# === Новые интересные функции ===

# Случайное сообщение за выбранную дату
st.subheader("🎲 Случайное сообщение за дату")
min_date = df["date"].min().date()
max_date = df["date"].max().date()

selected_date = st.date_input(
    "Выберите дату", 
    min_value=min_date, 
    max_value=max_date,
    value=min_date
)

msgs_on_date = df[df["date"].dt.date == selected_date]
if not msgs_on_date.empty:
    rand_msg = msgs_on_date.sample(1).iloc[0]
    st.markdown(f"**От:** {rand_msg['sender']}  \n**Время:** {rand_msg['date']}  \n\n{rand_msg['text']}")
else:
    st.write("В этот день сообщений нет.")

# Самое длинное сообщение
st.subheader("Самое длинное сообщение")
longest_msg = df.loc[df["msg_length"].idxmax()]
st.markdown(f"**От:** {longest_msg['sender']}  \n**Длина:** {longest_msg['msg_length']} символов  \n**Время:** {longest_msg['date']}  \n\n{longest_msg['text']}")

# Среднее количество сообщений в день
st.subheader("Среднее количество сообщений в день")
avg_msgs_day = daily_counts.mean()
st.write(f"{avg_msgs_day:.2f}")

# Топ 5 дней с максимальной активностью (разговоры-бурсты)
st.subheader("🔥 Топ 5 дней с максимальной активностью")
top_bursts = daily_counts.sort_values(ascending=False).head(5)
for day, count in top_bursts.items():
    st.write(f"{day}: {count} сообщений")

# Облако слов по самым частотным словам
st.subheader("☁️ Облако слов по самым частотным словам")

wordcloud_text = " ".join(words)
if wordcloud_text.strip():
    wc = WordCloud(width=800, height=400, background_color="white", stopwords=stopwords_ru).generate(wordcloud_text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt.gcf())
else:
    st.write("Нет данных для облака слов.")
