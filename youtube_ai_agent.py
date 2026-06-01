from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langdetect import detect
import re

app = FastAPI(title="YouTube AI Reply Bot")

# ------------------------------------
# MULTILINGUAL SENTIMENT MODEL
# ------------------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

# ------------------------------------
# QWEN LLM
# ------------------------------------
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct"
)

# ------------------------------------
# REQUEST MODEL
# ------------------------------------
class CommentRequest(BaseModel):
    comment: str


# ------------------------------------
# LANGUAGE DETECTION
# ------------------------------------
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"


# ------------------------------------
# SENTIMENT ANALYSIS
# ------------------------------------
def analyze_sentiment(text):

    result = sentiment_pipeline(text)[0]

    label = result["label"]

    # Cardiff labels:
    # LABEL_0 = Negative
    # LABEL_1 = Neutral
    # LABEL_2 = Positive

    if label == "LABEL_0":
        return "NEGATIVE"

    if label == "LABEL_1":
        return "NEUTRAL"

    return "POSITIVE"


# ------------------------------------
# REPLY DECISION
# ------------------------------------
def should_reply(sentiment):

    # Ignore negative comments
    if sentiment == "NEGATIVE":
        return False

    return True


# ------------------------------------
# CLEAN GENERATED TEXT
# ------------------------------------
def clean_reply(text):

    text = text.strip()

    stop_markers = [
        "User:",
        "Assistant:",
        "Comment:",
        "Language:",
        "Task:",
        "Write an email",
        "\n\n"
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0]

    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------
# GENERATE AI REPLY
# ------------------------------------
def generate_reply(comment):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly YouTube creator assistant. "
                "Reply in the same language as the comment. "
                "Keep replies short. "
                "Maximum 20 words. "
                "Do not explain. "
                "Do not use hashtags. "
                "Only output the reply."
            )
        },
        {
            "role": "user",
            "content": comment
        }
    ]

    prompt = generator.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    output = generator(
        prompt,
        max_new_tokens=25,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.2,
        return_full_text=False,
        eos_token_id=generator.tokenizer.eos_token_id
    )

    reply = output[0]["generated_text"]

    return clean_reply(reply)


# ------------------------------------
# API ENDPOINT
# ------------------------------------
@app.post("/reply")
def reply_comment(req: CommentRequest):

    comment = req.comment

    language = detect_language(comment)

    sentiment = analyze_sentiment(comment)

    # Negative comment → skip auto reply
    if not should_reply(sentiment):

        return {
            "comment": comment,
            "language": language,
            "sentiment": sentiment,
            "status": "MANUAL_REVIEW",
            "reply": None
        }

    reply = generate_reply(comment)

    return {
        "comment": comment,
        "language": language,
        "sentiment": sentiment,
        "status": "AUTO_REPLIED",
        "reply": reply
    }


# ------------------------------------
# LOCAL TESTING
# ------------------------------------
if __name__ == "__main__":

    test_comments = [
        "This is amazing work!",
        "बहुत अच्छा वीडियो 👍",
        "I didn't like this video",
        "Este video es increíble",
        "이 영상 너무 좋아요",
        "هذا فيديو رائع"
    ]

    for comment in test_comments:

        language = detect_language(comment)
        sentiment = analyze_sentiment(comment)

        print("\n-------------------------")
        print("Comment:", comment)
        print("Language:", language)
        print("Sentiment:", sentiment)

        if should_reply(sentiment):
            print("Reply:", generate_reply(comment))
        else:
            print("Reply skipped (manual review)")