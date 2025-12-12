import streamlit as st
import base64
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# Qwen モデル設定
# =========================
MODEL_NAME = "Qwen/Qwen3-8B"
# 軽くしたいなら:
# MODEL_NAME = "Qwen/Qwen3-4B"


@st.cache_resource
def load_model_and_tokenizer():
    """
    Qwen モデルと Tokenizer を 4bit 量子化でロード（キャッシュ付き）
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
    )
    return tokenizer, model


def summarize_qwen3(text: str, target_chars: int = 120) -> str:
    """
    Qwen3 を使って日本語要約する（英語禁止、<think> も削る）。
    """
    tokenizer, model = load_model_and_tokenizer()

    user_prompt = (
        f"次の文章を、日本語で、だいたい {target_chars} 文字程度に自然に要約してください。\n"
        f"・出力は必ず日本語のみ\n"
        f"・英語や推論過程（<think>など）を出力しない\n"
        f"・結果だけ簡潔に\n\n"
        f"【文章】\n{text}\n\n【要約】"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "あなたは日本語専用の要約アシスタントです。"
                "英語や他の言語、推論過程・自己コメントを絶対に出力してはいけません。"
                "回答は【要約】の後の文章のみとします。"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = generated[0][inputs["input_ids"].shape[1]:]
    output = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # 「【要約】」以降だけ抜き出す
    if "【要約】" in output:
        output = output.split("【要約】", 1)[-1].strip()

    # <think> や英字を念のため除去
    output = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    output = re.sub(r"[a-zA-Z]+", "", output).strip()

    return output


# =========================
# ダウンロードリンク作成
# =========================
def create_download_link(content, filename):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">要約結果をダウンロード</a>'
    return href


# =========================
# Streamlit UI 本体
# =========================
def main():
    st.title("要約")

    # セッション状態の初期化
    if "summary" not in st.session_state:
        st.session_state["summary"] = ""

    option = st.radio("選択してください", ("テキスト入力", "ファイルアップロード"))

    # ------- 入力部分 -------
    text = ""

    if option == "テキスト入力":
        # 🔹ここが「入力用のテキストボックス」
        text = st.text_area(
            "テキストを入力してください",
            key="input_text",          # key を明示して状態衝突を避ける
            height=200,
        )
    else:
        uploaded_file = st.file_uploader("ファイルを選択してください（.txt）", type=["txt"])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = file_bytes.decode("shift_jis")
                except UnicodeDecodeError:
                    st.error("UTF-8 / Shift_JIS として読み込めませんでした。")
                    text = ""
            # 読み込んだ内容を「編集できる」形で表示
            text = st.text_area(
                "読み込んだテキスト（必要なら編集してください）",
                value=text,
                key="uploaded_text",
                height=200,
            )

    # --- 文字数指定 ---
    target_chars = st.selectbox("文字数指定（目安）", [100, 200, 300], index=0)

    # --- 要約ボタン ---
    if st.button("要約する"):
        if not text.strip():
            st.warning("テキストが空です。入力するかファイルをアップロードしてください。")
        else:
            with st.spinner("Qwen3 で要約中..."):
                summary = summarize_qwen3(text, target_chars=target_chars)
                st.session_state["summary"] = summary

    # ------- 要約結果表示 -------
    st.subheader("要約結果")
    summary = st.session_state.get("summary", "")

    # こっちは「出力用」なので編集できてもいいし、
    # 編集させたくないなら disabled=True を付ける
    st.text_area(
        "要約結果",
        value=summary,
        height=200,
        key="summary_box",
    )
    st.text(f"文字数：{len(summary)}")

    if summary:
        st.markdown(
            create_download_link(summary, "summary.txt"),
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
