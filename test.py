import os, time, re, base64
import streamlit as st
from huggingface_hub import InferenceClient
from httpx import ConnectTimeout, ReadTimeout, HTTPError

API_KEY = st.secrets["huggingface"]["huggingface_APIKey"]

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
HF_TOKEN = API_KEY

client = InferenceClient(
    model=MODEL_ID,
    token=HF_TOKEN
)

# =========================
# 前処理（1本目から流用）
# =========================
def remove_strings(text: str) -> str:
    pattern = re.compile(r'【.*?】|[ＲR][ー-]\d+|\n|\t|\s+|■|＊')
    return pattern.sub('', text or "")

# =========================
# 出力正規化＆文字数カウント
# =========================
def normalize_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"[a-zA-Z]+", "", text).strip()
    text = text.replace("\n", "").replace("\r", "")
    return text

def count_chars(text: str) -> int:
    return len((text or "").replace("\n", "").replace("\r", ""))

# =========================
# HF応答抽出（バージョン差異に強く）
# =========================
def _extract_message_text(choice) -> str:
    msg = getattr(choice, "message", None)
    if msg is not None and not isinstance(msg, dict):
        content = getattr(msg, "content", None) or getattr(msg, "text", None)
    elif isinstance(msg, dict):
        content = msg.get("content") or msg.get("text")
    else:
        content = None
    return str(content or "").strip()

def _call_chat(client: InferenceClient, messages, max_tokens: int, temperature: float) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if not getattr(resp, "choices", None):
                return ""
            return _extract_message_text(resp.choices[0])

        except (ConnectTimeout, ReadTimeout):
            time.sleep(2 ** attempt)

        except HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", 500)
            if status >= 500:
                time.sleep(2 ** attempt)
            else:
                raise
    return ""

# =========================
# 最終保険：句読点で自然に切る
# =========================
def smart_cut_to_sentence(text: str, target_chars: int, lookback: int = 40) -> str:
    text = normalize_output(text)
    if len(text) <= target_chars:
        return text

    head = text[:target_chars]
    puncts = {"。", "！", "？"}
    start = max(0, target_chars - lookback)

    cut_pos = -1
    for i in range(target_chars - 1, start - 1, -1):
        if head[i] in puncts:
            cut_pos = i + 1
            break

    if cut_pos != -1:
        return head[:cut_pos]

    if target_chars >= 1:
        head = head[:-1] + "。"
    return head

# =========================
# 追加：ローカルで「ちょうどN文字」に強制整形（APIコールなし）
# =========================
def force_exact_chars(text: str, target_chars: int) -> str:
    text = normalize_output(text)

    if text and not text.endswith("。"):
        text += "。"

    if len(text) > target_chars:
        text = smart_cut_to_sentence(text, target_chars)

    fillers = ["ぜひ確かめたい。", "今こそ読みたい。", "続きが気になる。", "見逃せない。"]
    i = 0
    while len(text) < target_chars:
        add = fillers[i % len(fillers)]
        remain = target_chars - len(text)
        if remain <= 0:
            break
        if len(add) <= remain:
            text += add
        else:
            piece = add[: max(0, remain - 1)] + "。"
            text += piece
        i += 1

    if len(text) > target_chars:
        text = text[: target_chars - 1] + "。"

    return text

# =========================
# 仕上げ：LLMで整形（回数を絞る）
# =========================
def finalize_with_llm(
    client: InferenceClient,
    system_prompt: str,
    ad: str,
    target_chars: int,
    max_tokens: int,
    temperature: float,
    rounds: int = 3,
) -> str:
    ad = normalize_output(ad)

    for _ in range(rounds):
        length = count_chars(ad)
        if length == target_chars and ad.endswith("。"):
            return ad

        prompt = (
            f"次の文章を、意味を変えずに新聞向けの節度ある広告文として整形してください。\n"
            f"len()で数えて「{target_chars}文字ちょうど」に必ず合わせます。\n"
            f"【絶対条件】\n"
            f"・出力は整形後の文章のみ（前置き・解説・注釈なし）\n"
            f"・必ず日本語のみ\n"
            f"・固有名詞（人名/社名/商品名/地名/番組名など）を使わない\n"
            f"・プレゼントキャンペーン、応募方法、告知だけの文章など本筋に不要な要素は入れない\n"
            f"・改行なし一段落\n"
            f"・文章を途中で切らず、必ず完結させる\n"
            f"・文末は必ず「。」で終える\n"
            f"・誇大表現や断定は避け、上品で読みやすい表現にする\n"
            f"・広告文らしく、読者への呼びかけ（例：ぜひ、今こそ、確かめたい、など）や行動を促す一言を必ず入れる\n"
            f"・文字数は必ず {target_chars} 文字ちょうど\n\n"
            f"【元の文章（現在{length}文字）】\n{ad}\n\n"
            f"【整形後（{target_chars}文字ちょうど）】"
        )

        new_ad = _call_chat(
            client,
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not new_ad:
            break

        ad = normalize_output(new_ad)

    return ad

# =========================
# 本体：広告文生成（CTA入り）
# =========================
def generate_newspaper_ad_api(
    text: str,
    target_chars: int,
    temperature: float = 0.2,
    max_adjust_rounds: int = 2,
) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN が設定されていません。")

    client_local = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=60.0)

    cleaned = remove_strings(text)
    cleaned_len = len(cleaned)

    max_tokens = int(target_chars * 2.8) + 220

    system_prompt = (
        "あなたはテレビ番組の内容をもとに、新聞掲載用の広告文を作成する専門家です。"
        "必ず日本語のみで回答し、推論過程や自己コメント、タグ、英語を一切出力してはいけません。"
    )

    user_prompt = (
        f"次の文章はテレビ番組のナレーション原稿です（約{cleaned_len}文字）。"
        f"この内容をもとに、新聞に掲載できる広告文を作成してください。\n\n"
        f"【必須ルール】\n"
        f"・出力は広告文のみ（前置き・解説・注釈なし）\n"
        f"・必ず日本語のみ\n"
        f"・固有名詞（人名/社名/商品名/地名/番組名など）は使わない（一般名詞や言い換えにする）\n"
        f"・プレゼントキャンペーン、応募方法、告知だけの文章など本筋に不要な要素は入れない\n"
        f"・不適切な表現や誇大表現を避け、新聞向けに節度ある広告文にする\n"
        f"・読者に向けた呼びかけや行動を促す一言（例：ぜひ、確かめたい、見逃せない、など）を必ず入れる\n"
        f"・改行は使わず一段落\n"
        f"・文章を途中で切らず必ず完結させ、文末は「。」で終える\n"
        f"・文字数は len()で数えて {target_chars} 文字ちょうど\n\n"
        f"【原稿】\n{cleaned}\n\n"
        f"【新聞広告文（{target_chars}文字ちょうど）】"
    )

    ad = _call_chat(
        client_local,
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not ad:
        return "広告文の生成に失敗しました。"

    ad = normalize_output(ad)

    for _ in range(max_adjust_rounds):
        length = count_chars(ad)
        if length == target_chars and ad.endswith("。"):
            return ad

        if length > target_chars:
            diff = length - target_chars
            adjust_prompt = (
                f"次の新聞広告文は len() で {length} 文字です。指定は {target_chars} 文字ちょうどで、{diff} 文字超えています。\n"
                f"意味を変えず、固有名詞を使わず、不要なキャンペーン等を入れず、"
                f"広告文らしい呼びかけ（行動を促す一言）を残したまま、"
                f"文章を途中で切らずに必ず完結させ、文末「。」で、"
                f"ちょうど {target_chars} 文字に短く修正してください。\n"
                f"【厳守】出力は修正文のみ／改行なし／前置き・解説・英語・思考過程は禁止\n\n"
                f"【元の広告文】\n{ad}\n\n"
                f"【修正後（{target_chars}文字ちょうど）】"
            )
        else:
            diff = target_chars - length
            adjust_prompt = (
                f"次の新聞広告文は len() で {length} 文字です。指定は {target_chars} 文字ちょうどで、{diff} 文字足りません。\n"
                f"意味を変えず、固有名詞を使わず、不要なキャンペーン等を入れず、"
                f"広告文らしい呼びかけ（行動を促す一言）を必ず含め、"
                f"文章を途中で切らずに必ず完結させ、文末「。」で、"
                f"ちょうど {target_chars} 文字になるよう最小限だけ補って修正してください。\n"
                f"【厳守】出力は修正文のみ／改行なし／前置き・解説・英語・思考過程は禁止\n\n"
                f"【元の広告文】\n{ad}\n\n"
                f"【修正後（{target_chars}文字ちょうど）】"
            )

        ad_new = _call_chat(
            client_local,
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": adjust_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not ad_new:
            break

        ad = normalize_output(ad_new)

    ad = finalize_with_llm(
        client=client_local,
        system_prompt=system_prompt,
        ad=ad,
        target_chars=target_chars,
        max_tokens=max_tokens,
        temperature=temperature,
        rounds=2,
    )
    ad = normalize_output(ad)

    ad = force_exact_chars(ad, target_chars)
    return ad

# =========================
# ダウンロードリンク作成
# =========================
def create_download_link(content: str, filename: str) -> str:
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">生成結果をダウンロード</a>'
    return href

# =========================
# Streamlit UI 本体
# =========================
def main():
    st.title("新聞調 広告文 生成")

    if not HF_TOKEN:
        st.error("環境変数 HUGGINGFACEHUB_API_TOKEN が設定されていません。")
        st.stop()

    if "ad" not in st.session_state:
        st.session_state["ad"] = ""

    option = st.radio("入力方法を選択してください", ("テキスト入力", "ファイルアップロード"))

    text = ""

    if option == "テキスト入力":
        text = st.text_area(
            "広告文にしたい原稿テキストを入力してください（複数行OK）",
            key="input_text",
            height=220,
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
            text = st.text_area(
                "読み込んだテキスト（必要なら編集してください）",
                value=text,
                key="uploaded_text",
                height=220,
            )

    target_chars = st.number_input(
        "目標文字数（30〜400、未指定相当は120）",
        min_value=30,
        max_value=400,
        value=120,
        step=1,
    )

    # ★ temperature はUI表示せず固定
    temperature = 0.2

    if st.button("広告文を生成する"):
        if not text.strip():
            st.warning("テキストが空です。入力するかファイルをアップロードしてください。")
        else:
            with st.spinner("生成中..."):
                try:
                    ad = generate_newspaper_ad_api(
                        text=text,
                        target_chars=int(target_chars),
                        temperature=float(temperature),
                        max_adjust_rounds=2,
                    )
                except Exception as e:
                    st.error(f"生成に失敗しました: {e}")
                    ad = ""
                st.session_state["ad"] = ad

    st.subheader("生成結果")
    ad = st.session_state.get("ad", "")

    st.text_area(
        "新聞調 広告文",
        value=ad,
        height=200,
        key="ad_box",
    )
    st.text(f"文字数：{len(ad)}")

    if ad:
        st.markdown(
            create_download_link(ad, "newspaper_ad.txt"),
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
