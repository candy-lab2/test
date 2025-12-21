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
# ✅ API呼び出し回数の上限（最大5回）
# =========================
MAX_API_CALLS = 5
_api_call_count = 0

# =========================
# 前処理
# =========================
def remove_strings(text: str) -> str:
    pattern = re.compile(r'【.*?】|[ＲR][ー-]\d+|\n|\t|\s+|■|＊')
    return pattern.sub('', text or "")

def normalize_output(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"[a-zA-Z]+", "", text)
    return text.replace("\n", "").replace("\r", "").strip()

def count_chars(text: str) -> int:
    return len((text or "").replace("\n", "").replace("\r", ""))

# =========================
# 文分割
# =========================
def split_sentences(text: str) -> list[str]:
    text = normalize_output(text)
    parts = re.split(r"(。)", text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append(parts[i] + "。")
    return sentences

# =========================
# HF 応答抽出 & 呼び出し
# =========================
def _extract_message_text(choice) -> str:
    msg = getattr(choice, "message", None)
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")

def _call_chat(client, messages, max_tokens, temperature):
    """
    ✅ 最大 MAX_API_CALLS 回までしか API を呼ばない
    ※リトライ時も「呼び出し」としてカウントします
    """
    global _api_call_count

    if _api_call_count >= MAX_API_CALLS:
        return ""

    for attempt in range(3):
        try:
            if _api_call_count >= MAX_API_CALLS:
                return ""

            _api_call_count += 1
            resp = client.chat.completions.create(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return _extract_message_text(resp.choices[0]).strip()

        except (ConnectTimeout, ReadTimeout):
            time.sleep(2 ** attempt)
        except HTTPError as e:
            if getattr(e.response, "status_code", 500) >= 500:
                time.sleep(2 ** attempt)
            else:
                raise
    return ""

# =========================
# 後半だけ再調整（切らない）
# =========================
def adjust_tail_with_llm(
    client,
    head_sentences: list[str],
    tail_sentence: str,
    target_chars: int,
    temperature: float,
) -> str:
    head = "".join(head_sentences)

    prompt = (
        f"次の新聞広告文の【後半】だけを、意味を変えずに自然に書き直してください。\n"
        f"【条件】\n"
        f"・前半は変更しない\n"
        f"・全体がだいたい {target_chars} 文字前後になるよう調整\n"
        f"・改行なし一段落\n"
        f"・固有名詞を使わない\n"
        f"・誇大表現を避ける\n"
        f"・文末は必ず「。」で終える\n\n"
        f"【前半】\n{head}\n\n"
        f"【後半（修正対象）】\n{tail_sentence}\n\n"
        f"【修正後の後半】"
    )

    new_tail = _call_chat(
        client,
        [{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=temperature,
    )

    # 呼べなかった場合は元の末尾を維持
    if not new_tail:
        return head + normalize_output(tail_sentence)

    return head + normalize_output(new_tail)

# ============================================================
# ✅ ここから「文字数厳格化」部分（追加反映）
# ============================================================

# 最終保険：句読点で自然に切る
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

# 仕上げ：完結・文末「。」・ちょうどN文字（厳格）
def finalize_with_llm(
    client: InferenceClient,
    system_prompt: str,
    ad: str,
    target_chars: int,
    max_tokens: int,
    temperature: float,
    rounds: int = 8,
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
            f"・改行なし一段落\n"
            f"・文章を途中で切らず、必ず完結させる\n"
            f"・文末は必ず「。」で終える\n"
            f"・誇大表現や断定は避け、上品で読みやすい表現にする\n"
            f"・広告文らしく、読者への呼びかけや行動を促す一言を必ず入れる\n"
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

# ============================================================
# 広告文生成（精度優先） + ✅ 文字数厳格化（厳格化は最大3回）
# ============================================================
def generate_newspaper_ad_api(text: str, target_chars: int, temperature: float = 0.2) -> str:
    global _api_call_count
    _api_call_count = 0  # ✅ 生成ごとにリセット

    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN が設定されていません。")

    client = InferenceClient(model=MODEL_ID, token=HF_TOKEN, timeout=60.0)

    cleaned = remove_strings(text)
    max_tokens = int(target_chars * 3)
    #ここに追加
    long_target = target_chars + 25

    system_prompt = (
        "あなたはテレビ番組の内容をもとに、新聞掲載用の広告文を作成する専門家です。"
        "必ず日本語のみで回答し、推論過程や自己コメント、タグ、英語を一切出力してはいけません。"
    )

    # ① 自然さ最優先で生成（元のまま）
    ad = _call_chat(
        client,
        [{"role": "user", "content": (
            f"次の原稿内容をもとに、新聞に掲載できる広告文を作成してください。\n"
            f"・日本語のみ\n"
            f"・固有名詞を使わない\n"
            f"・改行なし一段落\n"
            f"・文末は必ず「。」\n"
            f"・文字数はおよそ {target_chars} 文字前後\n"
            f"・無理な文字数合わせはしない\n\n"
            f"【原稿】\n{cleaned}\n\n【広告文】"
        )}],
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if not ad:
        return ""

    ad = normalize_output(ad)
    sentences = split_sentences(ad)

    # ② 長すぎる場合のみ最後の1文を再調整（元のまま）
    if len(sentences) >= 2 and len(ad) > target_chars + 10:
        ad = adjust_tail_with_llm(
            client,
            head_sentences=sentences[:-1],
            tail_sentence=sentences[-1],
            target_chars=target_chars,
            temperature=temperature,
        )

    # ✅ ③ len()で target_chars 文字ちょうどに寄せる（厳格化は最大3回）
    ad = normalize_output(ad)

    max_tokens_strict = int(target_chars * 2.8) + 220

    # ── 厳格化：API最大5回の配分 ──
    # 調整ループ：最大4回（=4回）
    max_adjust_rounds = 4 # 2 -> 4

    for _ in range(max_adjust_rounds):
        length = count_chars(ad)
        if length == target_chars and ad.endswith("。"):
            return ad

        if length > target_chars:
            diff = length - target_chars
            adjust_prompt = (
                f"次の新聞広告文は len() で {length} 文字です。指定は {target_chars} 文字ちょうどで、{diff} 文字超えています。\n"
                f"意味を変えず、固有名詞を使わず、広告文らしい呼びかけを残したまま、"
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
                f"意味を変えず、固有名詞を使わず、広告文らしい呼びかけを必ず含め、"
                f"文章を途中で切らずに必ず完結させ、文末「。」で、"
                f"ちょうど {target_chars} 文字になるよう最小限だけ補って修正してください。\n"
                f"【厳守】出力は修正文のみ／改行なし／前置き・解説・英語・思考過程は禁止\n\n"
                f"【元の広告文】\n{ad}\n\n"
                f"【修正後（{target_chars}文字ちょうど）】"
            )

        ad_new = _call_chat(
            client,
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": adjust_prompt}],
            max_tokens=max_tokens_strict,
            temperature=temperature,
        )
        if not ad_new:
            break
        ad = normalize_output(ad_new)

    # finalize：最大1回（=1回） → 厳格化合計 最大3回
    ad = finalize_with_llm(
        client=client,
        system_prompt=system_prompt,
        ad=ad,
        target_chars=target_chars,
        max_tokens=max_tokens_strict,
        temperature=temperature,
        rounds=1,
    )
    ad = normalize_output(ad)

    # ✅ APIなし最終保険（ズレてたら句読点で調整）
    if count_chars(ad) != target_chars or not ad.endswith("。"):
        ad = smart_cut_to_sentence(ad, target_chars)

    if count_chars(ad) == target_chars and not ad.endswith("。") and target_chars >= 1:
        ad = ad[:-1] + "。"

    return ad

# =========================
# ダウンロードリンク
# =========================
def create_download_link(content: str, filename: str):
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 広告文をダウンロード</a>'

# =========================
# Streamlit UI
# =========================
def main():
    st.title("要約")

    text = st.text_area("広告文にしたい原稿を入力してください", height=260)

    target_chars = st.number_input(
        "文字数",
        min_value=30,
        max_value=500,
        value=200,
        step=1
    )

    if st.button("広告文を生成"):
        if not text.strip():
            st.warning("原稿テキストを入力してください。")
            return

        with st.spinner("広告文を生成中..."):
            try:
                ad = generate_newspaper_ad_api(text, target_chars)

                st.text_area("生成された広告文", ad, height=200)
                st.markdown(f"文字数：{len(ad)} 文字")
                st.markdown(create_download_link(ad, "newspaper_ad.txt"), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# =========================
# 実行
# =========================
if __name__ == "__main__":
    main()
