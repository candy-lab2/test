import streamlit as st

# Streamlitアプリの設定
st.title("シンプルな計算アプリ")
st.write("以下のフォームに数字を入力してください。")

# ユーザー入力を受け取る
number1 = st.number_input("最初の数字を入力してください：", value=0)
number2 = st.number_input("次の数字を入力してください：", value=0)

# 計算
result_add = number1 + number2
result_subtract = number1 - number2
result_multiply = number1 * number2
result_divide = None if number2 == 0 else number1 / number2

# 結果を表示
st.write(f"足し算の結果: {result_add}")
st.write(f"引き算の結果: {result_subtract}")
st.write(f"掛け算の結果: {result_multiply}")
if result_divide is None:
    st.write("割り算は0では計算できません。")
else:
    st.write(f"割り算の結果: {result_divide}")
