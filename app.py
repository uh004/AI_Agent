# app.py
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import gradio as gr

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# ----- 프로젝트 내부 모듈 -----
from models.state_types import InterviewState
from core.evaluator import evaluate_answer
from core.next_step import decide_next_step, change_strategy, route_next
from core.summarizer import summarize_interview
from core.generator import generate_question
from core.question_strategy import preProcessing_Interview

# ========== LLM ==========
llm = ChatOpenAI(model="gpt-4.1-mini")

def update_current_answer(state: InterviewState, user_answer: str) -> InterviewState:
    return {**state, "current_answer": user_answer.strip()}


# ========== LangGraph 구성 ==========
# 내부 노드: 사용자 답변을 state에 반영 (임시 키 'incoming_answer' 사용)??
def _update_answer_node(state: InterviewState) -> InterviewState:
    user_answer = state.get("current_answer", "")
    new_state = update_current_answer(state, user_answer)
    if "current_answer" in new_state:
        new_state.pop("current_answer")
    return new_state



# 그래프 정의 시작
builder = StateGraph(InterviewState)

# 노드 추가
builder.add_node("update_answer", _update_answer_node)
builder.add_node("evaluate", evaluate_answer)
builder.add_node("decide", decide_next_step)
builder.add_node("generate", generate_question)
builder.add_node("change_strategy", change_strategy)
builder.add_node("summarize", summarize_interview)

# 노드 연결
builder.set_entry_point("update_answer")
builder.add_edge("update_answer", "evaluate")
builder.add_edge("evaluate", "decide")
builder.add_conditional_edges(
    "decide",
    route_next,
    {
        "generate": "generate",
        "change_strategy" : "change_strategy",
        "summarize": "summarize",
    }
)
builder.add_edge("generate", END)
builder.add_edge("change_strategy", END)
builder.add_edge("summarize", END)

# 컴파일
graph = builder.compile()



# ========== Gradio UI ==========
# 세션 상태 초기화 함수
def initialize_state():
    return {
        "state": None,
        "interview_started": False,
        "interview_ended": False,
        "chat_history": []
    }

# 파일 업로드 후 인터뷰 초기화
def upload_and_initialize(file_obj, session_state):
    if file_obj is None:
        return session_state, "파일을 업로드해주세요."

    # Gradio는 file_obj.name 이 파일 경로야
    file_path = file_obj.name

    # 인터뷰 사전 처리
    state = preProcessing_Interview(file_path)
    session_state["state"] = state
    session_state["interview_started"] = True

    # 첫 질문 저장
    first_question = state["current_question"]
    session_state["chat_history"].append(["🤖 AI 면접관", first_question])

    return session_state, session_state["chat_history"]

# 답변 처리 및 다음 질문 생성
def chat_interview(user_input, session_state):
    # (0) 인터뷰 미시작 처리
    if not session_state["interview_started"]:
        msg = "먼저 이력서를 업로드하고 인터뷰를 시작하세요."
        session_state["chat_history"].append(["AI 면접관", msg])
        return session_state, session_state["chat_history"], gr.update(value="")

    # (1) 사용자 답변 저장
    session_state["chat_history"].append(["지원자", user_input])
    state = update_current_answer(session_state["state"], user_input)

    # (2) 평가 → 결정 → 다음 행동 (단계별 실행)
    state = evaluate_answer(state)
    state = decide_next_step(state)

    next_step = state.get("next_step", "")

    if next_step == "generate":
        state = generate_question(state)
    elif next_step == "change_strategy":
        state = change_strategy(state)
    elif next_step == "summarize":
        state = summarize_interview(state)

    # (3) 상태 저장
    session_state["state"] = state

    # (4) 종료 여부 판단
    if state.get("next_step") == "end":
        session_state["interview_ended"] = True
        final_report = state.get("final_report", "인터뷰가 종료되었습니다.")
        session_state["chat_history"].append(["AI 면접관", final_report])
        return session_state, session_state["chat_history"], gr.update(value="")

    # (5) 다음 질문 제시
    next_question = state.get("current_question", "다음 질문을 준비 중입니다...")
    session_state["chat_history"].append(["AI 면접관", next_question])
    return session_state, session_state["chat_history"], gr.update(value="")

# Gradio 인터페이스 구성(**수정**)

# 테마 설정
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    font=["Noto Sans KR", "sans-serif"],
).set(
    body_background_fill="#f8fafc",
    block_background_fill="white",
    input_background_fill="#f1f5f9",
    button_primary_background_fill="#3b82f6",
    button_primary_background_fill_hover="#2563eb",
)

with gr.Blocks(theme=theme, title="AI 면접관") as demo:
    session_state = gr.State(initialize_state())

    # 제목
    gr.Markdown(
        """
        <div style="text-align: center; padding: 32px 0 16px;">
            <h1 style="font-size: 2.6em; margin: 0; color: #1e40af; font-weight: 700;">
                AI 면접관
            </h1>
            <p style="font-size: 1.1em; color: #4b5563; margin: 12px 0 0;">
                이력서를 업로드하고 KT AI/DX 직무 면접을 경험해보세요.
            </p>
        </div>
        """
    )

    # 파일 업로드 + 시작
    with gr.Row():
        file_input = gr.File(
            label="이력서 업로드 (PDF 또는 DOCX)",
            file_types=[".pdf", ".docx"],
            type="filepath"
        )
        upload_btn = gr.Button(
            "인터뷰 시작",
            variant="primary",
            size="lg"
        )

    # 상태 표시
    status_display = gr.Markdown("**상태:** 준비 중", elem_id="status")

    # 채팅창 (아이콘 있음)
    chatbot = gr.Chatbot(
        height=520,
        show_label=False,
        avatar_images=(
            "https://img.icons8.com/color/48/000000/user-male-circle.png",  # 지원자
            "https://img.icons8.com/emoji/48/000000/robot-emoji.png"        # AI
        ),
        render_markdown=True,
        bubble_full_width=False,
        latex_delimiters=[],
        elem_classes="chat"
    )

    # 입력창 (Enter 전용)
    user_input = gr.Textbox(
        show_label=False,
        placeholder="답변을 입력하고 Enter를 누르세요...",
        container=False
    )

    # === 보고서 전용 카드 영역 (숨김 → 종료 시 표시) ===
    report_card = gr.HTML(visible=False)

    # === 이벤트 ===
    def start_interview(file_obj, sess):
        if not file_obj:
            return sess, [["AI 면접관", "이력서를 먼저 업로드해주세요."]], "**상태:** 파일 없음", gr.update(visible=False)

        sess, _ = upload_and_initialize(file_obj, sess)
        first_q = sess["state"]["current_question"]
        chat = [["AI 면접관", first_q]]
        sess["chat_history"] = chat
        sess["interview_started"] = True
        return sess, chat, "**상태:** 면접 시작됨", gr.update(visible=False)

    upload_btn.click(
        start_interview,
        inputs=[file_input, session_state],
        outputs=[session_state, chatbot, status_display, report_card]
    )

    def respond(message, sess):
        if not sess.get("interview_started", False):
            return sess, sess["chat_history"] + [["AI 면접관", "인터뷰를 먼저 시작해주세요."]], "**상태:** 대기 중", gr.update(visible=False)

        # sess["chat_history"].append(["지원자", message])
        sess, new_chat, _ = chat_interview(message, sess)

        # 상태 업데이트
        cur = sess["state"].get("current_strategy", "준비")
        status = f"**상태:** {cur} 면접 중"

        # === 인터뷰 종료 시 ===
        if sess["state"].get("next_step") == "end":
            status = "**상태:** 인터뷰 종료"
            report = sess["state"].get("final_report", "인터뷰가 종료되었습니다.")

            # 보고서 카드 생성 (HTML)
            report_html = f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:16px; padding:24px; margin-top:16px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="margin:0 0 16px; color:#1e40af; font-size:1.5em;">AI 면접 피드백 보고서</h2>
                <div style="white-space:pre-wrap; font-size:0.95em; line-height:1.6; color:#374151;">
                    {report.replace('============================================================', '').strip()}
                </div>
            </div>
            """
            report_card_update = gr.update(value=report_html, visible=True)
        else:
            report_card_update = gr.update(visible=False)

        sess["chat_history"] = new_chat
        return sess, new_chat, status, report_card_update

    user_input.submit(
        respond,
        inputs=[user_input, session_state],
        outputs=[session_state, chatbot, status_display, report_card]
    ).then(
        lambda: "", outputs=user_input
    )

# === CSS: 깔끔한 말풍선 + 카드 스타일 ===
demo.css = """
#status { text-align: center; font-size: 0.9em; color: #4b5563; margin: 8px 0; }
.chat .message {
    border-radius: 14px;
    padding: 11px 15px;
    margin: 6px 0;
    max-width: 82%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.chat .message.user {
    background: #dbeafe;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
}
.chat .message.bot {
    background: #f3f4f6;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
}
"""

# === 실행 ===
demo.launch(share=True)