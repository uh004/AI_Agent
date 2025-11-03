# app.py
import os
import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import fitz  # PyMuPDF
from docx import Document
import gradio as gr

from typing import List, Dict, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# ----- 프로젝트 내부 모듈 -----
from models.state_types import InterviewState
from models.schemas import ResumeAnalysis, QSItem, QSOutput, QSMultiOutput  # 이 3개는 models/schemas.py에 정의
from core.evaluator import evaluate_answer
from core.next_step import decide_next_step, change_strategy, route_next
from core.summarizer import summarize_interview
from core.generator import generate_question


# ========== 파일 텍스트 추출 ==========
def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. PDF 또는 DOCX만 허용됩니다.")


# ========== LLM ==========
llm = ChatOpenAI(model="gpt-4.1-mini")


# ========== 이력서 분석 ==========
def analyze_resume(state: InterviewState) -> InterviewState:
    resume_text = state["resume_text"]

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 인사담당자입니다. 다음 이력서 텍스트를 분석하여 핵심 요약과 주요 키워드를 도출하세요. "
         "결과는 JSON 형태로 반환하세요. "
         "1) summary: 3~5문장 요약  2) keywords: 핵심 역량·기술·성과·강점 리스트"),
        ("human", "분석할 이력서 텍스트:\n---\n{resume_text}")
    ])

    chain = prompt_template | llm.with_structured_output(ResumeAnalysis)
    result: ResumeAnalysis = chain.invoke({"resume_text": resume_text})

    return {
        **state,
        "resume_summary": result.summary,
        "resume_keywords": result.keywords,
    }


# ========== 질문 전략 생성 ==========
def generate_question_strategy(state: InterviewState) -> InterviewState:
    summary = state.get("resume_summary", "")
    keywords = state.get("resume_keywords", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 시니어 인사담당 면접관입니다.\n"
         "아래 이력서를 기반으로 **3명의 면접관(A/B/C)**에 대해 면접 질문 전략을 JSON ONLY로 생성하세요.\n\n"
         "면접관 역할:\n"
         "A = 잠재력(도전/문제해결/성장)\nB = 조직적합(협업/소통/문화)\nC = 직무역량(기술/성과)\n\n"
         "각 면접관은 3개 항목(경험/동기/논리)에 대해\n"
         "- direction: 평가 의도(1~2문장)\n"
         "- examples: 구체적 예시 질문 2~3개\n"),
        ("human",
         "이력서 요약:\n{summary}\n\n주요 키워드:\n{keywords}\n\nJSON만 출력하세요.")
    ])

    chain = prompt | llm.with_structured_output(QSMultiOutput)
    result: QSMultiOutput = chain.invoke({
        "summary": summary,
        "keywords": ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
    })

    strategy_dict = {
        "경험": {
            "A": result.potential.experience.examples[0],
            "B": result.organization.experience.examples[0],
            "C": result.job.experience.examples[0],
        },
        "동기": {
            "A": result.potential.motivation.examples[0],
            "B": result.organization.motivation.examples[0],
            "C": result.job.motivation.examples[0],
        },
        "논리": {
            "A": result.potential.logic.examples[0],
            "B": result.organization.logic.examples[0],
            "C": result.job.logic.examples[0],
        },
    }

    state["question_strategy"] = strategy_dict
    return state


# ========== 상태 업데이트 ==========
def preProcessing_Interview(file_path: str) -> InterviewState:
    resume_text = extract_text_from_file(file_path)

    state: InterviewState = {
        "resume_text": resume_text,
        "resume_summary": "",
        "resume_keywords": [],
        "question_strategy": {},
        "current_question": "",
        "current_answer": "",
        "current_strategy": "",
        "conversation": [],
        "evaluation": [],
        "next_step": "",
        "deep_counts": {}
    }

    state = analyze_resume(state)
    state = generate_question_strategy(state)

    # 첫 질문 선택 (카테고리 고정 + 면접관 랜덤)
    categories = ["경험", "동기", "논리"]
    interviewers = ["A", "B", "C"]
    cat = categories[0]
    iv = random.choice(interviewers)
    selected_question = state["question_strategy"][cat][iv]

    state["current_question"] = selected_question
    state["current_strategy"] = cat
    return state


def update_current_answer(state: InterviewState, user_answer: str) -> InterviewState:
    return {**state, "current_answer": user_answer.strip()}


# ========== LangGraph 구성 ==========
def _update_answer_node(state: InterviewState) -> InterviewState:
    # Gradio에서 전달된 답변이 state["current_answer"]에 이미 들어 있으므로 그대로 통과
    return state


builder = StateGraph(InterviewState)
builder.add_node("update_answer", _update_answer_node)
builder.add_node("evaluate", evaluate_answer)
builder.add_node("decide", decide_next_step)
builder.add_node("generate", generate_question)
builder.add_node("change_strategy", change_strategy)
builder.add_node("summarize", summarize_interview)

builder.set_entry_point("update_answer")
builder.add_edge("update_answer", "evaluate")
builder.add_edge("evaluate", "decide")
builder.add_conditional_edges(
    "decide",
    route_next,
    {"generate": "generate", "change_strategy": "change_strategy", "summarize": "summarize"}
)
builder.add_edge("generate", END)
builder.add_edge("change_strategy", END)
builder.add_edge("summarize", END)

graph = builder.compile()


# ========== Gradio UI ==========
def initialize_state():
    return {
        "state": None,
        "interview_started": False,
        "interview_ended": False,
        "chat_history": []  # List[Tuple[str|None, str|None]]
    }


def upload_and_initialize(file_obj, session_state):
    if file_obj is None:
        return session_state, [(None, "파일을 업로드해주세요.")]

    file_path = file_obj.name
    state = preProcessing_Interview(file_path)
    session_state["state"] = state
    session_state["interview_started"] = True
    session_state["interview_ended"] = False
    session_state["chat_history"] = [(None, state["current_question"])]
    return session_state, session_state["chat_history"]


def chat_interview(user_input, session_state):
    # 이미 종료된 세션이면 그대로 출력만 유지
    if session_state.get("interview_ended"):
        return session_state, session_state["chat_history"], gr.update(value="")

    if not session_state.get("interview_started"):
        return session_state, [(None, "먼저 이력서를 업로드하고 인터뷰를 시작하세요.")], gr.update(value="")

    # 사용자 발화 추가 (튜플: (user, bot))
    session_state["chat_history"].append((user_input, None))

    # 상태에 답변 반영 → 그래프 1회 실행
    session_state["state"] = update_current_answer(session_state["state"], user_input)
    session_state["state"] = graph.invoke(session_state["state"])

    # 종료 분기
    if session_state["state"]["next_step"] == "end":
        session_state["interview_ended"] = True

        # summarize_interview가 이미 state를 end로 바꿨으므로, 최종 보고서만 메시지로 추가
        # (필요 시 state의 conversation/evaluation을 포맷팅해 요약 문자열 구성 가능)
        final_msg = "✅ 인터뷰가 종료되었습니다. 좌측 콘솔 출력(혹은 서버 로그)의 피드백 보고서를 확인하세요."
        session_state["chat_history"].append((None, final_msg))
        return session_state, session_state["chat_history"], gr.update(value="")

    # 다음 질문 진행
    next_question = session_state["state"]["current_question"]
    session_state["chat_history"].append((None, next_question))
    return session_state, session_state["chat_history"], gr.update(value="")


with gr.Blocks() as demo:
    session_state = gr.State(initialize_state())

    gr.Markdown("# 🤖 AI 면접관\n이력서를 업로드하고 인터뷰를 시작하세요!")

    with gr.Row():
        file_input = gr.File(label="이력서 업로드 (PDF 또는 DOCX)")
        upload_btn = gr.Button("인터뷰 시작")

    chatbot = gr.Chatbot(label="AI 면접관", height=480)  # 튜플 기반
    user_input = gr.Textbox(show_label=False, placeholder="답변을 입력하고 Enter를 누르세요.")

    upload_btn.click(upload_and_initialize,
                     inputs=[file_input, session_state],
                     outputs=[session_state, chatbot])

    user_input.submit(chat_interview,
                      inputs=[user_input, session_state],
                      outputs=[session_state, chatbot, user_input])

    # 입력창 자동 비우기
    user_input.submit(lambda: "", None, user_input)

# Colab/서버 모두에서 안정적으로 동작하도록 queue + 외부 링크 권장
demo.queue().launch(share=True, inline=False)
