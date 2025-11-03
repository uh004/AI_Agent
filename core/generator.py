from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


from models.schemas import DeepQuestion
from models.state_types import InterviewState

llm = ChatOpenAI(model="gpt-4.1-mini")

def generate_question(state: InterviewState) -> InterviewState:
    """이전 답변과 평가를 바탕으로 심화 질문을 생성합니다."""
    question = state.get("current_question", "")
    answer = state.get("current_answer", "")
    strategy = state.get("current_strategy", "")
    resume_summary = state.get("resume_summary", "")
    keywords = ", ".join(state.get("resume_keywords", []))

    ev = state.get("evaluation", {})
    last_eval = ev.get(strategy, {})

    weak_points = []
    rationale_parts = []

    if isinstance(last_eval, dict):
        for category, metrics in last_eval.items():
            if isinstance(metrics, dict):
                zero_keys = [k for k, v in metrics.items() if v == 0]
                if zero_keys:
                    weak_points.append({category: zero_keys})
                metric_text = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
                rationale_parts.append(f"{category} - {metric_text}")

    rationale = "; ".join(rationale_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "당신은 인사담당 면접관입니다. "
        "지원자의 이전 답변을 기반으로 사고력, 문제 해결 방식, 기술적 깊이를 더 파악할 수 있는 '심화 질문'을 작성하세요. "
        "조건:\n"
        "- 한 문장으로, 자연스러운 공손체로 작성\n"
        "- 이전 질문과 답변 맥락 유지\n"
        "- JSON 형태로만 출력\n"
        "- 질문 내용은 반드시 weak_points에 있는 부족한 항목을 중심으로 만드세요"),
        ("human",
        "이력서 요약: {resume_summary}\n"
        "키워드: {keywords}\n"
        "현재 전략 영역: {strategy}\n"
        "이전 질문: {question}\n"
        "지원자 답변: {answer}\n"
        "최근 평가에서 부족한 항목: {weak_points}\n\n"
        "→ 위 정보를 바탕으로 한 단계 더 깊은 심화 면접 질문을 생성하세요.")
    ])

    chain = prompt | llm.with_structured_output(DeepQuestion)
    response: DeepQuestion = chain.invoke({
        "resume_summary": resume_summary,
        "keywords": keywords,
        "strategy": strategy,
        "question": question,
        "answer": answer,
        "rationale": rationale,
        "weak_points": weak_points
    })

    return {
        **state,
        "current_question": response.question.strip(),
        "current_answer": ""
    }

