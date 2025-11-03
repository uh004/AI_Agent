import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from models.state_types import InterviewState
from models.schemas import QSMultiOutput  
from core.resume_parser import extract_text_from_file, analyze_resume

llm = ChatOpenAI(model="gpt-4.1-mini")


def generate_question_strategy(state: InterviewState) -> InterviewState:
    summary = state.get("resume_summary", "")
    keywords = state.get("resume_keywords", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 시니어 인사담당 면접관입니다.\n"
         "아래 이력서를 기반으로 **3명의 면접관(A/B/C)**에 대해 면접 질문 전략을 만듭니다.\n\n"

         "면접관 역할:\n"
         "A = 잠재력 평가 (도전, 문제 해결, 성장 가능성)\n"
         "B = 조직 적합도 평가 (협업, 소통, 조직문화)\n"
         "C = 직무 역량 평가 (기술/업무 수행능력, 성과)\n\n"

         "각 면접관은 아래 3개 항목에 대해 질문 생성:\n"
         "1) 경력 및 경험\n"
         "2) 동기 및 커뮤니케이션\n"
         "3) 논리적 사고\n\n"

         "각 항목에 대해 반드시 다음 정보를 포함:\n"
         "- direction: 평가 의도/목적 (1~2문장)\n"
         "- examples: 실제 면접 질문 2~3개 (구체적, 맥락 기반, 정중체)\n\n"

         "출력 형식은 JSON ONLY이며 다음 스키마를 따르세요.\n"
         "{{\n"
         "  \"potential\": {{ \"experience\": ..., \"motivation\": ..., \"logic\": ... }},\n"
         "  \"organization\": {{ \"experience\": ..., \"motivation\": ..., \"logic\": ... }},\n"
         "  \"job\": {{ \"experience\": ..., \"motivation\": ..., \"logic\": ... }}\n"
         "}}\n\n"

         "예시 질문 스타일 가이드:\n"
         "- '~했던 경험이 있나요?' '~어떤 기여를 했나요?' '~어떻게 해결했나요?' 형식\n"
         "- 숫자/성과/구체적 사례 포함\n"
         "- 협업, 난관 해결, 역할 분명히 질문\n"
         "- 모호한 질문 금지\n"
         "- JSON 외 텍스트 출력 금지"
        ),
        ("human",
         "이력서 요약:\n{summary}\n\n"
         "주요 키워드:\n{keywords}\n\n"
         "**반드시 JSON만 출력하세요.**")
    ])

    chain = prompt | llm.with_structured_output(QSMultiOutput)
    result: QSMultiOutput = chain.invoke({
        "summary": summary,
        "keywords": ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
    })

    strategy_dict = {
        "경험": {
            "A": result.potential.experience.examples[0],  # 첫 질문만 사용
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



def preProcessing_Interview(file_path: str) -> InterviewState:
    """이력서 파일 입력 → 분석 → 질문전략 생성 → 첫 질문 선택"""

    # 파일 입력
    resume_text = extract_text_from_file(file_path)

    # 초기 state 설정
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
        "deep_counts" : {}
    }

    # 1) Resume 분석
    state = analyze_resume(state)

    # 2) 질문 전략 생성
    state = generate_question_strategy(state)

    # 3) 첫 질문 선택 (랜덤 면접관 & 랜덤 영역)
    strategy = state["question_strategy"]

    categories = ["경험", "동기", "논리"]
    interviewers = ["A", "B", "C"]

    cat = categories[0]   # 카테고리 고정
    iv = random.choice(interviewers)  # 면접관 랜덤

    selected_question = strategy[cat][iv]  # 지금 구조는 문자열이므로 바로 사용 가능

    # 상태 업데이트
    state["current_question"] = selected_question
    state["current_strategy"] = f"{cat}"

    return state
