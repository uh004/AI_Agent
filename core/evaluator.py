from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models.schemas import FourCriteriaEval
from models.state_types import InterviewState

# LLM (모델명은 필요 시 환경변수/설정에서 주입하도록 바꿀 수 있습니다.)
llm = ChatOpenAI(model="gpt-4.1-mini")

# ===== 헬퍼: state 스키마 보정 =====
def _ensure_state_schema(state: Dict) -> Dict:
    ev = state.get("evaluation")
    if ev is None:
        ev = {}
    elif isinstance(ev, list):
        new_ev = {}
        for item in ev:
            if not isinstance(item, dict):
                continue
            strat = item.get("strategy") or item.get("분야") or "기본"
            if all(k in item for k in ("구체성", "일관성", "적합성", "논리성")):
                new_ev[strat] = {
                    "구체성": item["구체성"],
                    "일관성": item["일관성"],
                    "적합성": item["적합성"],
                    "논리성": item["논리성"],
                }
            elif "scores" in item and isinstance(item["scores"], dict):
                sc = item["scores"]
                new_ev[strat] = {
                    "구체성": sc.get("specificity") or sc.get("concreteness") or 0,
                    "일관성": sc.get("consistency") or sc.get("coherence") or 0,
                    "적합성": sc.get("fit") or sc.get("relevance") or 0,
                    "논리성": sc.get("logic") or sc.get("soundness") or 0,
                }
        ev = new_ev
    elif not isinstance(ev, dict):
        ev = {}
    conv = state.get("conversation")
    if not isinstance(conv, list):
        conv = []
    state["evaluation"] = ev
    state["conversation"] = conv
    return state

def evaluate_answer(state: Dict) -> Dict:
    """답변을 4개 기준(구체성/일관성/적합성/논리성)으로 0/1 채점하고 state.evaluation을 갱신합니다."""
    state = _ensure_state_schema(state)

    question = state.get("current_question", "")
    answer   = state.get("current_answer", "")
    strategy = state.get("current_strategy", "기본")
    resume_ctx = {
        "summary": state.get("resume_summary", ""),
        "keywords": ", ".join(state.get("resume_keywords", []))
    }

    # 1) 기본 점수 틀 생성
    evaluation: Dict[str, Dict] = state.get("evaluation", {})
    if strategy not in evaluation:
        evaluation[strategy] = {
            "구체성": 0,
            "일관성": 1,   # ← 기본 1점
            "적합성": 0,
            "논리성": 0,
            "question": question,
            "answer": answer,
            "_n": 0,
            "_sum_구체성": 0,
            "_sum_일관성": 0,
            "_sum_적합성": 0,
            "_sum_논리성": 0,
        }

    # 2) LLM 평가 (구조화 출력)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 면접관이다. 다음 4개 항목을 서로 독립적으로 0 또는 1로 채점하라. "
         "각 항목은 score(0/1)와 rationale(1~2문장)만 포함한다.\n"
         "- 구체성: 수치·사실·사례·역할·과정·결과가 명확(1) / 추상적·근거부족(0)\n"
         "- 일관성: 주장-근거-사례 흐름이 자연(1) / 모순·단절(0)\n"
         "- 적합성: 직무/질문 의도에 직접 부합(1) / 일반론·동문서답(0)\n"
         "- 논리성: 원인→행동→결과가 논리 전개(1) / 비약·누락(0)"
        ),
        # 양호 예시
        ("human",
         "전략 영역: 경험\n질문: 성능을 어떻게 개선했나요?\n"
         "답변: OpenCV로 기울기 보정과 노이즈 제거를 추가하고, 사전을 최적화하여 "
         "인식률을 86%→94%(+8%p)로 높였습니다. 저는 전처리와 사전 커스터마이징을 담당했습니다."
        ),
        ("ai",
         '{{"specificity":{{"score":1,"rationale":"역할/방법/수치 명확"}},'
         '"consistency":{{"score":1,"rationale":"개선목표-방법-결과 흐름 일관"}},'
         '"fit":{{"score":1,"rationale":"질문 의도 및 직무에 부합"}},'
         '"logic":{{"score":1,"rationale":"원인→개선→성과의 논리 전개"}}}}'
        ),
        # 부족 예시
        ("human",
         "전략 영역: 경험\n질문: 성능을 어떻게 개선했나요?\n"
         "답변: 열심히 하다 보니 잘 되었고 팀워크가 좋았습니다."
        ),
        ("ai",
         '{{"specificity":{{"score":0,"rationale":"수치/방법/역할 부재"}},'
         '"consistency":{{"score":0,"rationale":"목표-근거-사례 흐름 불명확"}},'
         '"fit":{{"score":0,"rationale":"질문과 직접 연결 부족"}},'
         '"logic":{{"score":0,"rationale":"원인/과정/결과 서술 없음"}}}}'
        ),
        # 실제 평가
        ("human",
         "컨텍스트 요약: {resume_summary}\n키워드: {resume_keywords}\n"
         "전략 영역: {strategy}\n질문: {question}\n답변: {answer}\n")
    ])

    chain = prompt | llm.with_structured_output(FourCriteriaEval)
    result: FourCriteriaEval = chain.invoke({
        "resume_summary": resume_ctx["summary"],
        "resume_keywords": resume_ctx["keywords"],
        "strategy": strategy,
        "question": question,
        "answer": answer
    })

    # 3) 간단한 일관성 검증 (휴리스틱)
    if result.consistency.score == 1:
        q_lower = question.lower()
        a_lower = answer.lower()
        contradict_pairs = [
            ("아니오", "예"), ("없다", "있다"), ("실패", "성공"), ("낮다", "높다")
        ]
        if any(p in q_lower and n in a_lower or n in q_lower and p in a_lower
               for p, n in contradict_pairs):
            result.consistency.score = 0
            result.consistency.rationale = "질문과 답변에 상반되는 표현이 있어 흐름이 단절됨"

    # 4) 점수 저장(누적)
    ev = evaluation.get(strategy, {})

    ev.setdefault("_n", 0)
    ev.setdefault("_sum_구체성", 0)
    ev.setdefault("_sum_일관성", 0)
    ev.setdefault("_sum_적합성", 0)
    ev.setdefault("_sum_논리성", 0)
    ev.setdefault("question", "")
    ev.setdefault("answer", "")

    sp = int(result.specificity.score)
    co = int(result.consistency.score)
    fi = int(result.fit.score)
    lo = int(result.logic.score)

    ev["_n"] += 1
    ev["_sum_구체성"] += sp
    ev["_sum_일관성"] += co
    ev["_sum_적합성"] += fi
    ev["_sum_논리성"] += lo

    ev["구체성"] = sp
    ev["일관성"] = co
    ev["적합성"] = fi
    ev["논리성"] = lo
    ev["question"] = question
    ev["answer"]  = answer

    evaluation[strategy] = ev

    conversation: List[Dict[str, str]] = state.get("conversation", [])
    conversation.append({"question": question, "answer": answer})

    state_update = {
        **state,
        "evaluation": evaluation,
        "conversation": conversation,
    }
    return state_update
