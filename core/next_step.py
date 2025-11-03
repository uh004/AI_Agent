import random
from typing import Dict, List, Literal
from models.state_types import InterviewState

def decide_next_step(state: InterviewState) -> InterviewState:
    """
    규칙 기반 진행 제어 (evaluation = {분야:{지표:점수}}):
    - 현재 분야 점수 = evaluation[cur]의 값 합 / (지표수 * max_per_dim)
    - score >= threshold → 다음 분야(next_question) 전환 + 예시 질문 랜덤 선택
    - score < threshold → 동일 분야 심화(additional_question), deep_counts[cur] += 1
    - deep_counts[cur] > 3 → 강제 다음 분야 전환(next_question)
    - 마지막 분야에서 score >= threshold → end
    """
    # 이 함수 내부에서만 쓰는 상수
    threshold = 0.75          # 통과 기준 (75%)
    max_per_dim = 1           # 각 세부 지표 만점 (예: 0~2점)

    # 분야 순서 = question_strategy의 키 순서
    qs = state.get("question_strategy", {})
    seq = list(qs.keys())
    cur = state.get("current_strategy", (seq[0] if seq else ""))
    idx = (seq.index(cur) if cur in seq else 0)

    # 평가 딕셔너리: {분야:{지표:점수}}
    ev = state.get("evaluation", {})

    # --- 현재 분야 점수 계산(정규화 0~1) ---
    cur_field_scores = ev.get(cur, {})
    crits = ("구체성", "일관성", "적합성", "논리성")

    def _safe_int(x):
        try:
            return int(x)
        except Exception:
            return 0

    if isinstance(cur_field_scores, dict) and cur_field_scores.get("_n", 0) > 0:
    # 누적 방식(_n, _sum_*) 있으면: 키워드 내 여러 답변의 평균으로 판단
        n = int(cur_field_scores["_n"])
        avg_specificity = cur_field_scores.get("_sum_구체성", 0) / n
        avg_consistency = cur_field_scores.get("_sum_일관성", 0) / n
        avg_fit        = cur_field_scores.get("_sum_적합성", 0) / n
        avg_logic      = cur_field_scores.get("_sum_논리성", 0) / n
        score = (avg_specificity + avg_consistency + avg_fit + avg_logic) / 4.0
    else:
    # 누적 키가 없으면: 스냅샷 4항목(구/일/적/논)만으로 평균
        nums = [_safe_int(cur_field_scores.get(k, 0)) for k in crits] if isinstance(cur_field_scores, dict) else [0,0,0,0]
        score = sum(nums) / 4.0

    # --- 분야별 심화 누적 카운터 관리 ---
    deep_counts = state.get("deep_counts", {})
    cur_deep = int(deep_counts.get(cur, 0))

    # --- 분기 로직 ---
    if (score >= threshold) or (cur_deep >= 2):
        # 마지막 분야인지 확인
        if seq and idx >= len(seq) - 1:
            # 마지막 분야 통과 → 종료
            next_state = {**state, "next_step": "summarize", "deep_counts": deep_counts}
        else:
            # 다음 분야로 이동
            next_strategy = seq[idx + 1]

            next_state = {
                **state,
                "next_step": "change_strategy",
                "current_strategy": next_strategy,
                "deep_counts": deep_counts
            }

    else:
        # 기준 미달 → 같은 분야에서 추가 질문
        cur_deep += 1
        deep_counts[cur] = cur_deep
        next_state = {**state, "next_step": "generate", "deep_counts": deep_counts}
    return next_state


def change_strategy(state: InterviewState) -> InterviewState:
    """다음 분야로 전환하고 질문 하나를 선택."""
    qs = state.get("question_strategy", {})
    cur = state.get("current_strategy", "")
    block = qs.get(cur, {})
    questions = [v for v in block.values() if isinstance(v, str) and v.strip()]
    selected = random.choice(questions) if questions else "다음 분야 질문을 준비 중입니다."

    return {
        **state,
        "current_question": selected,
        "current_answer": "",
        "next_step": ""
    }


def route_next(state: InterviewState) -> Literal["generate", "change_strategy", "summarize"]:
    step = state.get("next_step", "additional_question")
    if step == "summarize" or step == "end":
        return "summarize"
    elif step == "change_strategy":
        return "change_strategy"
    else:
        return "generate"
