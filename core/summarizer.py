from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from models.state_types import InterviewState

llm = ChatOpenAI(model="gpt-4.1-mini")

#  최종평가 체인
final_eval_prompt = ChatPromptTemplate.from_template("""
당신은 기술 면접관입니다.
다음은 면접 질문별 요약과 평균 점수입니다.

평균 점수: {avg_score}/10
질문별 평가 요약:
{summary_for_llm}

이를 바탕으로 전체 인터뷰에 대한 종합평가를 3문장 이내로 작성하세요.
- 강점과 개선점을 모두 포함
- 객관적이고 자연스러운 평가 문체
""")
final_eval_chain = final_eval_prompt | llm | StrOutputParser()


#  인터뷰 요약 함수 - (출력: print는 프롬프트 백엔드 부분쪽에 출력되서 형식 바꿈)
def summarize_interview(state: dict):
    """AI 면접 피드백 보고서를 Gradio에서 표시 가능한 문자열 형태로 반환"""
    evaluations = state.get("evaluation", {})
    order = list(state.get("question_strategy", {}).keys()) if isinstance(state.get("question_strategy", {}), dict) else ["경험","동기","논리"]

    # 문자열 누적 변수
    report = ""
    report += "============================================================\n"
    report += " [AI 면접 피드백 보고서]\n"
    report += "============================================================\n"

    if not evaluations:
        report += " 평가 내역이 없습니다.\n"
        report += "============================================================\n"
        state["next_step"] = "end"
        state["final_report"] = report
        return state

    crits = ["구체성", "일관성", "적합성", "논리성"]
    total_scores = []
    summary_for_llm_lines = []

    for s in order:
        ev = evaluations.get(s, {})
        n = int(ev.get("_n", 0))

        if n > 0:
            avg = {
                "구체성": ev.get("_sum_구체성", 0) / n,
                "일관성": ev.get("_sum_일관성", 0) / n,
                "적합성": ev.get("_sum_적합성", 0) / n,
                "논리성": ev.get("_sum_논리성", 0) / n,
            }
        else:
            avg = {k: float(ev.get(k, 0)) for k in crits}
            if sum(avg.values()) == 0:
                continue
            n = 1

        overall = round((sum(avg.values()) / 4) * 10)
        total_scores.append(overall)
        strengths = [k for k, v in avg.items() if v >= 0.75]
        weaknesses = [k for k, v in avg.items() if v < 0.5]

        report += f"\n[{s}] 종합 평가 (질문 {n}개 기반)\n"
        report += "------------------------------------------------------------\n"
        report += " 항목별 평균(0~1): " + ", ".join([f"{k} {avg[k]:.2f}" for k in crits]) + "\n"
        report += f" ▶ 종합 점수: {overall}/10\n"
        if strengths:
            report += f" ▶ 강점: {', '.join(strengths)}\n"
        if weaknesses:
            report += f" ▶ 개선 필요: {', '.join(weaknesses)}\n"

        if s == "경험":
            report += " ▸ 조언: 수치·역할·방법을 STAR(상황-과제-행동-결과) 구조로 일관되게 제시하세요.\n"
        elif s == "동기":
            report += " ▸ 조언: ‘왜 우리/왜 지금/왜 나’를 2~3문장 핵심 연결로 선명하게 설명하세요.\n"
        elif s == "논리":
            report += " ▸ 조언: 원인→대안→선택 근거→결과의 인과 뼈대를 먼저 말하고 사례를 덧붙이세요.\n"

        summary_for_llm_lines.append(
            f"- {s}: 종합 {overall}/10 | 평균(구체성 {avg['구체성']:.2f}, 일관성 {avg['일관성']:.2f}, 적합성 {avg['적합성']:.2f}, 논리성 {avg['논리성']:.2f})"
        )

    if total_scores:
        avg_score = round(sum(total_scores) / len(total_scores))
        report += "\n [인터뷰 전체 종합 평가]\n"
        report += "============================================================\n"
        report += f" 평균 점수: {avg_score}/10\n\n"

        final_feedback = final_eval_chain.invoke({
            "avg_score": avg_score,
            "summary_for_llm": "\n".join(summary_for_llm_lines)
        })
        report += f" {final_feedback}\n"
    else:
        report += "\n 종합 점수를 계산할 수 있는 평가 데이터가 부족합니다.\n"

    report += "============================================================\n"
    report += " 인터뷰가 종료되었습니다. 수고하셨습니다!\n"

    # state에 저장 (Gradio에서 표시)
    state["next_step"] = "end"
    state["final_report"] = report
    return state

