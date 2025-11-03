from typing import TypedDict, List, Dict

class InterviewState(TypedDict):
    resume_text: str
    resume_summary: str
    resume_keywords: List[str]
    question_strategy: Dict[str, Dict]
    current_question: str
    current_answer: str
    current_strategy: str
    conversation: List[Dict[str, str]]
    evaluation: List[Dict[str, str]]
    next_step: str
    deep_counts: Dict[str, Dict]
