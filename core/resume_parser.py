import fitz
import os
from docx import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.schemas import ResumeAnalysis
from models.state_types import InterviewState

llm = ChatOpenAI(model="gpt-4.1-mini")

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

def analyze_resume(state: InterviewState) -> InterviewState:
  '''
  이력서를 분석해 핵심을 요약하고 주요 키워드를 추출
  '''
  # 1. 이력서 텍스트 가져오기
  resume_text = state["resume_text"]

  # 2. 프롬프트 구성
  prompt_template = ChatPromptTemplate.from_messages([
      ("system",
        "당신은 인사 당담자입니다. 다음 이력서 텍스트를 분석하여 핵심 요약과 주요 키워드를 도출하세요. "
        "결과는 JSON 형태로 반환하세요. "
        "1. summary: 이력서 핵심 요약 (3~5문장) "
        "2. keywords: 주요 키워드 목록 (핵심 역량, 기술, 성과, 강점 등)"),
      ("human", "분석할 이력서 텍스트:\n---\n{resume_text}")
  ])

  # 3. LLM 실행 (Pydantic 구조화 출력)
  chain = prompt_template | llm.with_structured_output(ResumeAnalysis)
  result: ResumeAnalysis = chain.invoke({"resume_text": resume_text})

  # 4. 상태 업데이트 및 반환
  return {
      **state,
      "resume_summary": result.summary,
      "resume_keywords": result.keywords,
  }
