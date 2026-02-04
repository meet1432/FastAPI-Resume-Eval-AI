from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader
from docx import Document
from pydantic import BaseModel, Field
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer, pipeline
import re

app = FastAPI(title="Resume Evaluation By AI")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MATCH_MODEL_NAME = "TechWolf/JobBERT-v2"
NER_MODEL_NAME = "yashpwr/resume-ner-bert-v2"
SECTION_MODEL_NAME = "has-abi/distilBERT-finetuned-resumes-sections"

CATEGORIES = {
    "Frontend Technologies": ["html", "css", "javascript", "react", "angular", "vue", "jquery"],
    "Backend Technologies": ["python", "java", "node", "fastapi", "django", "flask", "spring", "dotnet", "c#"],
    "Data/ML": [
        "machine learning",
        "ml",
        "deep learning",
        "tensorflow",
        "pytorch",
        "scikit",
        "sklearn",
        "nlp",
        "rag",
        "vector",
        "embedding",
    ],
    "Cloud Platforms": ["aws", "azure", "gcp", "cloud"],
    "DevOps/Tools": ["docker", "kubernetes", "k8s", "ci/cd", "jenkins", "terraform", "ansible", "gitlab"],
    "Design": ["figma", "ui/ux", "graphic design", "photoshop", "illustrator", "adobe"],
    "SEO": ["seo", "search engine"],
    "Troubleshooting": ["debug", "troubleshoot", "root cause", "incident", "problem solving"],
    "Communication": ["communication", "collaboration", "stakeholder", "presentation"],
    "Leadership": ["lead", "mentored", "managed", "team lead"],
}


class ResumeEvaluationRequest(BaseModel):
    resume_text: str = Field(..., min_length=1, description="Raw resume text")
    job_description: str = Field(..., min_length=1, description="Target job description")


class ResumeEvaluationResponse(BaseModel):
    score: float = Field(..., ge=0, le=100)
    summary: str
    strengths: list[str]
    gaps: list[str]
    strengths_table: list[dict]
    weaknesses_table: list[dict]
    summary_recommendation: str
    overview: dict


@lru_cache(maxsize=1)
def get_match_components():
    tokenizer = AutoTokenizer.from_pretrained(MATCH_MODEL_NAME)
    model = AutoModel.from_pretrained(MATCH_MODEL_NAME)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def get_resume_ner():
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
    return pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


@lru_cache(maxsize=1)
def get_section_classifier():
    tokenizer = AutoTokenizer.from_pretrained(SECTION_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SECTION_MODEL_NAME)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def normalize_skill(skill: str) -> str:
    return skill.strip().lower()


def split_skill_text(text: str) -> list[str]:
    parts = re.split(r"[|,/;()\n]+", text)
    cleaned = [p.strip(" .:-").strip() for p in parts]
    return [c for c in cleaned if c]


def is_reasonable_skill(text: str) -> bool:
    if len(text) > 40:
        return False
    if len(text.split()) > 6:
        return False
    return True


def extract_entities(text: str) -> list[dict]:
    ner = get_resume_ner()
    results = ner(text)
    entities: list[dict] = []
    for item in results:
        label = (item.get("entity_group") or item.get("entity") or "").strip()
        word = (item.get("word") or item.get("text") or "").strip()
        if not label or not word:
            continue
        entities.append({"label": label, "text": word})
    return entities


def classify_sections(text: str) -> dict:
    classifier = get_section_classifier()
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    sections: dict[str, int] = {}
    for para in paragraphs[:40]:
        result = classifier(para)
        if isinstance(result, list) and result:
            label = result[0].get("label")
        else:
            label = None
        if label:
            sections[label] = sections.get(label, 0) + 1
    return sections


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(texts: list[str]) -> torch.Tensor:
    tokenizer, model = get_match_components()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs)
    embeddings = mean_pooling(output.last_hidden_state, inputs["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def extract_skills(text: str) -> list[str]:
    entities = extract_entities(text)
    skills: list[str] = []
    seen: set[str] = set()
    for ent in entities:
        if "skill" not in ent["label"].lower():
            continue
        for chunk in split_skill_text(ent["text"]):
            if not is_reasonable_skill(chunk):
                continue
            key = normalize_skill(chunk)
            if not key or key in seen:
                continue
            seen.add(key)
            skills.append(chunk)
    return skills


def format_skill_label(keyword: str) -> str:
    mapping = {
        "ml": "ML",
        "ai": "AI",
        "nlp": "NLP",
        "rag": "RAG",
        "aws": "AWS",
        "gcp": "GCP",
        "ci/cd": "CI/CD",
        "k8s": "K8s",
        "c#": "C#",
        "dotnet": ".NET",
        "javascript": "JavaScript",
        "tensorflow": "TensorFlow",
        "pytorch": "PyTorch",
        "sklearn": "scikit-learn",
        "scikit": "scikit-learn",
        "ui/ux": "UI/UX",
    }
    return mapping.get(keyword, keyword.title())


def extract_keyword_skills(text_lower: str) -> list[str]:
    keywords = []
    for items in CATEGORIES.values():
        keywords.extend(items)
    found: list[str] = []
    seen: set[str] = set()
    for kw in keywords:
        if has_keyword(text_lower, kw):
            label = format_skill_label(kw)
            key = normalize_skill(label)
            if key not in seen:
                seen.add(key)
                found.append(label)
    return found


def extract_overview(resume_text: str) -> dict:
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    name = lines[0] if lines else "Not specified"

    entities = extract_entities(resume_text)
    labels = [e["label"].lower() for e in entities]
    texts = [e["text"] for e in entities]

    name_from_ner = next((t for t, l in zip(texts, labels) if "name" in l), None)
    if name_from_ner:
        name = name_from_ner

    edu_keywords = ["bachelor", "master", "phd", "b.tech", "m.tech", "mba", "bsc", "msc", "degree", "diploma"]
    education = "Not specified"
    for line in lines:
        if any(k in line.lower() for k in edu_keywords):
            education = line
            break

    degree = next((t for t, l in zip(texts, labels) if "degree" in l), None)
    college = next((t for t, l in zip(texts, labels) if "college" in l), None)
    grad_year = next((t for t, l in zip(texts, labels) if "graduation" in l), None)
    if degree or college or grad_year:
        parts = [p for p in [degree, college, grad_year] if p]
        education = ", ".join(parts)

    years = re.findall(r"(\d+\+?)\s*years", resume_text, flags=re.IGNORECASE)
    experience = f"{years[0]} years" if years else "Not specified"
    years_from_ner = next((t for t, l in zip(texts, labels) if "experience" in l), None)
    if years_from_ner:
        experience = years_from_ner

    role_lines = [
        l
        for l in lines
        if re.search(r"developer|engineer|designer|analyst|manager|intern", l, re.IGNORECASE)
    ]
    experience_roles = "; ".join(role_lines[:3]) if role_lines else "Not specified"
    designations = [t for t, l in zip(texts, labels) if "designation" in l]
    if designations:
        experience_roles = "; ".join(designations[:3])

    return {
        "name": name,
        "education": education,
        "experience": experience,
        "experience_roles": experience_roles,
    }


def has_keyword(text_lower: str, keyword: str) -> bool:
    if " " in keyword or "/" in keyword or "-" in keyword:
        return keyword in text_lower
    return re.search(rf"\b{re.escape(keyword)}\b", text_lower) is not None


def find_keywords(text_lower: str, keywords: list[str]) -> list[str]:
    found: list[str] = []
    for kw in keywords:
        if has_keyword(text_lower, kw):
            found.append(kw)
    return found


def has_metrics(text: str) -> bool:
    patterns = [
        r"\b\d+%\b",
        r"\b\d+\s*(x|times)\b",
        r"\b(increased|decreased|reduced|improved|optimized|saved)\b",
    ]
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def extract_text_from_pdf(file: UploadFile) -> str:
    reader = PdfReader(file.file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_docx(file: UploadFile) -> str:
    doc = Document(file.file)
    return "\n".join(p.text for p in doc.paragraphs).strip()


@app.get("/", response_class=HTMLResponse)
def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="File is required")

    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file)
    else:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from the file")

    return {"text": text}


@app.post("/evaluate", response_model=ResumeEvaluationResponse)
def evaluate(payload: ResumeEvaluationRequest) -> ResumeEvaluationResponse:
    if not payload.resume_text.strip() or not payload.job_description.strip():
        raise HTTPException(status_code=400, detail="resume_text and job_description are required")

    embeddings = encode_texts([payload.resume_text, payload.job_description])
    similarity = float(
        torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()
    )
    score = max(0.0, min(100.0, (similarity + 1.0) * 50.0))

    resume_skills = extract_skills(payload.resume_text)
    job_skills = extract_skills(payload.job_description)
    resume_lower = payload.resume_text.lower()
    job_lower = payload.job_description.lower()

    resume_kw = extract_keyword_skills(resume_lower)
    job_kw = extract_keyword_skills(job_lower)

    if len(resume_skills) < 2:
        resume_skills = list(dict.fromkeys(resume_skills + resume_kw))
    if len(job_skills) < 2:
        job_skills = list(dict.fromkeys(job_skills + job_kw))
    resume_skill_set = {normalize_skill(s) for s in resume_skills}
    job_skill_set = {normalize_skill(s) for s in job_skills}

    strengths = [s for s in resume_skills if normalize_skill(s) in job_skill_set]
    gaps = [s for s in job_skills if normalize_skill(s) not in resume_skill_set]

    if not strengths and resume_kw and job_kw:
        job_kw_set = {normalize_skill(s) for s in job_kw}
        strengths = [s for s in resume_kw if normalize_skill(s) in job_kw_set]

    if not gaps and job_kw:
        resume_kw_set = {normalize_skill(s) for s in resume_kw}
        gaps = [s for s in job_kw if normalize_skill(s) not in resume_kw_set]

    summary_parts = [f"Similarity score {score:.1f}/100."]
    if strengths:
        summary_parts.append(f"Matched skills: {', '.join(strengths[:8])}.")
    if gaps:
        summary_parts.append(f"Missing skills: {', '.join(gaps[:8])}.")
    summary = " ".join(summary_parts)

    strengths_table: list[dict] = []
    weaknesses_table: list[dict] = []
    overview = extract_overview(payload.resume_text)
    section_counts = classify_sections(payload.resume_text)
    if section_counts:
        overview["sections_found"] = ", ".join(sorted(section_counts.keys()))

    if score >= 70:
        strengths_table.append(
            {
                "type": "Overall Fit",
                "argument": f"High semantic match ({score:.1f}/100) between the resume and job description.",
            }
        )
    elif score >= 50:
        strengths_table.append(
            {
                "type": "Overall Fit",
                "argument": f"Moderate semantic match ({score:.1f}/100) with room to improve alignment.",
            }
        )
    else:
        weaknesses_table.append(
            {
                "type": "Overall Fit",
                "argument": f"Low semantic match ({score:.1f}/100); resume content diverges from job needs.",
            }
        )

    if overview.get("experience") != "Not specified":
        strengths_table.append(
            {
                "type": "Experience",
                "argument": f"Experience detected: {overview['experience']}.",
            }
        )
    else:
        weaknesses_table.append(
            {
                "type": "Experience",
                "argument": "The resume does not clearly indicate total years of experience.",
            }
        )

    if overview.get("experience_roles") != "Not specified":
        strengths_table.append(
            {
                "type": "Role Alignment",
                "argument": f"Roles/Designations found: {overview['experience_roles']}.",
            }
        )

    if strengths:
        strengths_table.append(
            {
                "type": "Skill Match",
                "argument": f"Matched skills ({min(len(strengths), 6)} shown): {', '.join(strengths[:6])}.",
            }
        )

    if gaps:
        weaknesses_table.append(
            {
                "type": "Specific Technologies",
                "argument": f"Missing skills ({min(len(gaps), 8)} shown): {', '.join(gaps[:8])}.",
            }
        )

    if overview.get("education") == "Not specified":
        weaknesses_table.append(
            {
                "type": "Education",
                "argument": "No degree/education details detected in the resume text.",
            }
        )
    else:
        strengths_table.append(
            {
                "type": "Education",
                "argument": f"Education details detected: {overview['education']}.",
            }
        )

    if has_metrics(payload.resume_text):
        strengths_table.append(
            {
                "type": "Impact",
                "argument": "Quantitative impact detected (metrics/percentages/results present).",
            }
        )
    else:
        weaknesses_table.append(
            {
                "type": "Impact",
                "argument": "No measurable outcomes detected; add metrics to highlight impact.",
            }
        )

    expected_sections = ["education", "experience", "skills", "projects", "certifications", "summary"]
    present_sections = {s.lower() for s in section_counts.keys()}
    missing_sections = [s for s in expected_sections if s not in present_sections]
    if missing_sections:
        weaknesses_table.append(
            {
                "type": "Resume Sections",
                "argument": f"Missing or unclear sections: {', '.join(missing_sections)}.",
            }
        )
    else:
        strengths_table.append(
            {
                "type": "Resume Sections",
                "argument": "All key resume sections are present and clearly labeled.",
            }
        )

    for label, keywords in CATEGORIES.items():
        job_hits = find_keywords(job_lower, keywords)
        resume_hits = find_keywords(resume_lower, keywords)
        if job_hits and resume_hits:
            strengths_table.append(
                {
                    "type": label,
                    "argument": f"Resume mentions {', '.join(resume_hits[:5])} and job requires {', '.join(job_hits[:5])}.",
                }
            )
        elif job_hits and not resume_hits:
            weaknesses_table.append(
                {
                    "type": label,
                    "argument": f"Job expects {', '.join(job_hits[:6])}, but none are found in the resume.",
                }
            )

    summary_recommendation = (
        "The candidate shows relevant experience but should strengthen the resume by emphasizing missing "
        "skills and adding measurable impact. Prioritize the most critical missing skills from the job "
        "description and include concrete project results."
    )

    return ResumeEvaluationResponse(
        score=round(score, 2),
        summary=summary,
        strengths=strengths,
        gaps=gaps,
        strengths_table=strengths_table,
        weaknesses_table=weaknesses_table,
        summary_recommendation=summary_recommendation,
        overview=overview,
    )
