---
title: Resume Evaluation By AI
emoji: 📚
colorFrom: purple
colorTo: gray
sdk: docker
pinned: false
short_description: Resume evaluation based on job description
---

## Overview

Resume evaluation based on job description. The app scores resume-job fit, extracts skills, highlights strengths and weaknesses, and provides a summary recommendation.

## Screenshot

Add a screenshot here once hosted (Hugging Face Spaces blocks binary files in git without Xet/LFS).

## Run Locally

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Open `http://localhost:7860` in your browser.

## API

- `GET /health` - Health check
- `POST /evaluate` - Evaluate resume against job description
- `POST /extract-text` - Extract text from PDF/DOCX

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## LIVE TEST URL

- 'https://meet1432-resume-evaluation-by-ai.hf.space/'
