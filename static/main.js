const resumeEl = document.getElementById("resume");
const jobEl = document.getElementById("job");
const resumeFileEl = document.getElementById("resumeFile");
const jobFileEl = document.getElementById("jobFile");
const evaluateBtn = document.getElementById("evaluateBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const scoreEl = document.getElementById("score");
const summaryEl = document.getElementById("summary");
const strengthsEl = document.getElementById("strengths");
const gapsEl = document.getElementById("gaps");
const strengthsTableBody = document.querySelector("#strengthsTable tbody");
const weaknessesTableBody = document.querySelector("#weaknessesTable tbody");
const recommendationEl = document.getElementById("recommendation");
const overviewList = document.getElementById("overviewList");

function renderTags(container, items) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.innerHTML = "<span class=\"muted\">None</span>";
    return;
  }
  items.forEach((item) => {
    const span = document.createElement("span");
    span.className = "tag";
    span.textContent = item;
    container.appendChild(span);
  });
}

function renderTable(container, rows) {
  container.innerHTML = "";
  if (!rows || rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 2;
    td.className = "muted";
    td.textContent = "None";
    tr.appendChild(td);
    container.appendChild(tr);
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    const tdType = document.createElement("td");
    const tdArg = document.createElement("td");
    tdType.textContent = row.type || "";
    tdArg.textContent = row.argument || "";
    tr.appendChild(tdType);
    tr.appendChild(tdArg);
    container.appendChild(tr);
  });
}

function renderOverview(overview) {
  overviewList.innerHTML = "";
  if (!overview) {
    return;
  }
  const items = [
    `Name: ${overview.name || "Not specified"}`,
    `Education: ${overview.education || "Not specified"}`,
    `Experience: ${overview.experience || "Not specified"}`,
    `Experience Roles: ${overview.experience_roles || "Not specified"}`,
  ];
  if (overview.sections_found) {
    items.push(`Sections Found: ${overview.sections_found}`);
  }
  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    overviewList.appendChild(li);
  });
}
async function uploadAndFill(fileInput, targetTextArea) {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    return;
  }
  statusEl.textContent = "Extracting text...";
  evaluateBtn.disabled = true;
  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/extract-text", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.detail || "File upload failed.";
      return;
    }
    targetTextArea.value = data.text || "";
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "Upload failed. Please try again.";
  } finally {
    evaluateBtn.disabled = false;
    fileInput.value = "";
  }
}

async function evaluate() {
  const resume_text = resumeEl.value.trim();
  const job_description = jobEl.value.trim();
  if (!resume_text || !job_description) {
    statusEl.textContent = "Please provide both resume and job description.";
    return;
  }

  statusEl.textContent = "Evaluating...";
  evaluateBtn.disabled = true;
  try {
    const res = await fetch("/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resume_text, job_description }),
    });
    const data = await res.json();
    if (!res.ok) {
      statusEl.textContent = data.detail || "Request failed.";
      return;
    }
    scoreEl.textContent = `Score: ${data.score}`;
    summaryEl.textContent = data.summary || "";
    renderTags(strengthsEl, data.strengths);
    renderTags(gapsEl, data.gaps);
    renderTable(strengthsTableBody, data.strengths_table);
    renderTable(weaknessesTableBody, data.weaknesses_table);
    recommendationEl.textContent = data.summary_recommendation || "";
    renderOverview(data.overview);
    resultEl.style.display = "block";
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "Network error. Please try again.";
  } finally {
    evaluateBtn.disabled = false;
  }
}

evaluateBtn.addEventListener("click", evaluate);
clearBtn.addEventListener("click", () => {
  resumeEl.value = "";
  jobEl.value = "";
  statusEl.textContent = "";
  resultEl.style.display = "none";
});

resumeFileEl.addEventListener("change", () => uploadAndFill(resumeFileEl, resumeEl));
jobFileEl.addEventListener("change", () => uploadAndFill(jobFileEl, jobEl));
