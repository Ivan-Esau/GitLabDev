"""
MCP-Server für GitLab (fuzzy resolve_project, describe_project mit Branch-Fallback)
+ Fork-/Issue-Mirror-/Pipeline-Wait-Utilities für den Coding-Agent
+ Generisches llm_generate (kein harter Gemini-Only-Call mehr)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import os
import warnings
import base64
import difflib
import time
import json

import gitlab
from gitlab import exceptions as gl_exc
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
mcp = FastMCP("gitlab")
warnings.filterwarnings("ignore", module="gitlab")

# ---------------------- LLM Konfig ----------------------
LLM_PROVIDER     = os.getenv("LLM_PROVIDER", "gemini").lower()  # gemini | openai | custom
DEFAULT_MODEL    = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")

# Gemini
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL_TMPL  = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# OpenAI (optional)
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Custom (optional)
CUSTOM_LLM_BASE_URL = os.getenv("CUSTOM_LLM_BASE_URL", "")


# ---------------------- Helpers ----------------------
def _gitlab_client() -> gitlab.Gitlab:
    url = os.getenv("GITLAB_URL", "https://gitlab.com")
    token = os.environ["GITLAB_TOKEN"]
    gl = gitlab.Gitlab(url=url, private_token=token, keep_base_url=True)
    gl.auth()
    return gl

def _try_decode_b64(b: str) -> str:
    try:
        return base64.b64decode(b).decode("utf-8", errors="replace")
    except Exception:
        return b

def _best_fuzzy_match(query: str, candidates: List[gitlab.v4.objects.Project]) -> List[Dict[str, Any]]:
    q = query.lower()
    rows: List[Dict[str, Any]] = []
    for p in candidates:
        name_full = (p.name_with_namespace or "").lower()
        path_full = (p.path_with_namespace or "").lower()
        score = max(
            difflib.SequenceMatcher(None, q, name_full).ratio(),
            difflib.SequenceMatcher(None, q, path_full).ratio(),
        )
        rows.append({
            "id": p.id,
            "name": p.name_with_namespace,
            "path": p.path_with_namespace,
            "web_url": p.web_url,
            "score": round(score, 4),
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows

def _project_from_ref(gl: gitlab.Gitlab, project_ref: Union[str, int]):
    try:
        return gl.projects.get(int(project_ref))
    except Exception:
        pass
    candidates = gl.projects.list(search=str(project_ref), per_page=50, get_all=False)
    if not candidates:
        raise ValueError(f"Projekt '{project_ref}' nicht gefunden")
    ranked = _best_fuzzy_match(str(project_ref), candidates)
    return gl.projects.get(ranked[0]["id"])


# ---------------------- GitLab Tools ----------------------
@mcp.tool()
def whoami() -> Dict[str, Any]:
    gl = _gitlab_client()
    u = gl.user
    return {"id": u.id, "username": u.username, "name": u.name, "web_url": u.web_url}

@mcp.tool()
def resolve_project(project_ref: str, limit: int = 10) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    try:
        pid = int(project_ref)
        p = gl.projects.get(pid)
        return [{"id": p.id, "name": p.name_with_namespace, "path": p.path_with_namespace,
                 "web_url": p.web_url, "score": 1.0}]
    except Exception:
        pass
    results = gl.projects.list(search=project_ref, per_page=limit, get_all=False)
    if not results:
        return []
    ranked = _best_fuzzy_match(project_ref, results)
    return ranked[:limit]

@mcp.tool()
def describe_project(project_ref: str, ref: str = None, tree_limit: int = 200) -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = None
    try:
        proj = gl.projects.get(int(project_ref))
    except Exception:
        ranked = _best_fuzzy_match(project_ref, gl.projects.list(search=project_ref, per_page=50))
        if ranked:
            proj = gl.projects.get(ranked[0]["id"])
        else:
            raise ValueError(f"Projekt '{project_ref}' nicht gefunden")

    branch = ref or getattr(proj, "default_branch", None) or "master"

    # README
    readme_text = None
    for cand in ["README.md", "README", "readme.md", "Readme.md"]:
        try:
            f = proj.files.get(file_path=cand, ref=branch)
            readme_text = f.decode()
            break
        except Exception:
            continue

    tree = proj.repository_tree(path=None, ref=branch, per_page=tree_limit, recursive=True)

    return {
        "id": proj.id,
        "name": proj.name_with_namespace,
        "path": proj.path_with_namespace,
        "web_url": proj.web_url,
        "default_branch": getattr(proj, "default_branch", None),
        "readme": readme_text,
        "root_tree": tree,
    }

@mcp.tool()
def list_projects(search: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    projs = gl.projects.list(search=search, per_page=limit, get_all=False)
    return [{"id": p.id, "name": p.name_with_namespace, "path": p.path_with_namespace, "web_url": p.web_url} for p in projs]

@mcp.tool()
def list_issues(project_ref: str, state: str = "opened", limit: int = 20) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issues = proj.issues.list(state=state, per_page=limit, get_all=False)
    return [{"iid": i.iid, "title": i.title, "state": i.state, "web_url": i.web_url} for i in issues]

@mcp.tool()
def create_issue(project_ref: str, title: str, description: str = "") -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    i = proj.issues.create({"title": title, "description": description})
    return {"iid": i.iid, "web_url": i.web_url, "title": i.title}

@mcp.tool()
def add_comment(project_ref: str, issue_iid: int, body: str) -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issue = proj.issues.get(issue_iid, iid=True)
    note = issue.notes.create({"body": body})
    return {"note_id": note.id, "body": note.body}

@mcp.tool()
def list_merge_requests(project_ref: str, state: str = "opened", limit: int = 20) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    mrs = proj.mergerequests.list(state=state, per_page=limit, get_all=False)
    return [{
        "iid": mr.iid,
        "title": mr.title,
        "state": mr.state,
        "source_branch": mr.source_branch,
        "target_branch": mr.target_branch,
        "web_url": mr.web_url,
    } for mr in mrs]

@mcp.tool()
def create_merge_request(
    project_ref: str,
    source_branch: str,
    target_branch: str,
    title: str,
    description: str = "",
    remove_source_branch: bool = False,
) -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    mr = proj.mergerequests.create({
        "source_branch": source_branch,
        "target_branch": target_branch,
        "title": title,
        "description": description,
        "remove_source_branch": remove_source_branch,
    })
    return {"iid": mr.iid, "web_url": mr.web_url, "title": mr.title}

@mcp.tool()
def merge_merge_request(
    project_ref: str,
    mr_iid: int,
    merge_when_pipeline_succeeds: bool = False,
    squash: bool = False,
    should_remove_source_branch: bool = False,
) -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    mr = proj.mergerequests.get(mr_iid, iid=True)
    mr.merge({
        "merge_when_pipeline_succeeds": merge_when_pipeline_succeeds,
        "squash": squash,
        "should_remove_source_branch": should_remove_source_branch,
    })
    mr.refresh()
    return {"iid": mr.iid, "state": mr.state, "merged_at": getattr(mr, "merged_at", None)}

@mcp.tool()
def list_pipelines(project_ref: str, status: Optional[str] = None, ref: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    params: Dict[str, Any] = {"per_page": limit}
    if status:
        params["status"] = status
    if ref:
        params["ref"] = ref
    pipes = proj.pipelines.list(get_all=False, **params)
    return [{"id": p.id, "status": p.status, "ref": p.ref, "web_url": p.web_url} for p in pipes]

@mcp.tool()
def trigger_pipeline(project_ref: str, ref: str, variables: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    vars_list = [{"key": k, "value": v} for k, v in (variables or {}).items()]
    p = proj.pipelines.create({"ref": ref, "variables": vars_list} if vars_list else {"ref": ref})
    return {"id": p.id, "status": p.status, "web_url": p.web_url}

@mcp.tool()
def list_jobs(project_ref: str, pipeline_id: Optional[int] = None, scope: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    if pipeline_id:
        pipe = proj.pipelines.get(pipeline_id)
        jobs = pipe.jobs.list(per_page=limit, get_all=False, scope=scope)
    else:
        jobs = proj.jobs.list(per_page=limit, get_all=False, scope=scope)
    return [{"id": j.id, "name": j.name, "status": j.status, "stage": j.stage, "web_url": j.web_url} for j in jobs]

@mcp.tool()
def list_commits(
    project_ref: str,
    ref_name: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    params: Dict[str, Any] = {"per_page": limit}
    if ref_name:
        params["ref_name"] = ref_name
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    commits = proj.commits.list(get_all=False, **params)
    return [{"id": c.id, "short_id": c.short_id, "title": c.title, "author_name": c.author_name, "created_at": c.created_at, "web_url": c.web_url} for c in commits]

@mcp.tool()
def list_branches(project_ref: str, limit: int = 50) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    branches = proj.branches.list(per_page=limit, get_all=False)
    return [{"name": b.name, "default": getattr(b, "default", False), "protected": getattr(b, "protected", False)} for b in branches]

@mcp.tool()
def create_branch(project_ref: str, branch: str, ref: str = "main") -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    b = proj.branches.create({"branch": branch, "ref": ref})
    return {"name": b.name, "ref": ref}

@mcp.tool()
def get_file(project_ref: str, file_path: str, ref: str = None, decode: bool = True) -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    branch = ref or getattr(proj, "default_branch", None) or "master"
    f = proj.files.get(file_path=file_path, ref=branch)
    content_b64 = f.content
    out: Dict[str, Any] = {"file_path": file_path, "ref": branch, "content_base64": content_b64}
    if decode:
        try:
            out["content"] = f.decode()
        except Exception:
            out["content"] = _try_decode_b64(content_b64)
    return out

@mcp.tool()
def upsert_file(
    project_ref: str,
    file_path: str,
    branch: str,
    content: str,
    commit_message: str,
) -> Dict[str, Any]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    try:
        f = proj.files.get(file_path=file_path, ref=branch)
        f.content = content
        f.save(branch=branch, commit_message=commit_message)
        action = "updated"
    except gl_exc.GitlabGetError:
        proj.files.create({
            "file_path": file_path,
            "branch": branch,
            "content": content,
            "commit_message": commit_message,
        })
        action = "created"
    return {"file_path": file_path, "branch": branch, "action": action}

@mcp.tool()
def list_repo_tree(
    project_ref: str,
    path: str | None = None,
    ref: str = None,
    recursive: bool = False,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    branch = ref or getattr(proj, "default_branch", None) or "master"
    items = proj.repository_tree(path=path, ref=branch, per_page=limit, recursive=recursive)
    return items

@mcp.tool()
def fork_project(project_ref: str | int, namespace: Optional[str] = None) -> Dict[str, Any]:
    """
    Forkt ein Projekt in den eigenen Account oder in einen angegebenen Namespace.
    """
    gl = _gitlab_client()
    src = _project_from_ref(gl, project_ref)
    data: Dict[str, Any] = {}
    if namespace:
        data["namespace_path"] = namespace
    fork = src.forks.create(data)
    return {
        "id": fork.id,
        "name": fork.name_with_namespace,
        "path": fork.path_with_namespace,
        "web_url": fork.web_url,
        "import_status": getattr(fork, "import_status", None),
    }

@mcp.tool()
def wait_for_fork(fork_id: int, poll_interval: int = 2, timeout: int = 120) -> Dict[str, Any]:
    """
    Pollt den Importstatus des Forks bis 'finished' oder Timeout.
    """
    gl = _gitlab_client()
    start = time.time()
    while time.time() - start < timeout:
        f = gl.projects.get(fork_id)
        status = getattr(f, "import_status", "finished")
        if status in ("finished", "none"):
            return {"status": "finished"}
        time.sleep(poll_interval)
    return {"status": "timeout"}

@mcp.tool()
def clone_issue(
    src_project_ref: str | int,
    dst_project_ref: str | int,
    issue_iid: int,
    copy_labels: bool = True,
) -> Dict[str, Any]:
    """
    Kopiert ein einzelnes Issue vom Quell- ins Zielprojekt.
    """
    gl = _gitlab_client()
    src = _project_from_ref(gl, src_project_ref)
    dst = _project_from_ref(gl, dst_project_ref)
    iss = src.issues.get(issue_iid, iid=True)
    desc = f"(Forked from {src.id}#{iss.iid})\n\n{iss.description or ''}"
    new_issue = dst.issues.create({
        "title": iss.title,
        "description": desc,
        "labels": iss.labels if copy_labels else []
    })
    return {"src_iid": iss.iid, "dst_iid": new_issue.iid, "dst_web_url": new_issue.web_url}

@mcp.tool()
def mirror_issues(
    src_project_ref: str | int,
    dst_project_ref: str | int,
    state: str = "opened",
    limit: Optional[int] = None,
    copy_labels: bool = True,
) -> List[Dict[str, Any]]:
    """
    Spiegelt mehrere Issues (z.B. alle offenen) vom Quellprojekt ins Zielprojekt.
    """
    gl = _gitlab_client()
    src = _project_from_ref(gl, src_project_ref)
    dst = _project_from_ref(gl, dst_project_ref)
    issues = src.issues.list(state=state, get_all=True)
    if limit is not None:
        issues = issues[:limit]
    out = []
    for i in issues:
        desc = f"(Forked from {src.id}#{i.iid})\n\n{i.description or ''}"
        ni = dst.issues.create({
            "title": i.title,
            "description": desc,
            "labels": i.labels if copy_labels else []
        })
        out.append({"src_iid": i.iid, "dst_iid": ni.iid})
    return out

@mcp.tool()
def wait_for_pipeline(
    project_ref: str | int,
    pipeline_id: int,
    timeout_sec: int = 900,
    poll: int = 5,
) -> Dict[str, Any]:
    """
    Wartet auf Abschluss einer Pipeline.
    """
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    start = time.time()
    while time.time() - start < timeout_sec:
        p = proj.pipelines.get(pipeline_id)
        if p.status in ("success", "failed", "canceled", "skipped", "manual"):
            return {"status": p.status}
        time.sleep(poll)
    return {"status": "timeout"}

@mcp.tool()
def get_job_log(project_ref: str | int, job_id: int) -> Dict[str, Any]:
    """
    Holt den Text-Log eines Jobs (falls verfügbar).
    """
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    job = proj.jobs.get(job_id)
    try:
        log_txt = job.trace().decode("utf-8", errors="replace")
    except Exception:
        log_txt = "(kein Log verfügbar)"
    return {"id": job.id, "status": job.status, "name": job.name, "log": log_txt}


# ---------------------- Generisches LLM-Tool ----------------------
@mcp.tool()
def llm_generate(prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
    """
    Einfache Text-Generierung über den in .env gesetzten Provider.
    Kein Tool-Calling, nur reiner Prompt -> Text.
    """
    provider = LLM_PROVIDER
    model = model or DEFAULT_MODEL

    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY fehlt")
        url = GEMINI_URL_TMPL.format(model=model)
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json", "X-Goog-Api-Key": GEMINI_API_KEY}
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return json.dumps(data, ensure_ascii=False)

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY fehlt")
        url = f"{OPENAI_BASE_URL}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return json.dumps(data, ensure_ascii=False)

    elif provider == "custom":
        if not CUSTOM_LLM_BASE_URL:
            raise RuntimeError("CUSTOM_LLM_BASE_URL fehlt")
        url = CUSTOM_LLM_BASE_URL
        payload = {"prompt": prompt, "model": model, "temperature": temperature}
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # Erwartet: {"text": "..."}
        return data.get("text", json.dumps(data, ensure_ascii=False))

    else:
        raise RuntimeError(f"LLM_PROVIDER '{provider}' nicht unterstützt.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
