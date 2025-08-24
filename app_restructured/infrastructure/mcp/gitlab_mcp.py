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
import re

import gitlab
from gitlab import exceptions as gl_exc
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from app_restructured.core.config import (
    LLM_PROVIDER,
    DEFAULT_MODEL,
    GEMINI_API_KEY,
    GEMINI_URL_TMPL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    CUSTOM_LLM_BASE_URL,
    OLLAMA_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
)

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

# gitlab_mcp.py
from gitlab import exceptions as gl_exc

@mcp.tool()
def merge_merge_request(
    project_ref: str,
    mr_iid: int,
    auto_merge: bool | None = None,                 # GitLab ≥17.11
    merge_when_pipeline_succeeds: bool = False,     # Fallback für ältere Server
    squash: bool = False,
    should_remove_source_branch: bool = False,
    merge_commit_message: str | None = None,
    squash_commit_message: str | None = None,
    sha: str | None = None,
) -> dict:
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    mr = proj.mergerequests.get(mr_iid, iid=True)

    payload: dict[str, object] = {}
    if auto_merge is not None:
        payload["auto_merge"] = auto_merge
    else:
        payload["merge_when_pipeline_succeeds"] = merge_when_pipeline_succeeds
    if squash:
        payload["squash"] = True
    if should_remove_source_branch:
        payload["should_remove_source_branch"] = True
    if sha:
        payload["sha"] = sha
    if merge_commit_message and merge_commit_message.strip():
        payload["merge_commit_message"] = merge_commit_message.strip()
    if squash and squash_commit_message and squash_commit_message.strip():
        payload["squash_commit_message"] = squash_commit_message.strip()

    def _do_merge(data: dict):
        mr.merge(**data)  # python-gitlab macht ein PUT /merge mit diesen kwargs

    try:
        _do_merge(payload)
    except gl_exc.GitlabError as e:
        if "merge_commit_message is invalid" in str(e):
            # Retry ohne Message
            payload.pop("merge_commit_message", None)
            try:
                _do_merge(payload)
            except gl_exc.GitlabError:
                # Letzter Versuch mit einer minimalistischen Fallback-Message
                payload["merge_commit_message"] = (
                    f"Merge {mr.source_branch} into {mr.target_branch} – {mr.title}"
                )
                _do_merge(payload)
        else:
            raise

    # <-- Kein mr.refresh() mehr!
    mr = proj.mergerequests.get(mr_iid, iid=True)  # Zustand neu laden

    return {
        "iid": mr.iid,
        "state": mr.state,
        "merged_at": getattr(mr, "merged_at", None),
        "web_url": mr.web_url,
        "merge_commit_sha": getattr(mr, "merge_commit_sha", None),
    }

@mcp.tool()
def delete_merge_request(project_ref: str, mr_iid: int) -> Dict[str, Any]:
    """
    Löscht einen Merge Request (soft delete laut GitLab-API).
    Voraussetzungen:
      - Nur Admins oder Projekt-Owner dürfen löschen (API-Restriktion).
    Rückgabe: einfache Bestätigung + Web-URL falls noch verfügbar.
    """
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    try:
        mr = proj.mergerequests.get(mr_iid, iid=True)
    except gl_exc.GitlabGetError as e:
        # Bereits weg oder falsche IID
        raise ValueError(f"Merge Request {mr_iid} nicht gefunden: {e}") from e

    try:
        mr.delete()  # python-gitlab Wrapper für DELETE /merge_requests/:iid
    except gl_exc.GitlabError as e:
        # z.B. 403 (keine Rechte)
        raise RuntimeError(f"Löschen fehlgeschlagen: {e}") from e

    return {"iid": mr_iid, "deleted": True}


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

# ============================================================================
# ENHANCED ISSUE MANAGEMENT TOOLS
# ============================================================================

@mcp.tool()
def update_issue_state(project_ref: str, issue_iid: int, state_event: str) -> Dict[str, Any]:
    """Update issue state. state_event can be 'close' or 'reopen'"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issue = proj.issues.get(issue_iid, iid=True)
    issue.state_event = state_event
    issue.save()
    return {
        "iid": issue.iid,
        "title": issue.title,
        "state": issue.state,
        "web_url": issue.web_url
    }

@mcp.tool()
def close_issue(project_ref: str, issue_iid: int) -> Dict[str, Any]:
    """Close an issue"""
    return update_issue_state(project_ref, issue_iid, "close")

@mcp.tool()
def reopen_issue(project_ref: str, issue_iid: int) -> Dict[str, Any]:
    """Reopen a closed issue"""
    return update_issue_state(project_ref, issue_iid, "reopen")

@mcp.tool()
def update_issue(
    project_ref: str, 
    issue_iid: int, 
    title: Optional[str] = None,
    description: Optional[str] = None,
    labels: Optional[List[str]] = None,
    assignee_ids: Optional[List[int]] = None,
    milestone_id: Optional[int] = None,
    due_date: Optional[str] = None,
    weight: Optional[int] = None
) -> Dict[str, Any]:
    """Update multiple issue fields at once"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issue = proj.issues.get(issue_iid, iid=True)
    
    updates = {}
    if title is not None:
        updates["title"] = title
    if description is not None:
        updates["description"] = description
    if labels is not None:
        updates["labels"] = labels
    if assignee_ids is not None:
        updates["assignee_ids"] = assignee_ids
    if milestone_id is not None:
        updates["milestone_id"] = milestone_id
    if due_date is not None:
        updates["due_date"] = due_date
    if weight is not None:
        updates["weight"] = weight
    
    for key, value in updates.items():
        setattr(issue, key, value)
    issue.save()
    
    return {
        "iid": issue.iid,
        "title": issue.title,
        "state": issue.state,
        "labels": getattr(issue, 'labels', []),
        "assignees": [{"id": a.id, "name": a.name} for a in getattr(issue, 'assignees', [])],
        "milestone": getattr(issue.milestone, 'title', None) if hasattr(issue, 'milestone') and issue.milestone else None,
        "due_date": getattr(issue, 'due_date', None),
        "weight": getattr(issue, 'weight', None),
        "web_url": issue.web_url
    }

@mcp.tool()
def assign_issue(project_ref: str, issue_iid: int, assignee_ids: List[int]) -> Dict[str, Any]:
    """Assign users to an issue"""
    return update_issue(project_ref, issue_iid, assignee_ids=assignee_ids)

@mcp.tool()
def add_issue_labels(project_ref: str, issue_iid: int, labels: List[str]) -> Dict[str, Any]:
    """Add labels to an issue (preserves existing labels)"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issue = proj.issues.get(issue_iid, iid=True)
    
    current_labels = getattr(issue, 'labels', [])
    new_labels = list(set(current_labels + labels))  # Merge and deduplicate
    
    return update_issue(project_ref, issue_iid, labels=new_labels)

@mcp.tool()
def remove_issue_labels(project_ref: str, issue_iid: int, labels: List[str]) -> Dict[str, Any]:
    """Remove specific labels from an issue"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issue = proj.issues.get(issue_iid, iid=True)
    
    current_labels = getattr(issue, 'labels', [])
    new_labels = [label for label in current_labels if label not in labels]
    
    return update_issue(project_ref, issue_iid, labels=new_labels)

@mcp.tool()
def set_issue_milestone(project_ref: str, issue_iid: int, milestone_id: Optional[int]) -> Dict[str, Any]:
    """Set or remove milestone from an issue (None to remove)"""
    return update_issue(project_ref, issue_iid, milestone_id=milestone_id)

@mcp.tool()
def set_issue_due_date(project_ref: str, issue_iid: int, due_date: Optional[str]) -> Dict[str, Any]:
    """Set or remove due date from an issue (format: YYYY-MM-DD, None to remove)"""
    return update_issue(project_ref, issue_iid, due_date=due_date)

@mcp.tool()
def create_branch_from_issue(project_ref: str, issue_iid: int, branch_name: Optional[str] = None) -> Dict[str, Any]:
    """Create a new branch from an issue (auto-generates name if not provided)"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issue = proj.issues.get(issue_iid, iid=True)
    
    if not branch_name:
        # Auto-generate branch name from issue
        safe_title = re.sub(r'[^a-zA-Z0-9\-_]', '-', issue.title.lower())
        safe_title = re.sub(r'-+', '-', safe_title).strip('-')
        branch_name = f"issue-{issue.iid}-{safe_title}"
    
    # Get default branch
    default_branch = proj.default_branch or "main"
    
    # Create branch
    branch = proj.branches.create({"branch": branch_name, "ref": default_branch})
    
    return {
        "name": branch.name,
        "commit": {"id": branch.commit["id"], "message": branch.commit["message"]},
        "issue_iid": issue.iid,
        "issue_title": issue.title,
        "web_url": f"{proj.web_url}/-/tree/{branch_name}"
    }

@mcp.tool()
def list_project_labels(project_ref: str) -> List[Dict[str, Any]]:
    """List all available labels in the project"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    labels = proj.labels.list(all=True)
    return [{
        "id": l.id,
        "name": l.name,
        "color": l.color,
        "description": getattr(l, 'description', ''),
        "subscribed": getattr(l, 'subscribed', False)
    } for l in labels]

@mcp.tool()
def list_project_milestones(project_ref: str, state: str = "active") -> List[Dict[str, Any]]:
    """List project milestones. state can be 'active', 'closed', or 'all'"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    milestones = proj.milestones.list(state=state, all=True)
    return [{
        "id": m.id,
        "iid": m.iid,
        "title": m.title,
        "description": getattr(m, 'description', ''),
        "state": m.state,
        "due_date": getattr(m, 'due_date', None),
        "web_url": getattr(m, 'web_url', f"{proj.web_url}/-/milestones/{m.iid}")
    } for m in milestones]

@mcp.tool()
def create_milestone(
    project_ref: str, 
    title: str, 
    description: str = "", 
    due_date: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new milestone (due_date format: YYYY-MM-DD)"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    
    milestone_data = {"title": title, "description": description}
    if due_date:
        milestone_data["due_date"] = due_date
    
    milestone = proj.milestones.create(milestone_data)
    return {
        "id": milestone.id,
        "iid": milestone.iid,
        "title": milestone.title,
        "description": milestone.description,
        "state": milestone.state,
        "due_date": getattr(milestone, 'due_date', None),
        "web_url": getattr(milestone, 'web_url', f"{proj.web_url}/-/milestones/{milestone.iid}")
    }

@mcp.tool()
def list_project_members(project_ref: str) -> List[Dict[str, Any]]:
    """List project members (for assigning issues)"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    members = proj.members.list(all=True)
    return [{
        "id": m.id,
        "username": m.username,
        "name": m.name,
        "access_level": m.access_level,
        "web_url": getattr(m, 'web_url', f"https://gitlab.com/{m.username}")
    } for m in members]

@mcp.tool()
def search_issues(
    project_ref: str, 
    search_query: str, 
    state: str = "opened",
    labels: Optional[List[str]] = None,
    assignee_username: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Advanced issue search with filters"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    
    search_params = {
        "search": search_query,
        "state": state,
        "per_page": limit
    }
    
    if labels:
        search_params["labels"] = ",".join(labels)
    if assignee_username:
        search_params["assignee_username"] = assignee_username
    
    issues = proj.issues.list(**search_params)
    return [{
        "iid": i.iid,
        "title": i.title,
        "state": i.state,
        "labels": getattr(i, 'labels', []),
        "assignees": [{"username": a.username, "name": a.name} for a in getattr(i, 'assignees', [])],
        "milestone": getattr(i.milestone, 'title', None) if hasattr(i, 'milestone') and i.milestone else None,
        "due_date": getattr(i, 'due_date', None),
        "web_url": i.web_url
    } for i in issues]

@mcp.tool()
def link_issue_to_merge_request(
    project_ref: str, 
    issue_iid: int, 
    mr_iid: int, 
    link_type: str = "closes"
) -> Dict[str, Any]:
    """Link an issue to a merge request. link_type: 'closes', 'fixes', 'resolves', 'implements'"""
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    
    # Get the MR and add a comment linking to the issue
    mr = proj.mergerequests.get(mr_iid, iid=True)
    link_text = f"{link_type.capitalize()} #{issue_iid}"
    
    # Add the link as a comment
    note = mr.notes.create({"body": link_text})
    
    return {
        "issue_iid": issue_iid,
        "mr_iid": mr_iid,
        "link_type": link_type,
        "note_id": note.id,
        "message": f"Linked issue #{issue_iid} to MR !{mr_iid} with relationship '{link_type}'"
    }

# --- Issue-Details & Kommentare ------------------------------------------------
@mcp.tool()
def get_issue(
    project_ref: str,
    issue_iid: int,
    include_notes: bool = True,
    include_discussions: bool = False,
    notes_order_by: str = "created_at",
    notes_sort: str = "asc",   # asc|desc (GitLab default ist desc)
    notes_limit: int = 200,
) -> Dict[str, Any]:
    """
    Liefert alle Details eines Issues (Titel, Beschreibung, Labels, usw.).
    Optional: Kommentare (Notes) und/oder Discussions.
    """
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issue = proj.issues.get(issue_iid, iid=True)

    out: Dict[str, Any] = {
        "iid": issue.iid,
        "id": issue.id,
        "title": issue.title,
        "state": issue.state,
        "description": issue.description,
        "author": getattr(issue, "author", None),
        "assignees": getattr(issue, "assignees", []),
        "labels": issue.labels,
        "milestone": getattr(issue, "milestone", None),
        "created_at": issue.created_at,
        "updated_at": issue.updated_at,
        "web_url": issue.web_url,
    }

    if include_notes:
        notes = issue.notes.list(
            get_all=True,
            order_by=notes_order_by,
            sort=notes_sort,
        )
        out["notes"] = [{
            "id": n.id,
            "body": n.body,
            "author": getattr(n, "author", None),
            "created_at": n.created_at,
            "updated_at": n.updated_at,
            "system": getattr(n, "system", False),
        } for n in notes][:notes_limit]

    if include_discussions:
        discs = issue.discussions.list(get_all=True)
        out["discussions"] = []
        for d in discs:
            out["discussions"].append({
                "id": d.id,
                "individual_note": getattr(d, "individual_note", False),
                "notes": [{
                    "id": nn.id,
                    "body": nn.body,
                    "author": getattr(nn, "author", None),
                    "created_at": nn.created_at,
                    "updated_at": nn.updated_at,
                    "system": getattr(nn, "system", False),
                } for nn in getattr(d, "notes", [])]
            })
    return out


@mcp.tool()
def list_issue_notes(
    project_ref: str,
    issue_iid: int,
    order_by: str = "created_at",
    sort: str = "desc",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Gibt nur die Notes (Kommentare) eines Issues zurück – leichter & schneller als get_issue(include_notes=True).
    """
    gl = _gitlab_client()
    proj = _project_from_ref(gl, project_ref)
    issue = proj.issues.get(issue_iid, iid=True)
    notes = issue.notes.list(get_all=True, order_by=order_by, sort=sort)
    return [{
        "id": n.id,
        "body": n.body,
        "author": getattr(n, "author", None),
        "created_at": n.created_at,
        "updated_at": n.updated_at,
        "system": getattr(n, "system", False),
    } for n in notes][:limit]


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
def llm_generate(prompt: str,
                 model: str | None = None,
                 temperature: float = 0.2) -> str:
    """
    Einfaches Prompt->Text Tool ohne Tool-Calling.
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

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY fehlt")
        url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {OPENAI_API_KEY}"}
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return json.dumps(data, ensure_ascii=False)

    if provider == "ollama":
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": model or OLLAMA_DEFAULT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        # /api/generate streams normally; non-stream returns 'response'
        return data.get("response", json.dumps(data, ensure_ascii=False))

    if provider == "custom":
        if not CUSTOM_LLM_BASE_URL:
            raise RuntimeError("CUSTOM_LLM_BASE_URL fehlt")
        url = CUSTOM_LLM_BASE_URL
        payload = {"prompt": prompt, "model": model, "temperature": temperature}
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", json.dumps(data, ensure_ascii=False))

    raise RuntimeError(f"LLM_PROVIDER '{provider}' nicht unterstützt.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
