"""PLACEHOLDER project-level authorization.

The real platform will supply authenticated principals (likely Group 3).
`authorize_project` is the single check every service entry point and every
tool call goes through, so swapping in the real implementation is one place.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AuthorizationError(PermissionError):
    pass


class Principal(BaseModel):
    user_id: str
    project_ids: list[str] = Field(default_factory=list)  # "*" = all (dev only)
    roles: list[str] = Field(default_factory=lambda: ["consultant"])

    def can_access(self, project_id: str) -> bool:
        return "*" in self.project_ids or project_id in self.project_ids

    def can_approve(self) -> bool:
        return bool({"approver", "engagement_lead", "admin"} & set(self.roles))


def authorize_project(principal: Principal, project_id: str, action: str = "read") -> None:
    if not principal.can_access(project_id):
        raise AuthorizationError(f"user '{principal.user_id}' may not {action} project '{project_id}'")
    if action == "approve" and not principal.can_approve():
        raise AuthorizationError(f"user '{principal.user_id}' lacks approver role")


def dev_principal(user_id: str = "dev-user", allowed: str = "*") -> Principal:
    """DEV ONLY: principal with configurable project access and approver role."""
    projects = [p.strip() for p in allowed.split(",") if p.strip()] or ["*"]
    return Principal(user_id=user_id, project_ids=projects, roles=["consultant", "approver"])
