"""The folding rulebook as a page: every rule and tag of ``fold.py``, with its evidence.

The rules are code -- a change to one is a fold version, a commit and a decision entry, never an
edit on a page -- so this is read-only. What it adds to the code is the evidence: for every
model's error files (mine-v3), how many pairs each rule forgave and each tag tagged, per set.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_config, get_session, require_auth
from app.config import Settings
from app.models import AsrModel
from app.services import error_store
from app.services.fold import RULEBOOK, TAGS, TIERS, FoldRule, fold_version

router = APIRouter(tags=["fold"], dependencies=[Depends(require_auth)])


def _rule(rule: FoldRule, action: str) -> dict[str, Any]:
    return asdict(rule) | {"action": action}


@router.get("/fold/rulebook")
def get_rulebook(
    session: Session = Depends(get_session), settings: Settings = Depends(get_config)
) -> dict[str, Any]:
    """Every rule and tag, in reading order, and per model and set what each one covered.

    A file derived under another fold version than this harness's is still listed, marked
    ``current: false``: its counts are what an older rulebook did.
    """
    current = fold_version()
    evidence = []
    for model in session.scalars(sa.select(AsrModel).order_by(AsrModel.slug)):
        for found in error_store.list_files(settings.models.root / model.slug)[0]:
            evidence.append(
                {
                    "model": model.slug,
                    "model_name": model.name,
                    "run": found.run,
                    "set": found.set,
                    "fold_version": found.fold_version,
                    "current": found.fold_version == current,
                }
                | error_store.rule_evidence(found.path)
            )
    return {
        "fold_version": current,
        "tiers": [{"tier": tier, "title": title} for tier, title in TIERS.items()],
        "rules": [_rule(r, "fold") for r in RULEBOOK] + [_rule(t, "tag") for t in TAGS],
        "evidence": evidence,
    }
