"""add pots and verification tier

Two pots replace the hashed split (D63).

``episodes.pot`` is the frozen commitment -- ``gold`` or ``train`` -- assigned to a whole episode
against a duration target *before* any of its clips is annotated. The old scheme hashed the episode
id into train/val/test, which could express a ratio but never an amount: there was no way to ask it
for five hours of benchmark audio, and a ratio over episode *count* says nothing about duration
when episodes run from minutes to four hours. It also stratified nothing.

``episodes.split`` survives as the derived label, and a CHECK ties the two together: a gold-pot
episode is always ``test``, a train-pot episode is always ``train`` or ``val``. Gold is
one-directional -- an episode never leaves it -- because a benchmark recording that later turns up
in training invalidates every number measured against it, silently. Train and val may be redrawn
freely; they hold data of the same standard.

``segment_labels.verification_tier`` records how much attention a decision actually got:
``verified`` (clip played, transcript read) or ``screened`` (waved through on the cross-ASR
disagreement signal without listening). Both are legitimate ways to build a corpus and they are not
the same claim. Without the column every screened row would export as a human verification it never
was, and the disagreement gate could never be measured, because its own decision would have been
overwritten by a confirmation nobody made.

Backfill is deliberate on both columns. Existing episodes take their pot from the split they
already have, so no episode moves. Existing labels become ``verified``: they were written by the
one-clip-at-a-time flow that predates screening, which is exactly what ``verified`` means.

Revision ID: c823089f43f1
Revises: be40855f6bca
Create Date: 2026-09-06 01:24:11.309442

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c823089f43f1"
down_revision: str | Sequence[str] | None = "be40855f6bca"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "episodes",
        sa.Column("pot", sa.String(length=16), nullable=False, server_default="unassigned"),
    )
    op.add_column(
        "episodes", sa.Column("pot_assigned_at", sa.DateTime(timezone=True), nullable=True)
    )

    # Derive the pot from the split every existing episode already has, so nothing moves. This runs
    # before the CHECK is added, because the CHECK is what it has to satisfy.
    op.execute(
        """
        UPDATE episodes
           SET pot = CASE
                       WHEN split = 'test' THEN 'gold'
                       WHEN split IN ('train', 'val') THEN 'train'
                       ELSE 'unassigned'
                     END,
               pot_assigned_at = COALESCE(split_assigned_at, now())
         WHERE pot = 'unassigned'
        """
    )

    op.create_check_constraint("pot_allowed", "episodes", "pot IN ('gold', 'train', 'unassigned')")
    op.create_check_constraint(
        "pot_matches_split",
        "episodes",
        "(pot = 'gold' AND split = 'test') OR (pot = 'train' AND split IN ('train', 'val'))"
        " OR pot = 'unassigned'",
    )
    op.create_index("ix_episodes_pot", "episodes", ["pot"])

    op.add_column(
        "segment_labels",
        sa.Column(
            "verification_tier",
            sa.String(length=16),
            nullable=False,
            server_default="verified",
        ),
    )
    op.create_check_constraint(
        "verification_tier_allowed",
        "segment_labels",
        "verification_tier IN ('verified', 'screened')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("verification_tier_allowed", "segment_labels", type_="check")
    op.drop_column("segment_labels", "verification_tier")

    op.drop_index("ix_episodes_pot", table_name="episodes")
    op.drop_constraint("pot_matches_split", "episodes", type_="check")
    op.drop_constraint("pot_allowed", "episodes", type_="check")
    op.drop_column("episodes", "pot_assigned_at")
    op.drop_column("episodes", "pot")
