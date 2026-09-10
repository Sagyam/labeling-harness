"""gold is chosen per clip

D71 supersedes D63's episode-level pots. ``segments.pot`` (``gold`` / ``train``) replaces
``episodes.pot``: the owner puts individual clips in gold by hand, and a gold clip exports as the
``test`` split whatever its episode is. ``episodes.split`` keeps only ``train`` / ``val`` /
``unassigned`` -- the line that subdivides the clips that are not gold.

Backfill keeps every clip where it was: a clip of a gold episode becomes a gold clip, and a
``test`` episode becomes ``train`` (its clips are now the ones carrying the benchmark). The
downgrade reverses it approximately -- an episode becomes gold if any of its clips is -- which is
lossy only for episodes that were split between pots, and those cannot exist under D63.

Revision ID: d71a0c3e5b10
Revises: c823089f43f1
Create Date: 2026-09-10 16:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d71a0c3e5b10"
down_revision: str | Sequence[str] | None = "c823089f43f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "segments",
        sa.Column("pot", sa.String(length=16), nullable=False, server_default="train"),
    )
    op.execute(
        """
        UPDATE segments s
           SET pot = 'gold'
          FROM episodes e
         WHERE e.id = s.episode_id AND e.pot = 'gold'
        """
    )
    op.create_check_constraint("pot_allowed", "segments", "pot IN ('gold', 'train')")
    op.create_index("ix_segments_pot", "segments", ["pot"])

    op.drop_index("ix_episodes_pot", table_name="episodes")
    op.drop_constraint("pot_matches_split", "episodes", type_="check")
    op.drop_constraint("pot_allowed", "episodes", type_="check")
    op.drop_column("episodes", "pot_assigned_at")
    op.drop_column("episodes", "pot")

    op.drop_constraint("split_allowed", "episodes", type_="check")
    op.execute("UPDATE episodes SET split = 'train' WHERE split = 'test'")
    op.create_check_constraint(
        "split_allowed", "episodes", "split IN ('train', 'val', 'unassigned')"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("split_allowed", "episodes", type_="check")
    op.create_check_constraint(
        "split_allowed", "episodes", "split IN ('train', 'val', 'test', 'unassigned')"
    )

    op.add_column(
        "episodes",
        sa.Column("pot", sa.String(length=16), nullable=False, server_default="unassigned"),
    )
    op.add_column(
        "episodes", sa.Column("pot_assigned_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.execute(
        """
        UPDATE episodes e
           SET pot = CASE
                       WHEN EXISTS (SELECT 1 FROM segments s
                                     WHERE s.episode_id = e.id AND s.pot = 'gold') THEN 'gold'
                       WHEN e.split IN ('train', 'val') THEN 'train'
                       ELSE 'unassigned'
                     END,
               split = CASE
                         WHEN EXISTS (SELECT 1 FROM segments s
                                       WHERE s.episode_id = e.id AND s.pot = 'gold') THEN 'test'
                         ELSE e.split
                       END,
               pot_assigned_at = now()
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

    op.drop_index("ix_segments_pot", table_name="segments")
    op.drop_constraint("pot_allowed", "segments", type_="check")
    op.drop_column("segments", "pot")
