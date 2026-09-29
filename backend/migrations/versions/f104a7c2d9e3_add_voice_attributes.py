"""A voice's gender and age bracket, assigned by the owner from listening (D104).

Revision ID: f104a7c2d9e3
Revises: e99b2d6f4a11
Create Date: 2026-09-29

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f104a7c2d9e3"
down_revision: str | Sequence[str] | None = "e99b2d6f4a11"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "voice_attributes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("voice", sa.String(length=16), nullable=False),
        sa.Column("gender", sa.String(length=16), nullable=True),
        sa.Column("age_bracket", sa.String(length=16), nullable=True),
        sa.Column("annotator", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "gender IS NULL OR gender IN ('female', 'male')",
            name=op.f("ck_voice_attributes_gender_allowed"),
        ),
        sa.CheckConstraint(
            "age_bracket IS NULL OR age_bracket IN "
            "('under_20', '20_39', '40_59', '60_79', '80_plus')",
            name=op.f("ck_voice_attributes_age_bracket_allowed"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_voice_attributes")),
    )
    op.create_index(op.f("ix_voice_attributes_voice"), "voice_attributes", ["voice"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_voice_attributes_voice"), table_name="voice_attributes")
    op.drop_table("voice_attributes")
