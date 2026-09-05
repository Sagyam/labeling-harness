"""strip speaker pii from episode metadata

The ingest form used to collect a speaker's name and their dialect/origin, and both were written
into ``episodes.metadata_jsonb->'speakers'`` and out into every analytics export. Neither is
collected any more (D56): a name is identity whether or not its owner is a public figure, and a
dialect label the owner cannot apply consistently would be stratified on and would bias every
result computed over it.

New rows are protected by the allowlist in ``app/services/speaker_meta.py``. This revision deals
with rows already written, by rewriting each speaker object down to the same allowlist -- ``role``
and ``gender`` -- and dropping the ``speakers`` key entirely where nothing survives.

**The downgrade cannot restore the names, and does not pretend to.** Deleting the PII is the point
of the revision; a downgrade that recreated it would defeat it, and the data is not kept anywhere
to recreate it from. Downgrading is therefore a no-op, which leaves the schema correct for the
previous revision -- the column shape never changed -- with the personal data still gone.

Revision ID: 9b1c0d4e2a71
Revises: edc4533896c2
Create Date: 2026-09-05 23:20:11.004512

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b1c0d4e2a71"
down_revision: str | Sequence[str] | None = "edc4533896c2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Keep in step with ``app.services.speaker_meta.ALLOWED_SPEAKER_FIELDS``. Spelled out here rather
#: than imported: a migration must keep doing what it did when it was written, even after the
#: application's allowlist changes.
ALLOWED = ("role", "gender")

#: Rebuild each speaker object from the allowlist, then drop speakers that came out empty, then
#: drop the whole `speakers` key if none are left. Rows without a `speakers` object are untouched.
_STRIP = sa.text(
    """
    WITH cleaned AS (
        SELECT
            e.id,
            COALESCE(
                (
                    SELECT jsonb_object_agg(s.key, s.kept)
                    FROM (
                        SELECT
                            spk.key,
                            (
                                SELECT COALESCE(jsonb_object_agg(f.key, f.value), '{}'::jsonb)
                                FROM jsonb_each(spk.value) AS f(key, value)
                                WHERE f.key = ANY(:allowed)
                            ) AS kept
                        FROM jsonb_each(e.metadata_jsonb -> 'speakers') AS spk(key, value)
                        WHERE jsonb_typeof(spk.value) = 'object'
                    ) AS s
                    WHERE s.kept <> '{}'::jsonb
                ),
                '{}'::jsonb
            ) AS speakers
        FROM episodes e
        WHERE e.metadata_jsonb ? 'speakers'
    )
    UPDATE episodes e
    SET metadata_jsonb = CASE
        WHEN c.speakers = '{}'::jsonb THEN e.metadata_jsonb - 'speakers'
        ELSE jsonb_set(e.metadata_jsonb, '{speakers}', c.speakers)
    END
    FROM cleaned c
    WHERE e.id = c.id
    """
)


def upgrade() -> None:
    """Rewrite every stored speaker block down to the allowlist."""
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite has no jsonb_each; the test suite migrates against Postgres, which is the only
        # database the harness stores a corpus in.
        return
    op.execute(_STRIP.bindparams(allowed=list(ALLOWED)))


def downgrade() -> None:
    """No-op: the deleted names are not recoverable, and recreating them is not wanted."""
