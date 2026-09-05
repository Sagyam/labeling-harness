"""add episode audio object key

Where the whole episode's normalised 16 kHz mono FLAC lives in object storage. Until now only the
clips cut out of it were kept and the full recording was deleted with the ingest job's work
directory, which meant nothing downstream could ever look at the episode as a whole.

That is what diarization needs. D58 removed the in-pipeline clustering stage: a diarizer good
enough to define a grouping variable is not something to run inside an ingest, and a better one
should be able to run later without re-ingesting anything. Retaining the audio is what makes that
possible -- pyannote 3.1 and its successors take a recording, not a bag of clips.

Nullable, because every episode imported before this has no such file and there is nothing to
point at.

Revision ID: be40855f6bca
Revises: 9b1c0d4e2a71
Create Date: 2026-09-06 00:52:56.477872

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "be40855f6bca"
down_revision: str | Sequence[str] | None = "9b1c0d4e2a71"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("episodes", sa.Column("audio_object_key", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("episodes", "audio_object_key")
