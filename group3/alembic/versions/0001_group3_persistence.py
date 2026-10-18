"""Create Group 3 persistence tables."""
from alembic import op
import sqlalchemy as sa

revision = "0001_group3_persistence"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    payload = sa.JSON()
    op.create_table("group3_users", sa.Column("id", sa.String(length=36), primary_key=True), sa.Column("payload", payload, nullable=False))
    op.create_table("group3_projects", sa.Column("id", sa.String(length=36), primary_key=True), sa.Column("payload", payload, nullable=False))
    op.create_table("group3_audit_events", sa.Column("id", sa.String(length=36), primary_key=True), sa.Column("payload", payload, nullable=False))

def downgrade():
    op.drop_table("group3_audit_events")
    op.drop_table("group3_projects")
    op.drop_table("group3_users")
