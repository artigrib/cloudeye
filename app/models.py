"""SQLAlchemy 2.0 ORM models."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    BigInteger,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Video(Base):
    """A single uploaded video and its extracted metadata."""

    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    filepath: Mapped[str] = mapped_column(String, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    format: Mapped[str] = mapped_column(String, nullable=False)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="uploaded")
    # Per-upload overrides for gpu/stage_keyframes.py's keyframe filter. NULL means "use
    # that stage's own default" (blur_threshold=100.0, similarity_threshold=0.98) - most
    # uploads never set these; they exist for footage the defaults misjudge (e.g. sharp
    # but low-texture rooms that the Laplacian-variance blur check reads as blurry).
    blur_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    similarity_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Per-upload fixed vocabulary for gpu/stage_vocab.py, skipping its OpenRouter call
    # entirely - for reproducing a stable vocabulary across repeated runs of the same
    # footage, given that call's own measured run-to-run non-determinism (see
    # docs/COMPARISON.md). NULL/empty means "use that stage's own OpenRouter call and
    # default-vocabulary fallback", unchanged. Same entry shape build_vocab_entries()
    # in gpu/stage_vocab.py already accepts (a name string, or a {"name", "size_class"}
    # dict) - not re-validated here beyond "is it a JSON array".
    vocab_override: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    # Which scene-vocabulary provider to use for this video - "openrouter"/"vertex", or
    # NULL to use the default (see app/services/model_catalog.DEFAULT_VOCAB_PROVIDER).
    # Independent of vocab_override above: this picks which provider makes the call,
    # vocab_override skips the call entirely regardless of which provider was picked.
    vocab_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    project: Mapped["Project | None"] = relationship(
        back_populates="videos", foreign_keys=[project_id], passive_deletes=True
    )
    scene: Mapped["Scene | None"] = relationship(
        back_populates="video", passive_deletes=True, uselist=False
    )


class Project(Base):
    """A room: groups one or more independently-reconstructed video scenes."""

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    # Circular FK with scenes.project_id: use_alter defers this constraint until both
    # tables exist, or table creation / migration ordering deadlocks.
    primary_scene_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("scenes.id", ondelete="SET NULL", use_alter=True, name="fk_projects_primary_scene"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
    # Hides a project from the default project list without touching its scenes/videos -
    # reversible, deletes nothing. See list_projects()'s include_archived param.
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    videos: Mapped[list["Video"]] = relationship(
        back_populates="project",
        foreign_keys=[Video.project_id],
        passive_deletes=True,
        cascade="all, delete-orphan",
    )
    scenes: Mapped[list["Scene"]] = relationship(
        back_populates="project",
        foreign_keys="Scene.project_id",
        passive_deletes=True,
        cascade="all, delete-orphan",
    )


class Scene(Base):
    """A single 3D reconstruction produced from one video's GPU pipeline run."""

    __tablename__ = "scenes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # DB-level half of worker idempotency: one scene per video, always.
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label: Mapped[str | None] = mapped_column(String, nullable=True)  # e.g. "horizontal", "vertical"

    status: Mapped[str] = mapped_column(String, nullable=False, default="queued")
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)

    mesh_path: Mapped[str | None] = mapped_column(String, nullable=True)
    pointcloud_path: Mapped[str | None] = mapped_column(String, nullable=True)
    occupancy_path: Mapped[str | None] = mapped_column(String, nullable=True)
    map_preview_path: Mapped[str | None] = mapped_column(String, nullable=True)
    transform_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # Path to the most recently generated Isaac Sim USD export (see usd_export.py). One
    # scene can have several cached variants on disk (per include_pointcloud/
    # include_fragments) - this just points at the latest one generated, for
    # observability; GET /{id}/usd resolves cache hits by filename, not this field.
    usd_path: Mapped[str | None] = mapped_column(String, nullable=True)

    grid_resolution: Mapped[float | None] = mapped_column(Float, nullable=True)
    grid_origin_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    grid_origin_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    grid_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    grid_height: Mapped[int | None] = mapped_column(Integer, nullable=True)

    floor_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    ceiling_y: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Not in the original field list, but required by the validation spec ("флаг сохранить
    # в scenes") to persist whether the alignment transform's R is a reflection (det=-1),
    # which matters for future mesh normal/winding-order handling.
    is_reflected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Below-floor phantom-point clip diagnostics from stage_align.py (see setup-log.md's
    # "reflective-floor phantom points" entry). NULL/False for scenes processed before
    # this field existed, not "known clean".
    points_below_floor_frac: Mapped[float | None] = mapped_column(Float, nullable=True)
    reflective_floor_suspected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Non-fatal object_max_extent check's flagged objects (scene_validator.py), joined
    # into one ready-to-display string, e.g. "bed_0 (2.92m)" - NULL if none. Doesn't
    # block the scene; surfaced so an oversized/likely-merged object doesn't ship
    # silently (see OBJECT_MAX_EXTENT_M's comment for the confirmed real case this
    # catches).
    oversized_object_warning: Mapped[str | None] = mapped_column(String, nullable=True)

    # vocab.json's "source" field (stage_vocab.py) - "openrouter"/"vertex"/"override"
    # going forward; a provider failure now fails the whole scene rather than falling
    # back to a substitute vocabulary (VocabProviderError, no vocab.json written at
    # all), so a "done" scene's vocabulary is always real. Older scenes can still carry
    # the retired "default_no_key"/"default_no_frames"/"default_after_error" values
    # from before that change. NULL for scenes processed before this field existed at
    # all, or where vocab.json was never written for some other reason.
    vocab_source: Mapped[str | None] = mapped_column(String, nullable=True)
    # The exact model string vocab.json's stage actually used (e.g.
    # "google/gemma-4-31b-it") - recorded at run time rather than inferred later from
    # current config, so it stays correct even after the default model changes.
    vocab_model: Mapped[str | None] = mapped_column(String, nullable=True)
    # This scene's own resolved vocabulary - vocab.json's "objects" array, reduced to
    # just {"name", "size_class"} per entry (see scene_ingest.read_vocab_entries()) -
    # persisted so a later `rerun` (pipeline_orchestrator.resolve_vocab_override) can
    # reuse it as the next run's vocab_override instead of making a fresh
    # OpenRouter/Vertex call. Without this, every rerun would get its own
    # independently-sampled vocabulary (see gpu/stage_vocab.py's docstring on
    # run-to-run non-determinism), making repeated runs of the same video
    # incomparable. NULL for scenes processed before this field existed, or where
    # vocab.json was never written at all (see vocab_source's docstring).
    vocab_entries: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    robot_start_x: Mapped[float | None] = mapped_column(Float, nullable=True)
    robot_start_z: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    video: Mapped["Video"] = relationship(back_populates="scene", passive_deletes=True)
    project: Mapped["Project"] = relationship(
        back_populates="scenes", foreign_keys=[project_id], passive_deletes=True
    )
    objects: Mapped[list["SceneObject"]] = relationship(
        back_populates="scene", passive_deletes=True, cascade="all, delete-orphan"
    )
    commands: Mapped[list["Command"]] = relationship(
        back_populates="scene", passive_deletes=True, cascade="all, delete-orphan"
    )


class SceneObject(Base):
    """A single object detected in a scene, with its 3D position and bounding box."""

    __tablename__ = "scene_objects"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # Always NULL from the current pipeline — no stage produces a free-text description
    # (SAM3 only segments, the vocabulary LLM call only names). Kept nullable for a future
    # captioning stage.
    description: Mapped[str | None] = mapped_column(String, nullable=True)

    pos_x: Mapped[float] = mapped_column(Float, nullable=False)
    pos_y: Mapped[float] = mapped_column(Float, nullable=False)
    pos_z: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_min_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_min_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_min_z: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_max_x: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_max_y: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_max_z: Mapped[float] = mapped_column(Float, nullable=False)

    num_views: Mapped[int] = mapped_column(Integer, nullable=False)
    num_points: Mapped[int] = mapped_column(Integer, nullable=False)
    is_fragment: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Currently a colored point-cloud .ply path, not a triangulated mesh — no meshing
    # stage exists yet. Named mesh_path per spec (future Isaac export target).
    mesh_path: Mapped[str | None] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    scene: Mapped["Scene"] = relationship(back_populates="objects", passive_deletes=True)


class Command(Base):
    """A single natural-language robot command issued against a scene, and its result."""

    __tablename__ = "commands"
    __table_args__ = (Index("ix_commands_scene_id_created_at", "scene_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scene_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False
    )
    user_text: Mapped[str] = mapped_column(String, nullable=False)
    parsed_action: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    scene: Mapped["Scene"] = relationship(back_populates="commands", passive_deletes=True)
