"""Base Pydantic schema configuration."""

from pydantic import BaseModel, ConfigDict


class AppBaseModel(BaseModel):
    """
    Base for all request / plain-object schemas.
    Forbids extra fields so unknown payload keys surface as validation errors.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ORMBaseModel(BaseModel):
    """
    Base for all response schemas that are constructed from ORM objects.
    Enables from_attributes mode so SQLAlchemy
    model instances can be passed directly to model_validate().
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
    )
