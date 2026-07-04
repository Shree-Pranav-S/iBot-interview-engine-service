"""Base Pydantic schema configuration."""

from pydantic import BaseModel, ConfigDict


class AppBaseModel(BaseModel):
    """Base for request and plain-object schemas.

    Forbids extra fields so unknown payload keys surface as validation errors.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )
