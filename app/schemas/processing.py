from pydantic import BaseModel


class AcceptedResponse(BaseModel):
    status: str = "accepted"


class HealthResponse(BaseModel):
    status: str = "ok"
