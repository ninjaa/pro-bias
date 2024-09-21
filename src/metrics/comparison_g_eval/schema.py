from pydantic import BaseModel


class ReasonScore(BaseModel):
    choice: str
    reason: str


class Steps(BaseModel):
    steps: list[str]
