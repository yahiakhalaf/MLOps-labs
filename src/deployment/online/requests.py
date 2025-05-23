from pydantic import BaseModel
from typing import Optional

class InferenceRequest(BaseModel):
    PassengerId: int
    Pclass: int
    Name: str
    Sex: str
    Age: Optional[float] 
    SibSp: int
    Parch: int
    Ticket: str
    Fare: Optional[float] 
    Cabin: Optional[str] 
    Embarked: str
