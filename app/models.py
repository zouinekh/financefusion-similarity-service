from pydantic import BaseModel

class TextPair(BaseModel):
    text1: str
    text2: str

class SimilarityResponse(BaseModel):
    similarity: float
    label: str
    confidence: float
class SimilarityRequest(BaseModel):
    text1: str
    text2: str