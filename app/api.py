from fastapi import APIRouter
from app.models import SimilarityResponse, TextPair
from app.services import NLIService 
from app.embeddingService import EmbeddingService  
nli_service = NLIService()
embedding_service = EmbeddingService() 

router = APIRouter()

@router.post("/similarity", response_model=SimilarityResponse)
async def get_similarity(payload: TextPair):
    similarity_score = embedding_service.compare_texts(payload.text1, payload.text2)
    nli_result = nli_service.check_contradiction(payload.text1, payload.text2)

    return SimilarityResponse(
        similarity=similarity_score,
        label=nli_result["label"],
        confidence=nli_result["confidence"]
    )
