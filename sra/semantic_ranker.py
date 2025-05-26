from sentence_transformers import SentenceTransformer, util


class SemanticRanker:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def rank_resumes(self, resumes: list[str], job_description: str) -> list[tuple[str, float]]:
        job_embedding = self.model.encode(job_description, convert_to_tensor=True)
        resume_embeddings = self.model.encode(resumes, convert_to_tensor=True)
        scores = util.cos_sim(job_embedding, resume_embeddings)[0]
        ranked = sorted(zip(resumes, scores, strict=False), key=lambda x: x[1], reverse=True)
        return [(res, float(score)) for res, score in ranked]
