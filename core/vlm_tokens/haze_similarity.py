import torch


class HazeSimilarity:

    @staticmethod
    def compute_similarity(image_features, text_features):
        """
        Computes cosine similarity between one image embedding
        and multiple text embeddings.

        Args:
            image_features : (1, 512)
            text_features  : (N, 512)

        Returns:
            similarities : (N,)
        """

        similarities = image_features @ text_features.T

        return similarities.squeeze(0)