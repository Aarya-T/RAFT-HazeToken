import torch


class GlobalHazeScore:

    @staticmethod
    def compute(similarities):
        """
        similarities:
            Tensor of length 8

        First 4 values:
            Haze prompts

        Last 4 values:
            Clear prompts

        Returns:
            global haze score (0-1)
        """

        haze_scores = similarities[:4]
        clear_scores = similarities[4:]

        mean_haze = haze_scores.mean()
        mean_clear = clear_scores.mean()

        score = torch.sigmoid(mean_haze - mean_clear)

        return score.item()